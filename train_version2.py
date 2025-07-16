import argparse
import os
from datetime import datetime
from collections import namedtuple

import torch
from torch.nn.utils import clip_grad_norm_
from torch import nn
from torchinfo import summary

import torchvision.datasets as dset
from torchvision.transforms import (CenterCrop, Compose,
                                    ToTensor, RandAugment)
from timm.data.mixup import Mixup
from timm.data.random_erasing import RandomErasing

from sklearn.metrics import classification_report
from tqdm.auto import tqdm

from model import *
from utils import plot_losses, CustomLogger


convnextv2_models = {
    "convnextv2_atto": convnextv2_atto,
    "convnextv2_simple": convnextv2_simple,
    "convnextv2_femto": convnextv2_femto,
}

dartsnext_models = {
    "dartsnext_atto": dartsnext_atto,
    "dartsnext_simple": dartsnext_simple,
}

def create_genotype(block_config):
    block_fields = ', '.join([f'block_{idx}' for idx in range(len(block_config))])
    return namedtuple('Genotype', block_fields)

def main(args):
    mixup_args = {
        'mixup_alpha': 0.8,
        'cutmix_alpha': 1.0,
        'cutmix_minmax': None,
        'prob': 0.4,
        'switch_prob': 0.5,
        'mode': 'elem',
        'label_smoothing': 0.1,
        'num_classes': args.num_classes
    }

    mixup = Mixup(**mixup_args)
    rand_erasing = RandomErasing(probability=0.25, max_area=1/4, mode="pixel")

    # CIFAR10
    if args.model_name in convnextv2_models:
        model = convnextv2_models[args.model_name](dims=args.dims, num_classes=args.num_classes, patch_size=args.patch_size)

    if args.model_name in dartsnext_models:
        block_config = [2, 2, 6, 2]
        Genotype = create_genotype(block_config)
        test_genotype = Genotype(block_0=[('dw_conv_7x7', 0), ('dw_conv_7x7', 1)], block_1=[('dw_conv_3x3', 0), ('dw_conv_7x7', 1)], block_2=[('dw_conv_3x3', 0), ('dw_conv_3x3', 1), ('dw_conv_11x11', 2), ('dw_conv_3x3', 3), ('dw_conv_3x3', 4), ('dw_conv_5x5', 5)], block_3=[('dw_conv_9x9', 0), ('dw_conv_3x3', 1)])
        
        model = dartsnext_models[args.model_name](dims=args.dims, num_classes=args.num_classes, patch_size=args.patch_size, genotype=test_genotype)

    elif args.model_name == "pcdarts":
        initial_channel = 36
        model_layer = 20
        is_auxiliary = False
        Genotype = namedtuple('Genotype', 'normal normal_concat reduce reduce_concat')
        genotype = Genotype(block_0=[('dw_conv_7x7', 0), ('dw_conv_7x7', 1)], block_1=[('dw_conv_3x3', 0), ('dw_conv_7x7', 1)], block_2=[('dw_conv_3x3', 0), ('dw_conv_3x3', 1), ('dw_conv_11x11', 2), ('dw_conv_3x3', 3), ('dw_conv_3x3', 4), ('dw_conv_5x5', 5)], block_3=[('dw_conv_9x9', 0), ('dw_conv_3x3', 1)])
        model = pcdarts(C=initial_channel, num_classes=args.num_classes, layers=model_layer, auxiliary=is_auxiliary, genotype=genotype)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        betas=(args.adam_beta1, args.adam_beta2),
        weight_decay=args.adam_weight_decay,
        eps=args.adam_epsilon,
    )

    transforms_train = Compose([
        CenterCrop(args.resolution),
        RandAugment(num_ops=2),
        ToTensor()
    ])

    transforms_test = Compose([
        CenterCrop(args.resolution),
        ToTensor()
    ])

    if not args.dataset_name:
        raise ValueError(
            "You must specify a dataset name."
        )

    if args.dataset_name == 'cifar10':
        train_data = dset.CIFAR10(root=args.train_data_dir, train=True, download=True, transform=transforms_train)
        test_data = dset.CIFAR10(root=args.train_data_dir, train=False, download=True, transform=transforms_test)

    elif args.dataset_name == 'cifar100':
        train_data = dset.CIFAR100(root=args.train_data_dir, train=True, download=True, transform=transforms_train)
        test_data = dset.CIFAR100(root=args.train_data_dir, train=False, download=True, transform=transforms_test)

    train_dataloader = torch.utils.data.DataLoader(
        train_data,
        batch_size=args.train_batch_size,
        shuffle=True)
    test_dataloader = torch.utils.data.DataLoader(
        test_data, batch_size=args.eval_batch_size, shuffle=False)

    lr_scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, 
                                             max_lr=args.learning_rate,
                                             steps_per_epoch=len(train_dataloader),
                                             epochs=args.num_epochs)


    loss_fnc = nn.CrossEntropyLoss()

    current_date = datetime.today().strftime('%Y%m%d_%H%M%S')
    logs_path = './training_logs/{}-{}-{}-{}-{}/'.format(args.dataset_name, args.model_name, args.dims, args.patch_size, current_date)

    os.makedirs(logs_path, exist_ok=True)

    logger = CustomLogger("simple-convnext",
                            file_path=f"{logs_path}/training_log.txt")
    model_summary = str(summary(model, (1, 3, args.resolution, args.resolution),  verbose=0))
    logger.log_info(model_summary)

    global_step = 0
    losses = []
    valid_losses = []

    def validate_and_log(epoch, model, test_dataloader, loss_fnc, device, logger,
                        logs_path, train_losses, valid_losses, valid_every, is_final=False):

        model.eval()
        valid_loss = 0
        valid_labels = []
        valid_preds = []

        with torch.no_grad():
            for step, (images, labels) in enumerate(tqdm(test_dataloader, total=len(test_dataloader))):
                images = images.to(device)
                labels = labels.to(device)

                preds = model(images)
                loss = loss_fnc(preds, labels)
                valid_loss += loss.item()

                preds = preds.argmax(dim=-1)
                valid_labels.extend(labels.detach().cpu().tolist())
                valid_preds.extend(preds.detach().cpu().tolist())

        avg_valid_loss = valid_loss / len(test_dataloader)
        valid_losses.append(avg_valid_loss)

        # logging
        print(f"Valid loss: {avg_valid_loss}")
        print(classification_report(valid_labels, valid_preds))
        
        logger.log_info(f"Epoch {epoch}")
        logger.log_info(f"Valid loss: {avg_valid_loss}")
        logger.log_info(classification_report(valid_labels, valid_preds))

        # save checkpoint
        if not is_final:
            torch.save({
                'model_state': model.state_dict(),
                'optimizer_state': optimizer.state_dict(),
            }, os.path.join(logs_path, 'checkpoint.pt'))

            epoch_path = f"{logs_path}/{epoch}"
            os.makedirs(epoch_path, exist_ok=True)
            plot_losses(train_losses=train_losses,
                        valid_losses=valid_losses,
                        path=epoch_path,
                        valid_every=valid_every)
        else:
            torch.save({
                'model_state': model.state_dict(),
                'optimizer_state': optimizer.state_dict(),
            }, os.path.join(logs_path, "final_model.pt"))

            epoch_path =f"{logs_path}/final"
            os.makedirs(epoch_path, exist_ok=True)
            plot_losses(train_losses=losses,
                        valid_losses=valid_losses,
                        path=epoch_path,
                        valid_every=args.save_model_epochs)

    for epoch in range(args.num_epochs):
        model.train()
        if args.model_name == 'pcdarts':
            model.drop_path_prob = 0.3 * epoch / args.num_epochs
        progress_bar = tqdm(total=len(train_dataloader))
        progress_bar.set_description(f"Epoch {epoch}")
        losses_log = 0
        for step, (images, labels) in enumerate(train_dataloader):
            images = images.to(device)
            labels = labels.to(device)

            images, labels = mixup(images, labels)
            images = rand_erasing(images)

            preds = model(images)

            loss = loss_fnc(preds, labels)
            loss.backward()

            if args.use_clip_grad:
                clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()

            progress_bar.update(1)
            losses_log += loss.detach().item()
            logs = {
                "loss_avg": losses_log / (step + 1),
                "loss": loss.detach().item(),
                "lr": lr_scheduler.get_last_lr()[0],
                "step": global_step
            }

            progress_bar.set_postfix(**logs)
            global_step += 1
        progress_bar.close()

        losses.append(losses_log / (step + 1))

        if epoch % args.save_model_epochs == 0:
            validate_and_log(epoch, model, test_dataloader, loss_fnc, device,
                            logger, logs_path, losses,
                            valid_losses, args.save_model_epochs)

    if (args.num_epochs  - 1) % args.save_model_epochs != 0:
        validate_and_log(args.num_epochs - 1, model, test_dataloader, loss_fnc, device,
                        logger, logs_path, losses,
                        valid_losses, args.save_model_epochs, is_final=True)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Simple example of a training script.")
    ###
    parser.add_argument('--model_name', type=str, default='dartsnext_atto', help='which model to use')
    parser.add_argument('--dims', type=tuple, default=[64, 128, 256, 512], help='dims')
    parser.add_argument('--patch_size', type=int, default=1, help='patch size')
    parser.add_argument("--dataset_name", type=str, default='cifar10')
    parser.add_argument("--num_classes", type=int, default=10)
    
    ###
    parser.add_argument("--dataset_config_name", type=str, default=None)
    parser.add_argument("--train_data_dir",
                        type=str,
                        default='./dataset',
                        help="A folder containing the training data.")
    parser.add_argument("--resolution", type=int, default=32)
    parser.add_argument("--train_batch_size", type=int, default=256)
    parser.add_argument("--eval_batch_size", type=int, default=256)
    parser.add_argument("--num_epochs", type=int, default=100)
    parser.add_argument("--save_model_epochs", type=int, default=10)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--learning_rate", type=float, default=1e-3)
    parser.add_argument("--lr_warmup_steps", type=int, default=500)
    parser.add_argument("--adam_beta1", type=float, default=0.9)
    parser.add_argument("--adam_beta2", type=float, default=0.999)
    parser.add_argument("--adam_weight_decay", type=float, default=1e-1) #1e-1
    parser.add_argument("--adam_epsilon", type=float, default=1e-08)
    parser.add_argument("--use_clip_grad", type=bool, default=False)
    parser.add_argument("--logging_dir", type=str, default="logs")

    args = parser.parse_args()

    if args.dataset_name is None and args.train_data_dir is None:
        raise ValueError(
            "You must specify either a dataset name from the hub or a train data directory."
        )
    main(args)