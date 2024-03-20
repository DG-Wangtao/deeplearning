# %% [code]
# %% [code]
# %% [code]
# %% [code]
# %% [code]
# %% [code]
# %% [code]
# %% [code]
# %% [code]
# %% [code]
# !pip install scipy scikit-image torch torchvision pathlib wandb segmentation-models-pytorch
# !pip install wandb
# !pip install wandb --upgrade

import os
import random
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
plt.style.use("ggplot")


import torch
import torch.nn as nn
from torchvision import models, transforms
from torchvision.transforms import v2
from torch.nn.functional import relu, pad
from torch.utils.data import Dataset, DataLoader, random_split

from PIL import Image
from typing import Tuple
from pathlib import Path

import torch
from torch import nn, Tensor
import torch.nn.functional as F

# 添加训练和测试函数
from tqdm import tqdm
import wandb
import logging
import time
import torch.optim as optim
import segmentation_models_pytorch as smp

from datetime import datetime

import gc 


# TODO: image和mask名称不一样时跳过
class APODataSet(Dataset):
    # 格式不对的异常数据
    def __init__(self, img_dir, mask_dir: str, size) -> None:
        # 获取所有图片路径
        img_paths = list(Path(img_dir).glob("*"))
        mask_paths = list(Path(mask_dir).glob("*"))
        self.images = []
        self.masks = []
        for img_idx in range(len(img_paths)):
            img_path = img_paths[img_idx]
            img = self.load_image(img_path)
            num_channels = len(img.getbands())
            if num_channels != 3:
                continue
            
            mask_path = mask_paths[img_idx]
            self.images.append(img_path)
            self.masks.append(mask_path)
            
        self.transform = transforms.Compose([ transforms.Resize(size), transforms.ToTensor()])
        self.images = np.array(self.images)
        self.masks = np.array(self.masks)

    def load_image(self, path) -> Image.Image:
        "Opens an image via a path and returns it."
        return Image.open(path)
    
    #  重写 __len__() 方法 (optional but recommended for subclasses of torch.utils.data.Dataset)
    def __len__(self) -> int:
        "Returns the total number of samples."
        return len(self.images)

    # 重写 __getitem__() 方法 (required for subclasses of torch.utils.data.Dataset)
    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.images[index], self.masks[index]

class EnhanceDataSet(Dataset):
    def __init__(self, dataset, size, transform, cutmix):
        self.size = size
        self.dataset = dataset
        self.transform = transform
        self.cutmix = cutmix
#         transforms.Compose([ 
#                 transforms.Resize(size),
#                 transforms.RandomHorizontalFlip(),  # 随机水平翻转
#                 transforms.RandomVerticalFlip(),    # 随机垂直旋转
#                 transforms.RandomRotation(10) ,     # 随机旋转 （-10,10）度
#                 transforms.ToTensor()
#         ])

    def load_image(self, path) -> Image.Image:
        "Opens an image via a path and returns it."
        return Image.open(path)
    
    
    def __getitem__(self, index):
        imag_path, mask_path = self.dataset[index]
        image = self.load_image(imag_path)
        mask = self.load_image(mask_path)

        seed = np.random.randint(2147483647)
        torch.manual_seed(seed)
        image = self.transform(image)
        torch.manual_seed(seed)
        mask = self.transform(mask)
        
        # cut mix
        if self.cutmix:
            rand_index = random.randint(0, len(self.dataset)-1)
            rand_imag_path, rand_mask_path = self.dataset[rand_index]
            rand_img = self.load_image(rand_imag_path)
            rand_mask = self.load_image(rand_mask_path)

#             seed = np.random.randint(2147483647)
            torch.manual_seed(seed)
            rand_img = self.transform(rand_img)
            torch.manual_seed(seed)
            rand_mask = self.transform(rand_mask)

            lam = np.random.beta(1., 1.)
            bbx1, bby1, bbx2, bby2 = self.rand_bbox(rand_mask.size(), lam)

            image[:, bbx1:bbx2, bby1:bby2] = rand_img[:, bbx1:bbx2, bby1:bby2]
            mask[:, bbx1:bbx2, bby1:bby2] = rand_mask[:, bbx1:bbx2, bby1:bby2]
    
        mask = torch.where(mask>0.5,torch.ones_like(mask),torch.zeros_like(mask))
        
        return image.numpy(), mask.numpy()
        
    def __len__(self):
        return len(self.dataset)
    
    #  CutMix 的切块功能
    def rand_bbox(self, size, lam):
        W = size[1]
        H = size[2]
        cut_rat = np.sqrt(1. - lam)
        cut_w = int(W * cut_rat)
        cut_h = int(H * cut_rat)

        # uniform
        cx = np.random.randint(W)
        cy = np.random.randint(H)

        bbx1 = np.clip(cx - cut_w // 2, 0, W)
        bby1 = np.clip(cy - cut_h // 2, 0, H)
        bbx2 = np.clip(cx + cut_w // 2, 0, W)
        bby2 = np.clip(cy + cut_h // 2, 0, H)

        return bbx1, bby1, bbx2, bby2


# def train_collate_fn(batch):
#     size = [512, 512]
#     trans = transforms.Compose([ 
#                 transforms.RandomHorizontalFlip(),  # 随机水平翻转
#                 transforms.RandomVerticalFlip(),    # 随机垂直旋转
#                 transforms.RandomRotation(10) ,     # 随机旋转 （-10,10）度
#     ])
    
#     images = torch.empty(len(batch), 3, size[0], size[1])
#     masks = torch.empty(len(batch),1, size[0], size[1])
    
    
#     for i in range(len(batch)):
#         image, mask = batch[i]
#         seed = np.random.randint(2147483647)
#         torch.manual_seed(seed)
#         image = trans(image)
#         torch.manual_seed(seed)
#         mask = trans(mask)

        
#         # cut mix
#         rand_index = random.randint(0, len(batch)-1)
#         rand_img, rand_mask = batch[rand_index]

#         lam = np.random.beta(1., 1.)
#         bbx1, bby1, bbx2, bby2 = rand_bbox(rand_mask.size(), lam)

#         image[:, bbx1:bbx2, bby1:bby2] = rand_img[:, bbx1:bbx2, bby1:bby2]
#         mask[:, bbx1:bbx2, bby1:bby2] = rand_mask[:, bbx1:bbx2, bby1:bby2]
        
#         images[i] = image
#         masks[i] = mask
    
#     return images, masks

def initDataLoader(batch_size, size= [512, 512]):
    dataset =  APODataSet(img_dir = "/kaggle/input/dltrack/apo_images",
                          mask_dir = "/kaggle/input/dltrack/apo_masks",
                         size = size)

    total = len(dataset)
    train_size = int(0.8*total)
    validate_size = total - train_size

    train_data, validate_data = random_split(dataset, [train_size, validate_size])
    train_data = EnhanceDataSet(dataset = train_data, size = size, transform = 
                                        transforms.Compose([ 
                                            transforms.Resize(size),
                                            transforms.RandomHorizontalFlip(),  # 随机水平翻转
                                            transforms.RandomVerticalFlip(),    # 随机垂直旋转
                                            transforms.RandomRotation(10) ,     # 随机旋转 （-10,10）度
                                            transforms.ToTensor()
                                    ]), cutmix = True
                               )
    
    validate_data = EnhanceDataSet(dataset = validate_data, size = size, transform = 
                                        transforms.Compose([ 
                                            transforms.Resize(size),
                                            transforms.ToTensor()
                                    ]), cutmix = False
                               )
    print("dataset info\ntotal: {}, train_size: {}, validate_size: {}".format(total, len(train_data), len(validate_data)))

    trainloader = DataLoader(dataset=train_data,
                                         batch_size=batch_size,
                                         num_workers=0, 
                                         shuffle=True)
    valloader = DataLoader(dataset=validate_data,
                                        batch_size=1, 
                                        num_workers=0,
                                        shuffle=False)
    return trainloader, valloader


def showImage(loader):
    for i, data in enumerate(loader):
        images, masks = data
        orig_img = images[0]
        mask_img = masks[0]
        break

    # idx = random.randint(0, len(dataset))
    # orig_img, mask_img = dataset[idx]

    print(orig_img.shape)
    print(mask_img.shape)


    orig_img = orig_img.cpu().numpy().transpose(1, 2, 0)
    mask_img = mask_img.cpu().numpy().transpose(1, 2, 0)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize = (15, 12))

    ax1.imshow(orig_img)
    ax1.grid(False)
    ax1.axis('off')
    ax1.set_title("origin_img")

    ax2.imshow(mask_img, cmap="gray")
    ax2.grid(False)
    ax2.axis('off')
    ax2.set_title("mask_img")

    plt.show()




@torch.inference_mode()
def evaluate(model, dataloader, device, amp, experiment, epoch, logging = False):
    class_labels= { 1: "target" }
    model.eval()
    
    if logging:
        if epoch % 20 == 0:
            columns = ["epoch", "image_id", "image", "bceLoss", "diceLoss", "f1_score", "iouScore", "accuracy", "precision",]
            test_table = wandb.Table(columns=columns)
        
            artifact = wandb.Artifact("test_preds", type="raw_data")
    
    num_val_batches = len(dataloader)
    bce_loss = 0
    dice_loss = 0
    iou_score = 0

    if isinstance(model, nn.DataParallel):
        num_classes = model.module.num_classes
    else:
        num_classes = model.num_classes
    
    # 因为在非训练过程（推理过程中），已经在网络最后一层加了log和过滤
    # 因此这里的损失函数都要使用不带log的
    criterion = nn.BCELoss().cuda()
    diceloss = smp.losses.DiceLoss(mode='binary', from_logits=False).cuda()
    
    g_bce_loss = 0
    g_dice_loss = 0
    g_iou_score = 0
            
    g_accuracy = 0
    g_precision= 0
    g_f1_score = 0
    g_f2_score= 0
    
    idx = -1
    # iterate over the validation set
    with tqdm(total=num_val_batches, desc='Validation round', unit='batch', position=0 ,leave=True) as pbar:
        for batch in dataloader:
            idx += 1
            
            images, mask_true = batch

            # move images and labels to correct device and type
            images = images.to(device=device, dtype=torch.float32, memory_format=torch.channels_last).clone()
            mask_true = mask_true.to(device=device, dtype=torch.float32).clone()

            # predict the mask
            mask_pred = model(images)
            bce_loss = criterion(mask_pred, mask_true.float()).item()
            dice_loss = diceloss(mask_pred, mask_true).item()

            tp, fp, fn, tn = smp.metrics.get_stats(mask_pred, mask_true.long(), mode='binary', threshold=0.5)

            iou_score = smp.metrics.iou_score(tp, fp, fn, tn, reduction="micro").item()
            
            f1_score = smp.metrics.f1_score(tp, fp, fn, tn, reduction="micro").item()
            f2_score = smp.metrics.fbeta_score(tp, fp, fn, tn, beta=2, reduction="micro").item()
        
            accuracy = smp.metrics.accuracy(tp, fp, fn, tn, reduction="macro").item()
            precision = smp.metrics.precision(tp, fp, fn, tn, reduction="macro").item()
    

        
            g_bce_loss += bce_loss
            g_dice_loss += dice_loss
        
            g_iou_score += iou_score
        
            g_f1_score += f1_score
            g_f2_score += f2_score
            
            g_accuracy += accuracy
            g_precision += precision
            
            pbar.update(images.shape[0])
            
            if logging:
                if test_table != None:
                    test_table.add_data(epoch, idx, 
                                    wandb.Image(images[0].float().cpu(),
                                                masks = { 
                                                    "predictions": {
                                                        "mask_data": mask_pred[0][0].cpu().numpy(), "class_labels": class_labels
                                                    },
                                                    "ground_truth": {
                                                        "mask_data": mask_true[0][0].cpu().numpy(), "class_labels": class_labels
                                                    },
                                    }),
                                    bce_loss, dice_loss, f1_score,
                                    iou_score, accuracy, precision)

        g_bce_loss = (g_bce_loss / max(num_val_batches, 1))
        g_dice_loss = (g_dice_loss / max(num_val_batches, 1))
        g_iou_score = (g_iou_score / max(num_val_batches, 1))
        g_accuracy = (g_accuracy / max(num_val_batches, 1))
        g_precision= (g_precision / max(num_val_batches, 1))
        g_f1_score = (g_f1_score / max(num_val_batches, 1))
        g_f2_score= (g_f2_score / max(num_val_batches, 1))

        pbar.set_postfix(**{"Validation bce loss": bce_loss, "dice loss": dice_loss, "IoU Score": iou_score})
    
    if logging:
        try:
            if test_table != None and artifact != None:
                artifact.add(test_table, "test_predictions")
                experiment.log_artifact(artifact)
                del test_table
                del artifact
            
            experiment.log({
                'ave_validation Loss': g_bce_loss + g_dice_loss,
                'ave_accuracy': g_accuracy,
                'ave_precision':g_precision,
                'ave_f1_score':g_f1_score,
                'ave_f2_score':g_f2_score,
                'average validation IoU Score': g_iou_score,
            })
        except Exception as e:
            print(e)
            pass
        

    return (dice_loss, iou_score)    


def train(model, device, project,
          epochs: int = 60,
          learning_rate: float = 1e-5, 
          weight_decay: float = 1e-8,
          momentum: float = 0.999,
          batch_size: int = 6,
          amp: bool = False,
          gradient_clipping: float = 1.0):
    
    trainloader, valloader = initDataLoader(batch_size)
    n_train = len(trainloader.dataset)
    n_val = len(valloader.dataset)
    showImage(trainloader)


    if isinstance(model, nn.DataParallel):
        num_classes = model.module.num_classes
        input_channels = model.module.input_channels
    else:
        num_classes = model.num_classes
        input_channels = model.input_channels
        

    # (Initialize logging)
    experiment = wandb.init(project=project, job_type="upload", resume='allow', anonymous='must', notes='水平和垂直翻转，旋转(-10,10)度，mixcut')
    experiment.config.update(
        dict(epochs=epochs, batch_size=batch_size, amp=True)
    )

    
    logging.info(f'''Starting training:
        Epochs:          {epochs}
        Batch size:      {batch_size}
        Learning rate:   {learning_rate}
        Training size:   {n_train}
        Validation size: {n_val}
        Device:          {device.type}
        Mixed Precision: {amp}
    ''')
    

    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=1, T_mult=2, eta_min=5e-5)  # goal: maximize Dice scor
    grad_scaler = torch.cuda.amp.GradScaler(enabled=amp)
    
    # 训练过程中，网络最后一层没有添加log，所以要使用带log的损失函数
    criterion = nn.BCEWithLogitsLoss().cuda()
    dice_loss = smp.losses.DiceLoss(mode='binary', from_logits=True).cuda()

    global_step = 0

    # 5. Begin training
    for epoch in range(1, epochs + 1):
        model.train()
        epoch_loss = 0
        with tqdm(total=n_train, desc=f'Epoch {epoch}/{epochs}', unit='batch') as pbar:
            for batch in trainloader:
                images, true_masks = batch

                assert images.shape[1] == input_channels, \
                    f'Network has been defined with {input_channels} input channels, ' \
                    f'but loaded images have {images.shape[1]} channels. Please check that ' \
                    'the images are loaded correctly.'

                images = images.to(device=device, dtype=torch.float32, memory_format=torch.channels_last)
                
                true_masks = true_masks.to(device=device, dtype=torch.long)

                with torch.autocast(device.type if device.type != 'mps' else 'cpu', enabled=amp):
                    masks_pred = model(images)
                    loss = criterion(masks_pred, true_masks.float())
                    loss += dice_loss(masks_pred, true_masks)
                    tp, fp, fn, tn = smp.metrics.get_stats(masks_pred, true_masks.long(), mode='binary', threshold=0.5)
                    iou_score = smp.metrics.iou_score(tp, fp, fn, tn, reduction="micro").item()
    
                optimizer.zero_grad(set_to_none=True)
                grad_scaler.scale(loss).backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clipping)
                grad_scaler.step(optimizer)
                grad_scaler.update()

                pbar.update(images.shape[0])
                global_step += 1
                epoch_loss += loss.item()
                pbar.set_postfix(**{'loss (batch)': epoch_loss/n_train})
                
                if global_step % 10 == 0:
                    experiment.log({
                        'learning rate': optimizer.param_groups[0]['lr'],
                        'train iou': iou_score,
                        'train loss': loss.item(),
                        'step': global_step,
                        'epoch': epoch
                    })

           # Evaluation round
#                 division_step = (n_train // batch_size)
#                 if division_step > 0:
#                     if global_step % division_step == 0:
#                         with torch.no_grad():
#                             val_score, iou_score = evaluate(model, valloader, device, amp, experiment, epoch, test_table, logging = False)
#                         torch.set_grad_enabled(True)
#                         model.train()
#                         scheduler.step(val_score)
        
        # 每10个 epoch 更新一遍 wandb
        with torch.no_grad():
            val_score, iou_score = evaluate(model, valloader, device, amp, experiment, epoch, logging = True)
        torch.set_grad_enabled(True)
        model.train()
        scheduler.step(val_score)
        
        gc.collect()
#         torch.cuda.empty_cache()

    experiment.finish()

def LoopDataLoader(epochs, batch_size):
    trainloader, valloader = initDataLoader(batch_size)
    n_train = len(trainloader.dataset)
    n_val = len(valloader.dataset)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    for epoch in range(1, epochs + 1):
        print("{}/{}".format(epoch, epochs))
        for batch in trainloader:
            images, true_masks = batch
            images = images.to(device=device, dtype=torch.float32, memory_format=torch.channels_last)
                
            true_masks = true_masks.to(device=device, dtype=torch.long)
        for batch in valloader:
            images = images.to(device=device, dtype=torch.float32, memory_format=torch.channels_last)
                
            true_masks = true_masks.to(device=device, dtype=torch.long)
            images, true_masks = batch
        

def StarTrain(project, model, epochs, batch_size):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if torch.cuda.device_count() > 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        model = nn.DataParallel(model)

    model = model.to(memory_format=torch.channels_last)
    model.to(device)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"模型参数量为：{total_params}")
    print("其详情为：")
    for name,parameters in model.named_parameters():
        print(name,':',parameters.size())
    train(model, device, project=project, epochs=epochs, batch_size=batch_size)

