
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))
import os

from dataloader import MyData, MyITSData,patched_data_Train_Aug
from dataloader import   Dense_haze_Train

import torch
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
import torch.nn as nn
from torchvision import models

# torch.cuda.set_device(1)  # 设置默认的CUDA设备
from tqdm import tqdm
import piq

import matplotlib
matplotlib.use('Agg')  # 关键：必须在 import pyplot 前执行
import matplotlib.pyplot as plt
import torchvision
from torch.cuda.amp import GradScaler, autocast
import time
from torch.fft import fft2, ifft2
import torch.nn.functional as F

import DWT_Block
import module
import WAFF
from torchvision import transforms
import loss
import argparse


parser = argparse.ArgumentParser()
# 1. 数据/类别
parser.add_argument("--category",   type=str,  default="NH-haze", choices=["NH-haze","O-haze","Dense-haze","ITS"])
parser.add_argument("--train_path", type=str, default="")  # 留空则走默认
parser.add_argument("--val_patch_path", type=str, default="")
parser.add_argument("--val_orig_path", type=str, default="")
# 2. 训练超参
parser.add_argument("--epochs",     type=int,   default=150)
parser.add_argument("--batch_size", type=int,   default=8)
parser.add_argument("--lr",         type=float, default=5e-4)#0.0005
parser.add_argument("--beta1",      type=float, default=0.5)
parser.add_argument("--beta2",      type=float, default=0.999)
parser.add_argument("--min_lr",     type=float, default=5e-5)#0.00005
parser.add_argument("--T_max",      type=int,   default=10, help="CosineAnnealingLR")
# 3. 系统
parser.add_argument("--gpu",        type=int,   default=0, help="GPU id, -1 for cpu")
parser.add_argument("--num_workers",type=int,   default=0)
parser.add_argument("--save_every", type=int,   default=10, help="save ckpt every N epochs")
# 4. 路径与开关
parser.add_argument("--resume",     type=int,   default=0, help="epoch to resume, 0 means train from scratch")
parser.add_argument("--output_root",type=str,   default="./lightweight/Ablation/woWA")

args = parser.parse_args()
if not args.train_path:
    args.train_path = {
        "NH-haze": "",
        "O-haze": "",
        "Dense-haze": "",

        "ITS": "",
        "reside6k": ""
    }[args.category]

if not args.val_patch_path:
    args.val_patch_path = {
        "NH-haze":"",
        "O-haze": "",
        "Dense-haze": "",

        "ITS": "",
        "reside6k": ""
    }[args.category]

if not args.val_orig_path:
    args.val_orig_path = args.val_patch_path  # 你原来大部分情况二者相同



BATCH_SIZE_TRAIN = args.batch_size
LEARNING_RATE    = args.lr
NUM_EPOCHS       = args.epochs
beta1, beta2     = args.beta1, args.beta2
category         = args.category
T_max, min_lr    = args.T_max, args.min_lr

# device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")
# print(f"GPU name: {torch.cuda.get_device_name(device)}")
# Get the directory where the script is located
script_dir = Path(__file__).resolve().parent




# 3. 路径
training_data_path         = os.path.normpath(args.train_path)
validation_patch_data_path = os.path.normpath(args.val_patch_path)
validation_original_data_path = os.path.normpath(args.val_orig_path)
print(f"train_data is: {category}")
print(f"train_data_path is: {training_data_path}")

# 4. 输出目录
base_dir = os.path.join(args.output_root, category)
figures_dir = os.path.join(base_dir, "figures")
validation_generation_dir = os.path.join(figures_dir, "Validation Generation")
models_dir  = os.path.join(base_dir, "models")
save_path   = models_dir
file_name_epoch_best_ssim = os.path.join(save_path, "dehazing_model_iter_ssim.pth")
file_name_epoch_best_psnr = os.path.join(save_path, "dehazing_model_iter_psnr.pth")

for d in [base_dir, figures_dir, validation_generation_dir, models_dir, save_path]:
    os.makedirs(d, exist_ok=True)




#NH_haze
training_data = patched_data_Train_Aug(training_data_path,aug_multiplier=2)
#ITS
# training_data = MyITSData(training_data_path,image_size=(256, 256),change_size = True)
validation_patch_data = MyData(validation_patch_data_path, image_size=(1024,1024))
validation_original_data = MyData(validation_original_data_path, image_size=(1024,1024))

# Create a DataLoader instance
training_data_loader = torch.utils.data.DataLoader(training_data, batch_size=BATCH_SIZE_TRAIN, shuffle=True,num_workers=args.num_workers,drop_last = True)
validation_patch_data_loader = torch.utils.data.DataLoader(validation_patch_data, batch_size=1, shuffle=False)
validation_original_data_loader = torch.utils.data.DataLoader(validation_original_data, batch_size=1, shuffle=False)

# Initialize the model
print(device)  # 应该输出 cuda:1
model = WAFF.Dehazing_Model().to(device)
# print(model)
# model = model.to(device)

# Initialize the optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, betas=(beta1, beta2))
scaler = GradScaler()
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=T_max, eta_min=min_lr)



# 创建文件夹
folders = [
    base_dir,
    figures_dir,
    validation_generation_dir,
    models_dir,
    save_path
]

for folder in folders:
    if not os.path.exists(folder):
        os.makedirs(folder)

criterion = nn.L1Loss()
# contrastive = loss.FCR()
def loss_DWT(input, target):
    # input = input.to(dtype= torch.float32)
    # target = target.to(dtype=torch.float32)
    input_dwt_low, input_dwt_high = DWT_Block.dwt_init(input)
    target_dwt_low, target_dwt_high = DWT_Block.dwt_init(target)

    input_dwt_low_abs = torch.abs(input_dwt_low)
    target_dwt_low_abs = torch.abs(target_dwt_low)
    input_dwt_high_abs = torch.abs(input_dwt_high)
    target_dwt_high_abs = torch.abs(target_dwt_high)

    loss_low = F.mse_loss(input_dwt_low_abs, target_dwt_low_abs)
    loss_high = F.mse_loss(input_dwt_high_abs, target_dwt_high_abs)
    loss = loss_low + loss_high
    return loss

def loss_FFT(input, target):
    # Compute the 2D Fourier Transform of the input and target images
    # input = input.to(dtype=torch.float32)
    # target = target.to(dtype=torch.float32)
    input_fft = fft2(input)
    target_fft = fft2(target)

    # Compute the magnitude of the Fourier Transforms
    input_mag = torch.abs(input_fft)
    target_mag = torch.abs(target_fft)

    # Compute and return the L1 loss
    return F.l1_loss(input_mag, target_mag)


loss = Loss.Loss(loss_weight=1.0, reduction='mean')
# contrast_loss = ContrastLoss(ablation=False)   # 需要负样本，保持默认 False
l1_loss = nn.L1Loss()
def loss_fn_ori(outputs, targets):


    DWT_loss = loss_DWT(outputs, targets)
    FFT_loss = loss_FFT(outputs, targets)
    loss = (7 * DWT_loss) + FFT_loss
    return loss
def loss_fn(outputs, targets,inputs):
    # input = input.to(dtype=torch.float32)
    # target = target.to(dtype=torch.float32)
    # Compute the L1 loss between the input and target images
    # ==========version1============
    DWT_loss = loss_DWT(outputs, targets)
    FFT_loss = loss_FFT(outputs, targets)
    # loss = (7 * DWT_loss) + FFT_loss
    # ==========version2============
    SASW_loss = Loss(outputs, targets)
    # ==========version3============
    # l1loss = criterion(outputs,targets)
    # con_loss = contrastive(outputs, targets, inputs)
    # loss = l1loss + (0.1* con_loss)
    loss = FFT_loss + (DWT_loss*7) + SASW_loss

    # return SASW_loss +  DWT_loss + FFT_loss +con_loss
    return loss

# List to store Losses and IQA scores
best_psnr = 0.0
best_ssim = 0.0
training_loss = []
validation_patch_loss = []
validation_whole_loss = []
validation_patch_psnr = []
validation_patch_ssim = []
validation_patch_mse = []
validation_whole_psnr = []
validation_whole_ssim = []
validation_whole_mse = []
learning_rate_list = []


start_epoch = 0
if args.resume > 0:
    ckpt = os.path.join(save_path, f"dehazing_model_iter{args.resume}.pth")
    if os.path.isfile(ckpt):
        model.load_state_dict(torch.load(ckpt, map_location=device))
        start_epoch = args.resume
        print(f"resume from epoch {start_epoch}")
    else:
        print(f"checkpoint {ckpt} not found, train from scratch")

for epoch in range(NUM_EPOCHS):
    print(f"Epoch: {epoch + 1}/{NUM_EPOCHS}")

    # Training
    model.train()
    batch_loss = 0.0

    start_time = time.time()

    progress_bar = tqdm(enumerate(training_data_loader), total=len(training_data_loader), desc='Training')
    # print(len(training_data_loader))
    for i, batch in progress_bar:
        inputs, targets = batch
        inputs = inputs.to(device)
        targets = targets.to(device)
        # inputs_raw = inputs.clone()
        inputs = module.normalize(inputs)
        targets = module.normalize(targets)
        # Forward Pass
        outputs = model(inputs)
        loss = loss_fn(outputs, targets, inputs)

        # Backward Pass
        optimizer.zero_grad()

        # with autocast():
        # outputs = model(inputs)
        # loss = loss_fn(outputs, targets)

        batch_loss += loss.item()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        # loss.backward()
        # optimizer.step()

        # batch_loss += loss.item()

        # Update progress bar
        progress_bar.set_description(f"Training Epoch {epoch + 1} - Batch {i + 1} - Loss: {loss.item():.4f}")
    scheduler.step()
    current_lr = optimizer.param_groups[0]['lr']
    learning_rate_list.append(current_lr)

    avg_train_loss = batch_loss / len(training_data_loader)
    training_loss.append(avg_train_loss)

    # Validation
    model.eval()

    validation_batch_patch_loss = 0.0
    validation_batch_whole_loss = 0.0
    validation_batch_patch_psnr = 0.0
    validation_batch_patch_ssim = 0.0
    validation_batch_patch_mse = 0.0
    validation_batch_whole_psnr = 0.0
    validation_batch_whole_ssim = 0.0
    validation_batch_whole_mse = 0.0
    validation_patch_gen_images = []
    validation_patch_gt_images = []
    validation_patch_haze_images = []
    with torch.no_grad():


        # Validation Patch Calculation
        for i, batch in enumerate(validation_patch_data_loader):  # validation_patch_data_loader
            inputs, targets = batch
            inputs = inputs.to(device)
            targets = targets.to(device)
            # inputs_raw = inputs.clone()
            inputs = module.normalize(inputs)
            targets = module.normalize(targets)

            outputs = model(inputs)
            # loss = loss_fn_ori(outputs, targets)
            loss = loss_fn(outputs, targets, inputs)
            validation_batch_patch_loss += loss.item()

            inputs = module.denormalize(inputs)
            outputs = module.denormalize(outputs)
            targets = module.denormalize(targets)

            validation_patch_gen_images.append(outputs.squeeze(0))
            validation_patch_gt_images.append(targets.squeeze(0))
            validation_patch_haze_images.append(inputs.squeeze(0))

            # Calculate PSNR, SSIM and MSE
            validation_batch_patch_psnr += piq.psnr(outputs, targets).item()
            validation_batch_patch_ssim += piq.ssim(outputs, targets, data_range=1., reduction='mean').item()
            validation_batch_patch_mse += torch.nn.functional.mse_loss(outputs, targets).item()

        validation_patch_loss.append(validation_batch_patch_loss / len(validation_patch_data_loader))
        validation_patch_psnr.append(validation_batch_patch_psnr / len(validation_patch_data_loader))
        validation_patch_ssim.append(validation_batch_patch_ssim / len(validation_patch_data_loader))
        validation_patch_mse.append(validation_batch_patch_mse / len(validation_patch_data_loader))
        average_ssim = validation_batch_patch_ssim / len(validation_patch_data_loader)
        average_psnr = validation_batch_patch_psnr / len(validation_patch_data_loader)
        print(f"this epoch model with SSIM: {average_ssim:.4f}")
        print(f"this epoch model with PSNR: {average_psnr:.4f}")
        if average_ssim > best_ssim:
            best_ssim = average_ssim
            torch.save(model.state_dict(), file_name_epoch_best_ssim)
            print(f"Saved best model with SSIM: {best_ssim:.4f}")
        if average_psnr > best_psnr:
            best_psnr = average_psnr
            torch.save(model.state_dict(), file_name_epoch_best_psnr)
            print(f"Saved best model with PSNR: {best_psnr:.4f}")

    # Saving the Validation Images Generated
    images = torch.cat([
        torch.stack(validation_patch_haze_images),
        torch.stack(validation_patch_gen_images),
        torch.stack(validation_patch_gt_images)
    ], dim=0)

    grid = torchvision.utils.make_grid(images, nrow=len(validation_patch_gen_images))

    # Convert the grid to a numpy array and transpose the dimensions for displaying
    grid = grid.cpu().numpy().transpose((1, 2, 0))

    # Display the grid
    plt.figure(figsize=(20, 10))
    plt.imshow(grid)
    plt.axis('off')

    # Save the grid to a file
    plt.savefig(os.path.join(validation_generation_dir, f'image_grid_{epoch + 1}.jpg'), dpi=300)

    # Close the figure to free up memory
    plt.close()

    # Plotting the Losses
    epochs = range(1, len(training_loss) + 1)

    plt.figure(figsize=(12, 6))

    plt.plot(epochs, training_loss, label='Training Loss')
    plt.plot(epochs, validation_patch_loss, label='Validation Patch Loss/test')
    # plt.plot(epochs, validation_whole_loss, label='Validation Whole Loss/val')

    plt.title('Training and Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    # Save the figure to a file
    plt.savefig(os.path.join(figures_dir, f'loss_plot_epoch.jpg'))  # 使用变量定义路径


    # Close the figure to free up memory
    plt.close()

    # Plotting the Learning Rate
    epochs = range(1, len(learning_rate_list) + 1)

    plt.figure(figsize=(12, 6))

    plt.plot(epochs, learning_rate_list, label='Learning Rate')

    plt.title('Learning Rate Each epoch')
    plt.xlabel('Epochs')
    plt.ylabel('Learning Rate')
    plt.legend()

    # Save the figure to a file
    # plt.savefig(f'./mytrain_EPA/figures/learning_rate_epoch.jpg')
    plt.savefig(os.path.join(figures_dir, f'learning_rate_epoch.jpg'))  # 使用变量定义路径
    # Close the figure to free up memory
    plt.close()

    # Plotting the PSNR
    plt.figure(figsize=(12, 6))
    # plt.plot(epochs, validation_whole_psnr, label='Validation Whole PSNR/val')
    plt.plot(epochs, validation_patch_psnr, label='Validation Patch PSNR/test')
    plt.title('Validation PSNR')
    plt.xlabel('Epochs')
    plt.ylabel('PSNR')
    plt.legend()
    # Save the figure to a file

    plt.savefig(os.path.join(figures_dir, f'psnr_plot_epoch.jpg'))
    # Close the figure to free up memory
    plt.close()

    # Plotting the SSIM
    plt.figure(figsize=(12, 6))
    # plt.plot(epochs, validation_whole_ssim, label='Validation Whole SSIM/val')
    plt.plot(epochs, validation_patch_ssim, label='Validation Patch SSIM/test')
    plt.title('Validation SSIM')
    plt.xlabel('Epochs')
    plt.ylabel('SSIM')
    plt.legend()
    # Save the figure to a file

    # Close the figure to free up memory
    plt.savefig(os.path.join(figures_dir, f'ssim_plot_epoch.jpg'))
    plt.close()

    # Plotting the MSE
    plt.figure(figsize=(12, 6))
    # plt.plot(epochs, validation_whole_mse, label='Validation Whole MSE/val')
    plt.plot(epochs, validation_patch_mse, label='Validation Patch MSE/test')
    plt.title('Validation MSE')
    plt.xlabel('Epochs')
    plt.ylabel('MSE')
    plt.legend()
    # Save the figure to a file

    plt.savefig(os.path.join(figures_dir, f'mse_plot_epoch.jpg'))

    # Close the figure to free up memory
    plt.close()

    end_time = time.time()
    print(
        f'Epoch:{epoch + 1} Training Loss: {avg_train_loss:.4f}, Time Taken for epoch: {end_time - start_time:.2f} seconds')


    if (epoch+1) % args.save_every ==0:
        ckpt_name = os.path.join(save_path, f'dehazing_model_iter_{epoch+1}.pth')
        torch.save(model.state_dict(), ckpt_name)

# Save the trained model
# torch.save(model.state_dict(), file_name_epoch_final)
print(f"Training complete. Best SSIM achieved: {best_ssim:.4f}")
print(f"Training complete. Best PSNR achieved: {best_psnr:.4f}")

