import torch
# import dehazing_model
import My_model

from pathlib import Path
from dataloader import MyData_Test,MyData
import piq
import os
import sys
from pathlib import Path
import Feature_Processing

from matplotlib.pyplot import imsave
# import MSB_Model
sys.path.append(str(Path(__file__).resolve().parent.parent))
# Get the directory where the script is located
script_dir = Path(__file__).resolve().parent

category = 'NH-haze'
#Select size of the input images. Default is 1024x1024
# Image_size=(512,512)
# Image_size=(1024,1024)


base_path = 'F:/Python_For_kitlov/4.desmoke/WLD-Net-main/src/lightweight/Ablation/64/'
picsaze = True
# gen_path = '/home/8T/lwj/dehaze/WLD-Net-main/src'                      MSB_MFM+fcb+saswloss
# Pick the model weights based on the dataset it was trained on. Comment out the other model paths
model_path= base_path +category+ '/models/dehazing_model_iter_psnr.pth'  #For model trained on NH-HAZE dataset

# folder_path='/home/8T/lwj/dehaze/WLD-Net-main/src/MY/L_MBlock_smfa/NH-Haze/outputs/epoch_100/'
folder_path  =  base_path+category+'/outputs/best/'
# folder_path='/home/8T/lwj/dehaze/WLD-Net-main/src/MY/L_MBlock_MFM_HF/NH-Haze/outputs/epoch_ssim/'
txt_name = 'results.txt'
model = My_model.Dehazing_Model()
# model = My_model.Dehazing_Model()
# print(model)
# print(model_path)
#model_path='./models/DH_dehazing_model_final.pth'  #For model trained on D-HAZE dataset
test_IMAGE_SIZE = (512, 512)
if category == 'NH-haze':
    input_path = r"F:/Python_For_kitlov/4.desmoke/WLD-Net-main/test_input/NH-Haze3"
if category == 'O-haze':
    input_path = "F:/Python_For_kitlov/4.desmoke/1_data/RESIDE/HSTS/synthetic"
if category == 'Dense-haze':
    input_path = "F:/Python_For_kitlov/4.desmoke/WLD-Net-main/test_input/Dense_1024"
if category == 'ITS':
    input_path = "F:/Python_For_kitlov/4.desmoke/1_data/RESIDE/SOTS/indoor"
if category == 'haze4k':
    input_path = "F:/Python_For_kitlov/4.desmoke/1_data/Haze4K/Haze4K/test"
    test_IMAGE_SIZE = (400, 400)
if category == 'reside6k':
    input_path = "F:/Python_For_kitlov/4.desmoke/1_data/RESIDE/HSTS/synthetic/"

Image_size=test_IMAGE_SIZE
#object path


if not os.path.exists(folder_path):
    os.makedirs(folder_path)


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def process_images(model, test_data, storage_path, Img_size=(1024, 1024)):
    test_data = MyData(test_data, image_size=(256, 256),change_imagesize=False)
    test_data_dataloader = torch.utils.data.DataLoader(test_data, batch_size=1, shuffle=False, num_workers=0)

    ssim_values = []
    psnr_values = []
    image_results = []  # 保存每张图片的结果

    i = 1
    for hazy_img, gt_img in test_data_dataloader:
        with torch.no_grad():
            hazy_img = hazy_img.to('cuda')
            gt_img = gt_img.to('cuda')
            hazy_img = Feature_Processing.normalize(hazy_img)
            dehazed_img = model(hazy_img)

        hazy_img = Feature_Processing.denormalize(hazy_img)
        dehazed_img = Feature_Processing.denormalize(dehazed_img)

        # 计算指标
        ssim_value = piq.ssim(gt_img, dehazed_img).item()
        psnr_value = piq.psnr(gt_img, dehazed_img, data_range=1.).item()

        print(f'Image {i}: PSNR={psnr_value:.4f}, SSIM={ssim_value:.4f}')

        ssim_values.append(ssim_value)
        psnr_values.append(psnr_value)

        # 保存每张图片的结果
        image_results.append(f'Image {i}: PSNR={psnr_value:.4f}, SSIM={ssim_value:.4f}')

        # 保存图片
        dehazed_img_np = dehazed_img.squeeze(0).cpu().numpy().transpose((1, 2, 0))
        if picsaze is True:
            imsave(storage_path + '/dehazed_{}.png'.format(i), dehazed_img_np)

        i += 1

    # 计算平均值和模型参数
    num_params = count_parameters(model)
    param_bytes = sum(p.numel() * p.element_size() for p in model.parameters())
    model_size_megabytes = param_bytes / (1024 ** 2)
    average_ssim = sum(ssim_values) / len(ssim_values)
    average_psnr = sum(psnr_values) / len(psnr_values)

    print(f'Average PSNR: {average_psnr:.4f}')
    print(f'Average SSIM: {average_ssim:.4f}')

    # 保存到txt文件（包含每张图片和平均值）
    with open(storage_path + txt_name, 'w') as f:
        # 写入每张图片的结果
        f.write('Per Image Results:\n')
        f.write('-' * 40 + '\n')
        for result in image_results:
            f.write(result + '\n')

        # 写入统计信息
        f.write('\n' + '=' * 40 + '\n')
        f.write(f'Number of trainable parameters: {num_params:,}\n')
        f.write(f'Model size: {model_size_megabytes:.2f} MB\n')
        f.write(f'Average SSIM: {average_ssim:.4f}\n')
        f.write(f'Average PSNR: {average_psnr:.4f}\n')

    print(f'Number of trainable parameters: {num_params:,}')
    print(f'Model size: {model_size_megabytes:.2f} MB')

model.load_state_dict(torch.load(model_path,map_location='cuda:0'))  # Replace with your model path
model.to('cuda')
# Set the model to evaluation mode
model.eval()

process_images(model, input_path, folder_path, Image_size)
print("All Images Processed")
