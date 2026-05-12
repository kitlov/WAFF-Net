import torch
import os
from natsort import natsorted
from torch.utils.data import Dataset
from PIL import Image
from torchvision import transforms
from pathlib import Path
from torchvision.transforms import Resize
from torchvision.transforms import functional as TF
import random
import numpy as np
import cv2
from math import ceil

class MyNhazeData(Dataset):
    def __init__(self, path,image_size=(1024,1024)):
        # 加载模糊图片文件名
        self.filename_original = sorted(os.listdir(os.path.join(path, 'hazy')), key=len)
        # 加载清晰图片文件名
        self.filename_target = sorted(os.listdir(os.path.join(path, 'clear')), key=len)

        # print(len(self.filename_target))
        # print(len(self.filename_original))

        # 使用 natsorted 对文件名进行自然排序
        self.filename_original = natsorted(self.filename_original)
        self.filename_target = natsorted(self.filename_target)

        # 构建模糊图片和清晰图片的对应关系
        self.filename_original = [os.path.join(path, 'hazy', original) for original in self.filename_original]
        self.filename_target = [os.path.join(path, 'clear', target) for target in self.filename_target]
        self.image_size = image_size

    def __len__(self):
        return len(self.filename_original)

    def __getitem__(self, idx):
        # 获取模糊图片路径
        filename_o = self.filename_original[idx]
        # 加载并处理模糊图片
        real = Image.open(filename_o)
        real = transforms.Resize(self.image_size)(real)
        real = transforms.functional.to_tensor(real)

        # 获取清晰图片路径
        filename_t = self.filename_target[idx]
        # 加载并处理清晰图片
        condition = Image.open(filename_t)
        condition = transforms.Resize(self.image_size)(condition)
        condition = transforms.functional.to_tensor(condition)

        return real, condition
class MyITSData(Dataset):
    def __init__(self, path, image_size=(512, 512)):
        # 加载模糊图片文件名
        self.filename_original = sorted(os.listdir(os.path.join(path, 'hazy')), key=len)
        # 加载清晰图片文件名
        self.filename_target = sorted(os.listdir(os.path.join(path, 'clear')), key=len)

        # print(len(self.filename_target))
        # print(len(self.filename_original))

        # 使用 natsorted 对文件名进行自然排序
        self.filename_original = natsorted(self.filename_original)
        self.filename_target = natsorted(self.filename_target)

        # 构建模糊图片和清晰图片的对应关系
        self.filename_original = [os.path.join(path, 'hazy', original) for original in self.filename_original]
        self.filename_target = {target.split('.')[0]: os.path.join(path, 'clear', target) for target in
                                self.filename_target}

        self.image_size = image_size

    def __len__(self):
        return len(self.filename_original)

    def __getitem__(self, idx):
        # 获取模糊图片路径
        filename_o = self.filename_original[idx]

        # 解析模糊图片文件名，找到对应的清晰图片编号
        target_id = os.path.basename(filename_o).split('_')[0]

        filename_t = self.filename_target[target_id]

        # 定义图像预处理操作
        resize = transforms.Resize(self.image_size)
        # norm = transforms.Normalize([0.5], [0.5])

        # 加载并处理模糊图片
        real = Image.open(filename_o)
        real = resize(real)
        real = transforms.functional.to_tensor(real)
        # real = norm(real)

        # 加载并处理清晰图片
        condition = Image.open(filename_t)
        condition = resize(condition)
        condition = transforms.functional.to_tensor(condition)
        # condition = norm(condition)

        return real, condition

class Myhaze4kData(Dataset):
    def __init__(self, path, image_size=(512, 512)):
        # 加载模糊图片文件名
        self.filename_original = sorted(os.listdir(os.path.join(path, 'hazy')), key=len)
        # 加载清晰图片文件名
        self.filename_target = sorted(os.listdir(os.path.join(path, 'clear')), key=len)

        # print(len(self.filename_target))
        # print(len(self.filename_original))

        # 使用 natsorted 对文件名进行自然排序
        self.filename_original = natsorted(self.filename_original)
        self.filename_target = natsorted(self.filename_target)

        # 构建模糊图片和清晰图片的对应关系
        self.filename_original = [os.path.join(path, 'hazy', original) for original in self.filename_original]
        self.filename_target = {target.split('.')[0]: os.path.join(path, 'clear', target) for target in
                                self.filename_target}

        self.image_size = image_size

    def __len__(self):
        return len(self.filename_original)

    def __getitem__(self, idx):
        # 获取模糊图片路径
        filename_o = self.filename_original[idx]

        # 解析模糊图片文件名，找到对应的清晰图片编号
        target_id = os.path.basename(filename_o).split('_')[0]

        filename_t = self.filename_target[target_id]

        # 定义图像预处理操作
        # resize = transforms.Resize(self.image_size)
        # norm = transforms.Normalize([0.5], [0.5])

        # 加载并处理模糊图片
        real = Image.open(filename_o)
        w, h = real.size

        # 1. 短边缩到 ≥256 且是 16 的倍数
        short_tgt = max(256, (min(h, w) // 16) * 16)
        ratio = short_tgt / min(h, w)
        new_w, new_h = int(w * ratio), int(h * ratio)
        real = real.resize((new_w, new_h), Image.LANCZOS)

        # 2. 中心裁成 16 倍数（最多去掉 15 像素）
        crop_w, crop_h = new_w // 16 * 16, new_h // 16 * 16
        real = transforms.CenterCrop((crop_h, crop_w))(real)

        real = transforms.ToTensor()(real)
        # real = resize(real)
        # real = transforms.functional.to_tensor(real)
        # real = norm(real)

        # 加载并处理清晰图片
        condition = Image.open(filename_t)
        condition = transforms.CenterCrop((crop_h, crop_w))(condition)
        condition = transforms.ToTensor()(condition)
        # condition = resize(condition)
        # condition = transforms.functional.to_tensor(condition)
        # condition = norm(condition)

        return real, condition

class MyData(Dataset): #In use
    def __init__(self, path, image_size=(512,512),change_imagesize=False):
        self.filename_original =sorted(os.listdir(path+'//hazy'), key=len)
        self.filename_target = sorted(os.listdir(path+'//clear'), key=len)
        
        self.filename_original=natsorted(self.filename_original)
        self.filename_target=natsorted(self.filename_target)

        i=0
        while i<len(self.filename_original):
            self.filename_original[i]=path+'/hazy/'+self.filename_original[i]
            self.filename_target[i]=path+'/clear/'+self.filename_target[i]
            i+=1
        
        self.image_size=image_size

        self.change_imagesize = change_imagesize
    def __len__(self):
        return len(self.filename_original)
    
    def __getitem__(self,idx):
        # filename_o=self.filename_original[idx]
        # filename_t=self.filename_target[idx]
        real = Image.open(self.filename_original[idx])
        condition = Image.open(self.filename_target[idx])
        # 1. 先保证能被 16 整除且为偶数
        W, H = real.size
        # newH = int(ceil(H / 16) * 16)  # 向上取到 16 倍数
        # newW = int(ceil(W / 16) * 16)
        newH = (H // 16) * 16  # 向上取到 16 倍数
        newW = (W // 16) * 16
        if newH != H or newW != W:  # 需要调整才 resize
            real = real.resize((newW, newH), Image.BILINEAR)
            condition = condition.resize((newW, newH), Image.BILINEAR)
        
        # resize=transforms.Resize(self.image_size)

        #norm=transforms.Normalize([0.5], [0.5])
        
        # real=Image.open(filename_o)
        if self.change_imagesize == True:
            real=resize(real)
        
        # condition=Image.open(filename_t)
        if self.change_imagesize == True:
            condition=resize(condition)
        
        real=transforms.functional.to_tensor(real)
        #real=norm(real)
        condition=transforms.functional.to_tensor(condition)
        #condition=norm(condition)
        
        return real, condition




class MyData_Test(Dataset): #In use
    def __init__(self, path, resize_dimen=(1024,1024)):
        self.filename_original = sorted(os.listdir(path+'//hazy'), key=len)
        self.filename_target = sorted(os.listdir(path+'//clear'), key=len)
        
        self.filename_original = natsorted(self.filename_original)
        self.filename_target = natsorted(self.filename_target)
        
        i = 0
        while i < len(self.filename_original):
            self.filename_original[i] = path+'/hazy/'+self.filename_original[i]
            self.filename_target[i] = path+'/clear/'+self.filename_target[i]
            i += 1
        
        self.resize = Resize((resize_dimen))  # Add this line to initialize the Resize transform
    
    def __len__(self):
        return len(self.filename_original)
    
    def __getitem__(self, idx):
        filename_o = self.filename_original[idx]
        filename_t = self.filename_target[idx]
        
        real = Image.open(filename_o)
        condition = Image.open(filename_t)
        
        real = self.resize(real)  # Add this line to resize the 'real' image
        condition = self.resize(condition)  # Add this line to resize the 'condition' image
        
        real = transforms.functional.to_tensor(real)
        condition = transforms.functional.to_tensor(condition)
        
        return real, condition



class MyData_Test_Single(Dataset):
    def __init__(self, path, resize_dimen=(1024,1024)):
        # Initialize the filename list first
        self.filename_original = sorted(os.listdir(os.path.join(path, 'Hazy')), key=len)
        self.filename_original = natsorted(self.filename_original)
        
        # Update paths
        self.filename_original = [os.path.join(path, 'Hazy', filename) 
                                for filename in self.filename_original]
        
        # Initialize the resize transform
        self.resize = Resize(resize_dimen)
    
    def __len__(self):
        return len(self.filename_original)
    
    def __getitem__(self, idx):
        filename_o = self.filename_original[idx]
        
        # Open and process the image
        real = Image.open(filename_o).convert('RGB')
        real = self.resize(real)
        real = transforms.functional.to_tensor(real)
        
        return real, real  # Return twice to match the expected format in test.py



class Dense_haze_Train(Dataset):
    def __init__(self, path, image_size=(1024, 1024), patch_size=256, stride=128, transform=None):
        # 加载并自然排序模糊图片文件名
        self.filename_original = natsorted([f for f in os.listdir(os.path.join(path, 'hazy'))
                                            if f.endswith(('.jpg', '.jpeg', '.png'))])
        # 加载并自然排序清晰图片文件名
        self.filename_target = natsorted([f for f in os.listdir(os.path.join(path, 'clear'))
                                          if f.endswith(('.jpg', '.jpeg', '.png'))])

        # 确保两个列表长度相同
        assert len(self.filename_original) == len(self.filename_target), "Number of hazy and clear images must match"

        # 构建完整路径
        self.filename_original = [os.path.join(path, 'hazy', f) for f in self.filename_original]
        self.filename_target = [os.path.join(path, 'clear', f) for f in self.filename_target]

        # 定义参数
        self.image_size = image_size
        self.patch_size = patch_size
        self.stride = stride
        self.transform = transform

        # 预计算每个图像的补丁数量
        self.patches_per_image = self._calculate_patches_per_image()

    def _calculate_patches_per_image(self):
        """计算每张图像可以提取的补丁数量"""
        h, w = self.image_size
        n_h = (h - self.patch_size) // self.stride + 1
        n_w = (w - self.patch_size) // self.stride + 1
        return n_h * n_w

    def __len__(self):
        # 总样本数 = 图像数量 × 每张图像的补丁数
        # print(len(self.filename_original) * self.patches_per_image)
        return len(self.filename_original) * self.patches_per_image

    def __getitem__(self, idx):
        # 计算对应的图像索引和补丁索引
        img_idx = idx // self.patches_per_image
        patch_idx = idx % self.patches_per_image

        # 加载图像
        real = Image.open(self.filename_original[img_idx]).convert('RGB')
        condition = Image.open(self.filename_target[img_idx]).convert('RGB')

        # 调整图像大小
        resize = transforms.Resize(self.image_size)
        real = resize(real)
        condition = resize(condition)

        # 转换为Tensor
        real = TF.to_tensor(real)
        condition = TF.to_tensor(condition)

        # 计算补丁位置
        h, w = self.image_size
        n_w = (w - self.patch_size) // self.stride + 1
        i = (patch_idx // n_w) * self.stride
        j = (patch_idx % n_w) * self.stride
        # print("h,w,n_w,i,j",h,w,n_w,i,j)
        _, H, W = real.shape
        i_max = H - self.patch_size
        j_max = W - self.patch_size
        i = min(i, i_max)
        j = min(j, j_max)
        # 提取补丁
        patch_real = real[:, i:i + self.patch_size, j:j + self.patch_size]
        patch_condition = condition[:, i:i + self.patch_size, j:j + self.patch_size]
        # print("patchlen",len(patch_condition))
        # 应用数据增强变换
        if self.transform:
            patch_real = self.transform(patch_real)
            patch_condition = self.transform(patch_condition)
        # print(len(patch_real))
        return patch_real, patch_condition


class data_Train_Aug(Dataset):
    def __init__(self, path, image_size=(1600, 1200), patch_size=256, stride=128, aug_multiplier=10):
        # 加载并自然排序模糊图片文件名
        self.filename_original = natsorted([f for f in os.listdir(os.path.join(path, 'hazy'))
                                            if f.endswith(('.jpg', '.jpeg', '.png'))])
        # 加载并自然排序清晰图片文件名
        self.filename_target = natsorted([f for f in os.listdir(os.path.join(path, 'clear'))
                                          if f.endswith(('.jpg', '.jpeg', '.png'))])

        # 确保两个列表长度相同
        assert len(self.filename_original) == len(self.filename_target), "Number of hazy and clear images must match"

        # 构建完整路径
        self.filename_original = [os.path.join(path, 'hazy', f) for f in self.filename_original]
        self.filename_target = [os.path.join(path, 'clear', f) for f in self.filename_target]

        # 定义参数
        self.image_size = image_size
        self.patch_size = patch_size
        self.stride = stride

        self.aug_multiplier = aug_multiplier
        # 预计算每个图像的补丁数量
        self.patches_per_image = self._calculate_patches_per_image()
        # print("每个图像的补丁数量", self.patches_per_image)

        self.PairedTransform = PairedTransformDehazeFlip()

    def _calculate_patches_per_image(self):
        """计算每张图像可以提取的补丁数量"""
        h, w = self.image_size
        # print(h,w)
        n_h = (h - self.patch_size) // self.stride + 1
        n_w = (w - self.patch_size) // self.stride + 1
        # print("n_h,n_w",n_h,n_w)
        return n_h * n_w

    def __len__(self):
        # 总样本数 = 图像数量 × 每张图像的补丁数
        # print("总样本数",len(self.filename_original) * self.patches_per_image)
        # return len(self.filename_original) * self.patches_per_image
        print("总样本数", len(self.filename_original) * self.patches_per_image * self.aug_multiplier)
        return len(self.filename_original) * self.patches_per_image * self.aug_multiplier

    def __getitem__(self, idx):
        # 计算对应的图像索引和补丁索引
        #
        base_idx = idx // self.aug_multiplier
        img_idx = base_idx // self.patches_per_image
        patch_idx = base_idx % self.patches_per_image
        # 加载图像
        hazy = Image.open(self.filename_original[img_idx]).convert('RGB')
        clear = Image.open(self.filename_target[img_idx]).convert('RGB')

        # 调整图像大小
        # resize = transforms.Resize(self.image_size)
        # hazy = resize(hazy)
        # clear = resize(clear)

        # 裁补丁
        hazy_patch = self._crop_patch(hazy, patch_idx)
        clear_patch = self._crop_patch(clear, patch_idx)

        hazy, clear = self.PairedTransform(hazy_patch, clear_patch)

        # hazy, clear =  self.aug(hazy_patch, clear_patch)
        # hazy = np.array(hazy_patch)  # PIL → numpy(H,W,3)
        # clear = np.array(clear_patch)
        # # print("train",hazy.shape)
        # transformed  = paired_aug(image=hazy,clear=clear)
        # hazy_A = transformed['image']
        # clear_A = transformed['clear']

        # print("hazy_A",hazy_A.shape)

        hazy = TF.to_tensor(hazy)
        clear = TF.to_tensor(clear)
        return hazy, clear

class patched_data_Train_Aug(Dataset):
    def __init__(self, path,  aug_multiplier=2):
        # 1. 直接扫“小图”文件夹（提前切好）
        self.filename_original = natsorted([f for f in os.listdir(os.path.join(path, 'hazy'))
                                            if f.endswith(('.jpg', '.jpeg', '.png'))])
        # 加载并自然排序清晰图片文件名
        self.filename_target = natsorted([f for f in os.listdir(os.path.join(path, 'clear'))
                                          if f.endswith(('.jpg', '.jpeg', '.png'))])
        assert len(self.filename_original) == len(self.filename_target), "小图数量必须一致"

        # 构建完整路径
        self.filename_original = [os.path.join(path, 'hazy', f) for f in self.filename_original]
        self.filename_target = [os.path.join(path, 'clear', f) for f in self.filename_target]


        self.aug_multiplier = aug_multiplier
        self.PairedTransform = PairedTransformDehazeFlip()   # 你的翻转/旋转类

    def __len__(self):
        # print(len(self.filename_original) * self.aug_multiplier)
        return len(self.filename_original) * self.aug_multiplier

    def __getitem__(self, idx):
        base_idx = idx // self.aug_multiplier
        hazy = Image.open(self.filename_original[base_idx]).convert('RGB')
        clear = Image.open(self.filename_target[base_idx]).convert('RGB')

        # 同步增强
        hazy, clear = self.PairedTransform(hazy, clear)

        return TF.to_tensor(hazy), TF.to_tensor(clear)
# class PairedRandomHorizontalFlip:
#     def __init__(self, p=0.5):
#         self.p = p
#
#     def __call__(self, img1, img2):
#         if random.random() < self.p:
#             img1 = TF.hflip(img1)
#             img2 = TF.hflip(img2)
#         return img1, img2
class PairedRandomRotate:
    def __init__(self, angles=(0, 90, 180, 270)):
        self.angles = list(angles)          # 支持任意角度集合

    def __call__(self, img1, img2):
        angle = random.choice(self.angles)
        if angle == 0:
            return img1, img2
        # TF.rotate 负号表示顺时针
        img1 = TF.rotate(img1, -angle, interpolation=TF.InterpolationMode.BILINEAR)
        img2 = TF.rotate(img2, -angle, interpolation=TF.InterpolationMode.BILINEAR)
        return img1, img2
class PairedTransformDehazeFlip:
    def __init__(self, hflip_p=0.5, **kwargs):
        self.flip = PairedRandomRotate(angles=(0,90,180,270))
        # self.dehaze = PairedTransformDehaze(**kwargs)

    def __call__(self, hazy, clear):
        hazy, clear = self.flip(hazy, clear)      # 1. 同步翻转
        # hazy, clear = self.dehaze(hazy, clear)    # 2. 其余去雾增强
        return hazy, clear
class PairedTransformDehaze:
    """
    去雾专用成对增强
    输入/输出：PIL Image → PIL Image
    """
    def __init__(self,
                 # sync_brightness=0.15,   # 同步亮度
                 # sync_contrast=0.15,     # 同步对比度
                 clahe_p=0.3,            # CLAHE 概率
                 # channel_shuffle_p=0.2,  # 通道置换
                 # synthetic_fog_p=0.4,    # 给 clear 加雾
                 noise_p=0.3):           # 仅 hazy 加噪声
        # self.sync_brightness = sync_brightness
        # self.sync_contrast   = sync_contrast
        self.clahe_p         = clahe_p
        # self.channel_shuffle_p = channel_shuffle_p
        # self.synthetic_fog_p = synthetic_fog_p
        self.noise_p         = noise_p

        # 一次性生成随机参数
        self._reset_params()

    # ---------- 随机参数 ----------
    def _reset_params(self):
        # self.alpha_b = random.uniform(1-self.sync_brightness, 1+self.sync_brightness)
        # self.alpha_c = random.uniform(1-self.sync_contrast,   1+self.sync_contrast)
        self.do_clahe   = random.random() < self.clahe_p
        # self.do_shuffle = random.random() < self.channel_shuffle_p
        # self.do_fog     = random.random() < self.synthetic_fog_p
        self.do_noise   = random.random() < self.noise_p
        # self.beta       = random.uniform(0.6, 1.2)   # 雾浓度
        # self.A          = random.uniform(0.7, 1.0)   # 全局大气光

    # ---------- 核心调用 ----------
    def __call__(self, hazy, clear):
        self._reset_params()

        # # 1. 同步亮度/对比度
        # hazy  = TF.adjust_brightness(hazy, self.alpha_b)
        # clear = TF.adjust_brightness(clear, self.alpha_b)
        # hazy  = TF.adjust_contrast(hazy, self.alpha_c)
        # clear = TF.adjust_contrast(clear, self.alpha_c)

        # 2. 同步 CLAHE（直方图均衡）
        if self.do_clahe:
            hazy  = self._clahe(hazy)
            clear = self._clahe(clear)

        # # 3. 同步通道随机置换
        # if self.do_shuffle:
        #     hazy  = self._channel_shuffle(hazy)
        #     clear = self._channel_shuffle(clear)

        # # 4. 给 clear 加合成雾（仅 clear → 更浓雾）
        # if self.do_fog:
        #     clear = self._add_fog(clear, beta=self.beta, A=self.A)

        # 5. 仅 hazy 加随机噪声（模拟相机噪声）
        if self.do_noise:
            hazy = self._add_noise(hazy)

        return hazy, clear

    # ---------- 子函数 ----------
    @staticmethod
    def _clahe(img: Image.Image) -> Image.Image:
        img_np = np.array(img)
        lab = cv2.cvtColor(img_np, cv2.COLOR_RGB2LAB)
        lab[:, :, 0] = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(lab[:, :, 0])
        img_np = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
        return Image.fromarray(img_np)

    # @staticmethod
    # def _channel_shuffle(img: Image.Image) -> Image.Image:
    #     channels = img.split()
    #     random.shuffle(channels)
    #     return Image.merge('RGB', channels)

    # @staticmethod
    # def _add_fog(img: Image.Image, beta=1.0, A=0.8) -> Image.Image:
    #     """经典大气退化模型 I(x)=J(x)t(x)+A(1-t(x))"""
    #     img_np = np.array(img) / 255.0
    #     h, w, _ = img_np.shape
    #     # 随机深度图 → 透射率
    #     depth = np.random.rand(h, w).astype(np.float32)
    #     depth = cv2.GaussianBlur(depth, (21, 21), 0)
    #     t = np.exp(-beta * depth)
    #     t = np.stack([t]*3, axis=-1)
    #     fog = img_np * t + A * (1 - t)
    #     return Image.fromarray((fog * 255).astype(np.uint8))

    @staticmethod
    def _add_noise(img: Image.Image, sigma=0.02) -> Image.Image:
        img_np = np.array(img) / 255.0
        noise = np.random.randn(*img_np.shape) * sigma
        noisy = np.clip(img_np + noise, 0, 1)
        return Image.fromarray((noisy * 255).astype(np.uint8))