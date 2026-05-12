import torch
import torch.nn as nn
import attention_pixel
import DWT_Block


import attention_channel
import module
# from torchinfo  import summary
import torch
import torchvision



#wave


class feat2(nn.Module):
    def __init__(self, n_FEAB_Blocks=[1,1,1,1], sampling=2, in_ch=3, processing_ch=32):
        super(feat2, self).__init__()

        self.split_1 = module.WaveDownampler2(processing_ch,processing_ch)
        self.split_2 = module.WaveDownampler2(processing_ch,processing_ch)
        self.split_3 = module.WaveDownampler2(processing_ch,processing_ch)
        self.split_4 = module.WaveDownampler2(processing_ch,processing_ch)


        self.lf_proc_1 = module.MCFEBlock(processing_ch, 2)
        self.lf_proc_2 = module.MCFEBlock(processing_ch, 2)
        self.lf_proc_3 = module.MCFEBlock(processing_ch, 2)
        self.lf_proc_4 = module.MCFEBlock(processing_ch, 2)

        self.hf_proc_1=module.HFPv2(in_channels=processing_ch, out_channels=processing_ch)
        self.hf_proc_2=module.HFPv2(in_channels=processing_ch, out_channels=processing_ch)
        self.hf_proc_3=module.HFPv2(in_channels=processing_ch, out_channels=processing_ch)
        self.hf_proc_4=module.HFPv2(in_channels=processing_ch, out_channels=processing_ch)

        self.hf_down_1=module.HF_Downv2(processing_ch)
        self.hf_down_2=module.HF_Downv2(processing_ch)
        self.hf_down_3=module.HF_Downv2(processing_ch)

        self.fusion1 = module.DWF(dim=processing_ch, height=2, reduction=8)#height: 表示输入特征图的数量
        self.fusion2 = module.DWF(dim=processing_ch, height=3, reduction=8)#height: 表示输入特征图的数量
        self.fusion3 = module.DWF(dim=processing_ch, height=3, reduction=8)#height: 表示输入特征图的数量

        self.fusion4 = module.DWF(dim=processing_ch, height=3, reduction=8)

        self.down1 = nn.Conv2d(3, processing_ch, kernel_size=3,stride=1, padding=1)
    def forward(self, x):
        # Splitting and processing the low and high frequency components
        x = self.down1(x)


        x1_l, x1_h = self.split_1(x)#[[-1, 32, 256, 256], [-1, 32, 256, 256]]

        x1_low=self.lf_proc_1(x1_l)#[-1, 32, 256, 256]


        x1_high=self.hf_proc_1(x1_h)#[-1, 32, 256, 256]

        x1_high_down=self.hf_down_1(x1_high)#[-1, 32, 64, 64]


        x2_l, x2_h = self.split_2(x1_low)# [[-1, 32, 64, 64], [-1, 32, 64, 64]]
        x2_low=self.lf_proc_2(x2_l)#[-1, 32, 64, 64]

        x2_high_fuse=x2_h+x1_high_down
        x2_high=self.hf_proc_2(x2_high_fuse)
        x2_high_down=self.hf_down_2(x2_high)

        x3_l, x3_h = self.split_3(x2_low)
        x3_low = self.lf_proc_3(x3_l)
        x3_high_fuse=x3_h+x2_high_down
        x3_high=self.hf_proc_3(x3_high_fuse)
        x3_high_down=self.hf_down_3(x3_high)

        x4_l, x4_h = self.split_4(x3_low)
        x4_low = self.lf_proc_4(x4_l)
        x4_high_fuse=x4_h+x3_high_down
        x4_high=self.hf_proc_4(x4_high_fuse)


        y4 = self.fusion1(x4_high, x4_low)

        y3 = self.fusion2(y4, x3_high, x3_low)

        y2 = self.fusion3(y3, x2_high, x2_low)

        y1 = self.fusion4(y2, x1_high, x1_low)

        #融合模块
        return y1

class Post_Proc_Module(nn.Module):
    def __init__(self, in_ch=32):
        super(Post_Proc_Module, self).__init__()
        self.channel_attention=attention_channel.Efficient_Channel_Attention(in_ch)
        self.pixel_attention=attention_pixel.Efficient_Pixel_Attention(in_ch)
        self.final_conv=nn.Conv2d(in_ch, in_ch//2, kernel_size=3, padding=1)
        self.final_conv1=nn.Conv2d(in_ch//2, 3, kernel_size=3, padding=1)
        self.relu=nn.ReLU()
        self.tanh=nn.Tanh()
        # self.deconv = FastDeconv(3, 3, kernel_size=3, stride=1, padding=1)
    def forward(self, x,y):


        x=self.channel_attention(x)
        x=self.pixel_attention(x)
        x=self.final_conv(x)
        x=self.relu(x)
        x=self.final_conv1(x)
        x=x+y#1
        x=self.tanh(x)
        return x
class Dehazing_Model(nn.Module):
    def __init__(self, n_FEAB_Blocks=[3,3,3,3], sampling=2, channels=32):
        super(Dehazing_Model, self).__init__()
        self.freq_proc=feat2(n_FEAB_Blocks=n_FEAB_Blocks, sampling=sampling, in_ch=3, processing_ch=channels)
        self.post_proc=Post_Proc_Module(in_ch=channels)

    def forward(self, x):
        x1=self.freq_proc(x)
        x2 = self.post_proc(x1, x)

        return x2
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


from torchinfo  import summary
print('==> Building model..')
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = Dehazing_Model().to(device)
# input_tensor = torch.randn(1, 3, 256, 256).to(device)
# print(model)
summary(model, (1,3,256,256))

num_params = count_parameters(model)
print(f'Number of trainable parameters: {num_params:,}\n')
num_params = count_parameters(model)
param_bytes = sum(p.numel() * p.element_size() for p in model.parameters())
model_size_megabytes = param_bytes / (1024**2)


print(f'Number of trainable parameters: {num_params:,}')
print(f'Model size: {model_size_megabytes:.2f} MB')