import torch
import torch.nn as nn
import FCB_class
import Attention_Block
import attention_channel
import attention_pixel
import attention_spatial
import torch.nn.functional as F
import DWT_Block
import MyDWT_Block
from mamba import SS2DBlock,LayerNorm2d
from torchinfo import summary
from pytorch_wavelets import DWTForward,DWTInverse
from einops import rearrange
from torch_wavelets import DWT_2D, IDWT_2D
import typing as t
import CGA
from timm.models.layers import DropPath
import numbers

class IWT(nn.Module):
    def __init__(self):
        super(IWT, self).__init__()
        self.requires_grad = False

    def forward(self, x):
        return iwt_init(x)

def iwt_init(x):
    r = 2
    in_batch, in_channel, in_height, in_width = x.size()
    out_batch, out_channel, out_height, out_width = in_batch,int(in_channel/(r**2)), r * in_height, r * in_width
    x1 = x[:, :out_channel, :, :] / 2
    x2 = x[:,out_channel:out_channel * 2, :, :] / 2
    x3 = x[:,out_channel * 2:out_channel * 3, :, :] / 2
    x4 = x[:,out_channel * 3:out_channel * 4, :, :] / 2

    h = torch.zeros([out_batch, out_channel, out_height,
                     out_width]).float().to(x.device)

    h[:, :, 0::2, 0::2] = x1 - x2 - x3 + x4
    h[:, :, 1::2, 0::2] = x1 - x2 + x3 - x4
    h[:, :, 0::2, 1::2] = x1 + x2 - x3 - x4
    h[:, :, 1::2, 1::2] = x1 + x2 + x3 + x4

    return h






class HFPv2(nn.Module):
    def __init__(self, in_channels,out_channels):
        super().__init__()
        # 方案1: 1×1 卷积（最轻量，仅调整通道）
        self.conv1x1 = nn.Conv2d(in_channels, out_channels, 1)

        # 方案2: 3×3 深度可分离卷积（轻量+局部纹理）
        self.dwconv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1, groups=out_channels),  # Depthwise
            nn.Conv2d(in_channels, out_channels, 1),  # Pointwise
        )

        # 通道注意力增强纹理响应
        self.attn = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, out_channels // 4, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // 4, out_channels, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        # 残差连接，避免过度处理
        identity = x
        out = self.dwconv(x)  # 或 self.conv1x1(x)
        out = out * self.attn(out)  # 增强显著纹理通道
        return out + identity  # 保留原始高频信息

class HF_Downv2(nn.Module):
    def __init__(self, channels, sampling=2):
        super(HF_Downv2, self).__init__()
        # 用 AvgPool 下采样 → 无参 + 抗 aliasing
        # self.pool = nn.AvgPool2d(sampling, stride=sampling)
        # 深度可分离 3×3
        self.dw = nn.Conv2d(channels, channels, 3, padding=1,
                            groups=channels, bias=False)
        self.pw = nn.Conv2d(channels, channels, 1, bias=False)
        self.fcb = FCB_class.FCB(channels, channels)   # 保留你原来的 FCB
        self.conv = nn.Conv2d(channels, channels, kernel_size=3, stride=sampling, padding=1)

    def forward(self, x):
        x = self.conv(x)
        x = self.pw(self.dw(x))
        x = self.fcb(x)
        return  x
class HF_Up(nn.Module):
    def __init__(self, channels):
        super(HF_Up, self).__init__()
        self.fcb_no_act = FCB_class.FCB_No_Act(channels, channels)
        self.pixel_attention = attention_pixel.Efficient_Pixel_Attention(channels)
        self.up = nn.ConvTranspose2d(channels, channels, kernel_size=3, stride=2, padding=1, output_padding=1,
                                     bias=False)
        self.fcb2 = FCB_class.FCB(channels, channels)

    def forward(self, x2_h, x2_l):
        x = x2_h + x2_l
        x = self.fcb_no_act(x)
        x = self.pixel_attention(x)
        x = self.up(x)
        x = self.fcb2(x)
        return x


class Fusion_Up(nn.Module):
    def __init__(self, channels):
        super(Fusion_Up, self).__init__()
        self.fcb_no_act = FCB_class.FCB_No_Act(channels, channels)
        self.fcb2 = FCB_class.FCB(channels, channels)
        self.pixel_attention = attention_pixel.Efficient_Pixel_Attention(channels)
        self.up = nn.ConvTranspose2d(channels, channels, kernel_size=3, stride=2, padding=1, output_padding=1,
                                     bias=False)

    def forward(self, x_prev, x_hf, x_lf):
        x = x_prev + x_hf
        x = x + x_lf
        x = self.fcb_no_act(x)
        x = self.pixel_attention(x)
        x = self.up(x)
        x = self.fcb2(x)
        return x





class DWF(nn.Module):
    def __init__(self, dim, height=2, reduction=8,upsampe = True):
        super(DWF, self).__init__()

        self.upsampe = upsampe
        self.height = height
        d = max(int(dim/reduction), 4)
        print("d",d)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.mlp = nn.Sequential(
            nn.Conv2d(dim, d, 1, bias=False),
            nn.ReLU(),
            nn.Conv2d(d, dim * height, 1, bias=False)

        )
        self.up = nn.ConvTranspose2d(dim, dim, kernel_size=2, stride=2, padding=0, bias=False)  # 使用转置卷积
        self.fcb2 = FCB_class.FCB(dim, dim)
        self.softmax = nn.Softmax(dim=1)
        self.pixel_attention = attention_pixel.Efficient_Pixel_Attention(dim)
        self.pad = nn.ReflectionPad2d(1)
        self.conv = nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=0)
        # self.FeedForward = FeedForward(dim=32, ffn_expansion_factor=2.66, bias=False)
    def forward(self, *in_feats):
        # print("in_feats",in_feats[0].shape)
        B, C, H, W = in_feats[0].shape

        in_feats = torch.cat(in_feats, dim=1)

        in_feats = in_feats.view(B, self.height, C, H, W)
        print(in_feats.shape)
        feats_sum = torch.sum(in_feats, dim=1)
        print(feats_sum.shape)
        attn = self.mlp(self.avg_pool(feats_sum))

        attn = self.softmax(attn.view(B, self.height, C, 1, 1))
        print(attn.shape)
        x = in_feats * attn
        print(x.shape)
        out = torch.sum(in_feats*attn, dim=1)
        #after addd
        out = self.pixel_attention(out)
        # out = self.up(out)
        # print("out1", out.shape)
        if self.upsampe:
            out = self.up(out)
        else:
            out = self.pad(out)
            out = self.conv(out)
        out = self.fcb2(out)#=========version1===============
        # out = self.FeedForward(out)#===========version2===========
        # print("out2",out.shape)
        return out




def normalize(tensor):
    return tensor * 2 - 1


def denormalize(tensor):
    return (tensor + 1) / 2

class DWT(nn.Module):
    def __init__(self):
        super(DWT, self).__init__()
        self.requires_grad = False

    def forward(self, x):
        # 使用 DWTForward 计算离散小波变换
        return self.DWT(x)
#

class WaveDownampler2(nn.Module):
    def __init__(self, in_channels,out_channels):
        super().__init__()

        self.dwt = DWT_2D(wave='haar')
        # self.conv_lh = nn.Conv2d(in_channels, in_channels, 3, 1, 1, groups=in_channels)
        # self.conv_hl = nn.Conv2d(in_channels, in_channels, 3, 1, 1, groups=in_channels)
        # self.to_att = nn.Sequential(
        #             nn.Conv2d(in_channels, in_channels, 1, 1, 0),
        #             nn.Sigmoid()
        # )
        self.pw = nn.Conv2d(in_channels * 4, in_channels, 1, 1, 0)
        # 1. 逐频率 3×3 深度卷积
        # self.dw_freq = nn.Conv2d(in_channels*3, in_channels*3, 3, 1, 1, groups=in_channels*3)
        # 2. 逐子带 1×1（3 组）
        # self.pw_sub = nn.Conv2d(in_channels*3, out_channels, 1)
        self.conv1x1_high = nn.Conv2d(in_channels * 3, out_channels, kernel_size=1, padding=0)
        self.spatial_conv = nn.Conv2d(in_channels, out_channels, stride=2, kernel_size=3, padding=1)
        self.psi = nn.Sequential(
            nn.Conv2d(in_channels, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )
        self.W_g = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(in_channels)
        )

        self.W_x = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(in_channels)
        )
        self.relu = nn.ReLU(inplace=True)
    def forward(self, x):
        #

        x1 = self.spatial_conv(x)
        x_dwt = self.dwt(x)
        x_ll, x_lh, x_hl, x_hh = x_dwt.chunk(4, dim=1)


        lh = self.W_g(x_ll + x_lh)
        hl = self.W_x(x_ll + x_hl)
        psi = self.relu(lh + hl)

        A = self.psi(psi)# version2

        # squeeze
        x_s = self.pw(x_dwt)

        o = torch.mul(x_s, A) + x1

        hi_bands = torch.cat([x_lh, x_hl, x_hh], dim=1)
        hi_bands = self.conv1x1_high(hi_bands)

        return o, hi_bands


class MCFEBlock(nn.Module):
    def __init__(self, dim, ffn_scale=2.0):
        super().__init__()

        self.norm1 = LayerNorm(dim,'WithBias')
        self.norm2 = LayerNorm(dim,'WithBias')

        # Multiscale Block
        self.safm = MCFE(dim)
        # Feedforward layer
        self.ccm = CCM(dim, ffn_scale)

    def forward(self, x):
        x = self.safm(self.norm1(x)) + x

        return x

class AttBlock2(nn.Module):
    def __init__(self, dim, ffn_scale=2.0):
        super().__init__()

        self.norm1 = LayerNorm(dim,'WithBias')
        self.norm2 = LayerNorm(dim,'WithBias')

        # Multiscale Block
        self.safm = SAFM(dim)
        # Feedforward layer
        self.ccm = CCM(dim, ffn_scale)

    def forward(self, x):
        x = self.safm(self.norm1(x)) + x
        x = self.ccm(self.norm2(x)) + x
        return x

class CCM(nn.Module):
    def __init__(self, dim, growth_rate=2.0):
        super().__init__()
        hidden_dim = int(dim * growth_rate)

        self.ccm = nn.Sequential(
            nn.Conv2d(dim, dim, 3, 1, 1),
            nn.GELU(),
            nn.Conv2d(dim, dim, 1, 1, 0)
        )

    def forward(self, x):
        return self.ccm(x)

class MCFE(nn.Module):
    def __init__(self, dim, n_levels=4):
        super().__init__()
        self.n_levels = n_levels # 特征分块的层数
        chunk_dim = dim // n_levels # 每个分块的通道数

        # Spatial Weighting（空间加权模块）
        # 使用深度可分离卷积（groups=chunk_dim）对每个分块进行空间加权
        self.mfr = nn.ModuleList(
            [nn.Conv2d(chunk_dim, chunk_dim, 3, 1, 1, groups=chunk_dim) for i in range(self.n_levels)]
        )

        # Feature Aggregation（特征聚合模块）
        # 使用 1x1 卷积对所有分块的特征进行融合
        self.aggr = nn.Conv2d(dim, dim, 1, 1, 0)

        # Activation（激活函数）
        self.act = nn.GELU()

        self.norm1 = nn.BatchNorm2d(dim)
        self.drop_path = DropPath(0.1)
    def forward(self, x):
        # 获取输入特征的高度和宽度
        shortcut = x.clone()
        h, w = x.size()[-2:]

        # 将输入特征按通道维度均匀分割为 n_levels 个分块
        xc = x.chunk(self.n_levels, dim=1)

        # 存储每个分块的处理结果
        out = []
        for i in range(self.n_levels):
            if i > 0:
                # 对分块进行降采样，池化后的尺寸为原尺寸的 1 / (2^i)
                p_size = (h // 2 ** i, w // 2 ** i)
                s = F.adaptive_max_pool2d(xc[i], p_size) # 自适应最大池化
                s = self.mfr[i](s) # 空间加权
                # 将降采样后的特征插值回原始尺寸
                s = F.interpolate(s, size=(h, w), mode='nearest')
            else:
                # 第一个分块不进行降采样，直接进行空间加权
                s = self.mfr[i](xc[i])
            out.append(s) # 将处理后的分块特征添加到结果列表中

        # 将所有分块特征在通道维度上拼接，并通过特征聚合模块融合
        out = self.aggr(torch.cat(out, dim=1))

        # 激活函数，并与输入特征逐元素相乘（残差结构）
        out = self.act(out) * x#1
        # x_out = shortcut + self.norm1(self.drop_path(self.mlp_cmm(out)))#2
        return out

class CA(nn.Module):
    def __init__(self, channels):
        super(CA, self).__init__()
        self.AdaptiveAvgPool = nn.AdaptiveAvgPool2d((1, 1))
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        out = self.sigmoid(self.AdaptiveAvgPool(x)) # 生成权重: (B,d,H,W)--pool-->(B,d,1,1)--sigmoid-->(B,d,1,1)
        out = out * x # 使用通道权重调整每个通道的重要性: (B,d,1,1) * (B,d,H,W) == (B,d,H,W)
        return out

class BiasFree_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(BiasFree_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return x / torch.sqrt(sigma + 1e-5) * self.weight


class WithBias_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(WithBias_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        mu = x.mean(-1, keepdim=True)
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return (x - mu) / torch.sqrt(sigma + 1e-5) * self.weight + self.bias


class LayerNorm(nn.Module):
    def __init__(self, dim, LayerNorm_type):
        super(LayerNorm, self).__init__()
        if LayerNorm_type == 'BiasFree':
            self.body = BiasFree_LayerNorm(dim)
        else:
            self.body = WithBias_LayerNorm(dim)

    def forward(self, x):
        h, w = x.shape[-2:]
        return to_4d(self.body(to_3d(x)), h, w)


def to_3d(x):
    return rearrange(x, 'b c h w -> b (h w) c')


def to_4d(x, h, w):
    return rearrange(x, 'b (h w) c -> b c h w', h=h, w=w)







