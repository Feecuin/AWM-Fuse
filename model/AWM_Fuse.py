import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
from pdb import set_trace as stx
import numbers
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
from mamba_ssm.ops.selective_scan_interface import selective_scan_fn, selective_scan_ref
from einops import rearrange
import math
from typing import Optional, Callable
from einops import rearrange, repeat
from functools import partial
import torchvision
from pytorch_wavelets import DWTForward, DWTInverse  # (or import DWT, IDWT)
import time
import torchvision.utils as vutils
import pywt
import pywt.data
from functools import partial
import clip
NEG_INF = -1000000
device_id0 = 'cuda:0'
device_id1 = 'cuda:1'

class WTConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=5, stride=1, bias=True, wt_levels=1, wt_type='db1'):
        super(WTConv2d, self).__init__()

        # assert in_channels == out_channels

        self.in_channels = in_channels
        self.wt_levels = wt_levels
        self.stride = stride
        self.dilation = 1

        self.wt_filter, self.iwt_filter = wavelet.create_wavelet_filter(wt_type, in_channels, in_channels, torch.float)
        self.wt_filter = nn.Parameter(self.wt_filter, requires_grad=False)
        self.iwt_filter = nn.Parameter(self.iwt_filter, requires_grad=False)

        self.wt_function = partial(wavelet.wavelet_transform, filters = self.wt_filter)
        self.iwt_function = partial(wavelet.inverse_wavelet_transform, filters = self.iwt_filter)

        self.base_conv = nn.Conv2d(in_channels, in_channels, kernel_size, padding='same', stride=1, dilation=1, groups=in_channels, bias=bias)
        self.base_scale = _ScaleModule([1,in_channels,1,1])

        self.wavelet_convs = nn.ModuleList(
            [nn.Conv2d(in_channels*4, in_channels*4, kernel_size, padding='same', stride=1, dilation=1, groups=in_channels*4, bias=False) for _ in range(self.wt_levels)]
        )
        self.wavelet_scale = nn.ModuleList(
            [_ScaleModule([1,in_channels*4,1,1], init_scale=0.1) for _ in range(self.wt_levels)]
        )

        if self.stride > 1:
            self.stride_filter = nn.Parameter(torch.ones(in_channels, 1, 1, 1), requires_grad=False)
            self.do_stride = lambda x_in: F.conv2d(x_in, self.stride_filter, bias=None, stride=self.stride, groups=in_channels)
        else:
            self.do_stride = None

    def forward(self, x):

        x_ll_in_levels = []
        x_h_in_levels = []
        shapes_in_levels = []

        curr_x_ll = x

        for i in range(self.wt_levels):
            curr_shape = curr_x_ll.shape
            shapes_in_levels.append(curr_shape)
            if (curr_shape[2] % 2 > 0) or (curr_shape[3] % 2 > 0):
                curr_pads = (0, curr_shape[3] % 2, 0, curr_shape[2] % 2)
                curr_x_ll = F.pad(curr_x_ll, curr_pads)

            curr_x = self.wt_function(curr_x_ll)
            curr_x_ll = curr_x[:,:,0,:,:]
            
            shape_x = curr_x.shape
            curr_x_tag = curr_x.reshape(shape_x[0], shape_x[1] * 4, shape_x[3], shape_x[4])
            curr_x_tag = self.wavelet_scale[i](self.wavelet_convs[i](curr_x_tag))
            curr_x_tag = curr_x_tag.reshape(shape_x)

            x_ll_in_levels.append(curr_x_tag[:,:,0,:,:])
            x_h_in_levels.append(curr_x_tag[:,:,1:4,:,:])

        next_x_ll = 0

        for i in range(self.wt_levels-1, -1, -1):
            curr_x_ll = x_ll_in_levels.pop()
            curr_x_h = x_h_in_levels.pop()
            curr_shape = shapes_in_levels.pop()

            curr_x_ll = curr_x_ll + next_x_ll

            curr_x = torch.cat([curr_x_ll.unsqueeze(2), curr_x_h], dim=2)
            next_x_ll = self.iwt_function(curr_x)

            next_x_ll = next_x_ll[:, :, :curr_shape[2], :curr_shape[3]]

        x_tag = next_x_ll
        assert len(x_ll_in_levels) == 0
        
        x = self.base_scale(self.base_conv(x))
        x = x + x_tag
        
        if self.do_stride is not None:
            x = self.do_stride(x)

        return x

class _ScaleModule(nn.Module):
    def __init__(self, dims, init_scale=1.0, init_bias=0):
        super(_ScaleModule, self).__init__()
        self.dims = dims
        self.weight = nn.Parameter(torch.ones(*dims) * init_scale)
        self.bias = None
    
    def forward(self, x):
        return torch.mul(self.weight, x)

class wavelet(nn.Module):
    def create_wavelet_filter(wave, in_size, out_size, type=torch.float):
        w = pywt.Wavelet(wave)
        dec_hi = torch.tensor(w.dec_hi[::-1], dtype=type)
        dec_lo = torch.tensor(w.dec_lo[::-1], dtype=type)
        dec_filters = torch.stack([dec_lo.unsqueeze(0) * dec_lo.unsqueeze(1),
                                   dec_lo.unsqueeze(0) * dec_hi.unsqueeze(1),
                                   dec_hi.unsqueeze(0) * dec_lo.unsqueeze(1),
                                   dec_hi.unsqueeze(0) * dec_hi.unsqueeze(1)], dim=0)

        dec_filters = dec_filters[:, None].repeat(in_size, 1, 1, 1)

        rec_hi = torch.tensor(w.rec_hi[::-1], dtype=type).flip(dims=[0])
        rec_lo = torch.tensor(w.rec_lo[::-1], dtype=type).flip(dims=[0])
        rec_filters = torch.stack([rec_lo.unsqueeze(0) * rec_lo.unsqueeze(1),
                                   rec_lo.unsqueeze(0) * rec_hi.unsqueeze(1),
                                   rec_hi.unsqueeze(0) * rec_lo.unsqueeze(1),
                                   rec_hi.unsqueeze(0) * rec_hi.unsqueeze(1)], dim=0)

        rec_filters = rec_filters[:, None].repeat(out_size, 1, 1, 1)

        return dec_filters, rec_filters

    def wavelet_transform(x, filters):
        b, c, h, w = x.shape
        pad = (filters.shape[2] // 2 - 1, filters.shape[3] // 2 - 1)
        x = F.conv2d(x, filters, stride=2, groups=c, padding=pad)
        x = x.reshape(b, c, 4, h // 2, w // 2)
        return x

    def inverse_wavelet_transform(x, filters):
        b, c, _, h_half, w_half = x.shape
        pad = (filters.shape[2] // 2 - 1, filters.shape[3] // 2 - 1)
        x = x.reshape(b, c * 4, h_half, w_half)
        x = F.conv_transpose2d(x, filters, stride=2, groups=c, padding=pad)
        return x

class Decoder(nn.Module):
    def __init__(self, input_channel, output_channel):
        super(Decoder, self).__init__()
        self.waveconvs = nn.ModuleList([
            WTConv2d(input_channel, input_channel) for _ in range(8)
        ])
        self.leaky_relus = nn.ModuleList([
            nn.LeakyReLU(negative_slope=0.1, inplace=True) for _ in range(8)
        ])
        self.output = nn.Sequential(
            nn.Conv2d(input_channel, output_channel, kernel_size=3, padding=1),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
            nn.Conv2d(output_channel, output_channel, kernel_size=1),
            nn.GELU()
        )
        self.res = nn.Conv2d(input_channel, output_channel, kernel_size=1)

    def forward(self, x):
        r1 = self.res(x)
        residual = x
        for i in range(8):
            x = self.waveconvs[i](x)
            x = self.leaky_relus[i](x)
            x = x + residual
            residual = x

        x = self.output(x) + r1
        return x



class ResBlock_sign(nn.Module):
    def __init__(self, in_channels,out_channels):
        super(ResBlock_sign, self).__init__()
        self.layers = nn.Sequential(nn.Conv2d(in_channels, out_channels, 3, stride=1, padding=1),
                                    nn.LeakyReLU(negative_slope=0.2, inplace=True),
                                    nn.Conv2d(out_channels,out_channels, 3, stride=1, padding=1),
                                    nn.LeakyReLU(negative_slope=0.2, inplace=True)
                                    )
        self.conv_1x1 = nn.Conv2d(out_channels, out_channels, kernel_size=1, padding=0)
        self.GeLU = nn.GELU()

    def forward(self, x):
        out_x = self.layers(x)
        short_x = self.conv_1x1(out_x)
        out_x = out_x + short_x
        out_x = self.GeLU(out_x)
        return out_x



class SEBlock(nn.Module):
    def __init__(self, in_channels, reduction=16):
        super(SEBlock, self).__init__()
        self.global_avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Linear(in_channels, in_channels // reduction, bias=False)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Linear(in_channels // reduction, in_channels, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):

        batch_size, channels, _, _ = x.size()
        y = self.global_avg_pool(x).view(batch_size, channels)
        y = self.fc1(y)
        y = self.relu(y)
        y = self.fc2(y)
        y = self.sigmoid(y).view(batch_size, channels, 1, 1)

        return x * y

class LTPM(nn.Module):
    def __init__(self, in_channels):
        super(LTPM, self).__init__()
        self.se = SEBlock(in_channels, reduction=16)
        self.se2 = SEBlock(in_channels, reduction=16)
        self.cross_attention = local_CrossAttention(in_channels, num_heads=8)  # Reused attention layer
        self.text_process = local_text_preprocess(768,in_channels)
        self.imagef2textf = local_imagefeature2textfeature(in_channels, in_channels, in_channels)
        self.MSDC = local_MultiScaleDilatedConv(in_channels, in_channels)
        self.Fea_fuse = local_FeatureWiseAffine(512, in_channels)


    def forward(self, ClipImage , FuseImage, text, x_size):

        B, L, C = FuseImage.shape
        FuseImage = FuseImage.view(B, C, *x_size).contiguous()  # [B,H,W,C]
        text_feat = self.text_process(text)
        image_feat = self.se(FuseImage)
        image_feat = self.se2(image_feat)
        image_feat = self.Fea_fuse(image_feat, ClipImage)
        image_feat = self.imagef2textf(image_feat)
        Attn = self.cross_attention(text_feat, image_feat, image_feat)
        Final_feat = self._process_and_resize(Attn, image_feat, FuseImage.shape, FuseImage.shape[2:])
        Final_feat = self.MSDC(Final_feat)
        Final_feat = rearrange(Final_feat, "b c h w -> b (h w) c").contiguous()
        return Final_feat

    def _process_and_resize(self, attn_feat, fused_feat, original_shape, target_size):
        """Helper function to pool, normalize, and resize features."""
        _,C,H,W = original_shape
        attn_feat = F.adaptive_avg_pool1d(attn_feat.permute(0, 2, 1), 1).permute(0, 2, 1)
        attn_feat = F.normalize(attn_feat, p=1, dim=2)
        resized_feat = (fused_feat * attn_feat).view(original_shape[0], C, H, W)
        return F.interpolate(resized_feat, size=target_size, mode='nearest')

class local_FeatureWiseAffine(nn.Module):
    def __init__(self, in_channels, out_channels, use_affine_level=True):
        super(local_FeatureWiseAffine, self).__init__()
        self.use_affine_level = use_affine_level
        self.MLP = nn.Sequential(
            nn.Linear(in_channels, in_channels * 2),
            nn.LeakyReLU(),
            nn.Linear(in_channels * 2, out_channels * (1 + self.use_affine_level))
        )

    def forward(self, x, image_embed):
        image_embed = image_embed.unsqueeze(1)
        batch = x.shape[0]
        if self.use_affine_level:
            gamma, beta = self.MLP(image_embed).view(batch, -1, 1, 1).chunk(2, dim=1)
            x = (1 + gamma) * x + beta
        return x


class local_text_preprocess(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(local_text_preprocess, self).__init__()
        self.conv = nn.Conv1d(in_channel, out_channel, 1, 1, 0)

    def forward(self, x):
        x = self.conv(x.permute(0, 2, 1))
        return x.permute(0, 2, 1)

class local_imagefeature2textfeature(nn.Module):
    def __init__(self, in_channel, mid_channel, hidden_dim):
        super(local_imagefeature2textfeature, self).__init__()
        self.conv = nn.Conv2d(in_channels=in_channel, out_channels=mid_channel, kernel_size=1)
        self.hidden_dim = hidden_dim

    def forward(self, x):
        _,_,H,W = x.shape
        x = self.conv(x)
        x = F.interpolate(x, [H, W], mode='nearest')
        x = x.contiguous().view(x.size(0), x.size().numel() // x.size(0) // self.hidden_dim, self.hidden_dim)
        return x

class local_CrossAttention(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super(local_CrossAttention, self).__init__()
        self.multihead_attn = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=num_heads)

    def forward(self, query, key, value):
        query = query.transpose(0, 1)
        key = key.transpose(0, 1)
        value = value.transpose(0, 1)

        attn_output, _ = self.multihead_attn(query, key, value)
        attn_output = attn_output.transpose(0, 1)

        return attn_output



class local_MultiScaleDilatedConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(local_MultiScaleDilatedConv, self).__init__()

        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, dilation=1)
        self.conv2 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=2, dilation=2)
        self.conv3 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=3, dilation=3)

        self.conv1x1 = nn.Conv2d(out_channels * 3, out_channels, kernel_size=1)

    def forward(self, x):
        out1 = F.relu(self.conv1(x))  
        out2 = F.relu(self.conv2(x))  
        out3 = F.relu(self.conv3(x))  

        out = torch.cat([out1, out2, out3], dim=1)

        out = self.conv1x1(out)
        return out


class GTPM(nn.Module):
    def __init__(self, out_channels):
        super(GTPM, self).__init__()
        self.conv1 = ResidualBlock(3, out_channels)
        self.conv2 = ResidualBlock(1, out_channels)
        self.resconv1 = ResidualBlock(out_channels, out_channels)
        self.resconv2 = ResidualBlock(out_channels, out_channels)
        self.cross_attention = global_CrossAttention(embed_dim=out_channels, num_heads=8)  # Reused attention layer
        self.text_process = global_text_preprocess(out_channels)
        self.image_process = global_ImagePreprocess(out_channels)
        self.imagef2textf = global_imagefeature2textfeature(out_channels, out_channels)

    def forward(self, imageA, imageB, ClipImage, text): # imageA = VIS; imageB = IR;

        text_feat = self.text_process(text)
        imageA_feat = self.conv1(imageA)
        imageB_feat = self.conv2(imageB)

        # Split features into left and right halves
        L_A, R_A = torch.split(imageA_feat, imageA_feat.size(1) // 2, dim=1)
        L_B, R_B = torch.split(imageB_feat, imageB_feat.size(1) // 2, dim=1)

        # Left-side feature fusion and attention
        L_fused = self.resconv1(torch.cat((L_A, L_B), dim=1))
        L_fused = self.imagef2textf(L_fused)
        L_fused = L_fused + self.image_process(ClipImage)
        L_attn = self.cross_attention(text_feat, L_fused, L_fused)

        # Right-side feature fusion, pooling, and attention
        R_A = F.max_pool2d(R_A, kernel_size=2, stride=2)
        R_B = F.max_pool2d(R_B, kernel_size=2, stride=2)
        R_fused = self.resconv2(torch.cat((R_A, R_B), dim=1))
        R_fused = self.imagef2textf(R_fused)
        R_fused = R_fused + self.image_process(ClipImage)
        R_attn = self.cross_attention(text_feat, R_fused, R_fused)

        # Adaptive pooling, normalization, and resizing
        L_final = self._process_and_resize(L_attn, L_fused, imageA_feat.shape[2:])
        R_final = self._process_and_resize(R_attn, R_fused, imageA_feat.shape[2:])
        
        # feature fusion
        feature_fusion = (L_final + R_final) /2 + imageA_feat + imageB_feat
        return feature_fusion

    def _process_and_resize(self, attn_feat, fused_feat, target_size):
        attn_feat = F.normalize(attn_feat, p=1, dim=1)  # [batch_size, 64]
        fused = fused_feat * attn_feat
        fused = fused.unsqueeze(2).unsqueeze(3)
        resized_feat = F.interpolate(fused, size=target_size, mode='nearest')

        return resized_feat







class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResidualBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, padding=0)
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        identity = self.conv(x)
        out = self.conv1(x)
        out = self.relu(out)

        out = self.conv2(out)

        out += identity
        out = self.relu(out)

        return out

class global_text_preprocess(nn.Module):
    def __init__(self, out_channels):
        super(global_text_preprocess, self).__init__()
        self.conv1d = nn.Conv1d(in_channels=512, out_channels=out_channels, kernel_size=1)

    def forward(self, x):
        x = x.unsqueeze(2)
        x = self.conv1d(x)
        x = x.squeeze(2)
        return x

class global_ImagePreprocess(nn.Module):
    def __init__(self, out_channels):
        super(global_ImagePreprocess, self).__init__()
        self.conv1d = nn.Conv1d(in_channels=512, out_channels=out_channels, kernel_size=1)

    def forward(self, x):
        x = x.unsqueeze(2)
        x = self.conv1d(x)
        x = x.squeeze(2)
        return x



class global_imagefeature2textfeature(nn.Module):
    def __init__(self, in_channel, hidden_dim):
        super(global_imagefeature2textfeature, self).__init__()
        self.conv = nn.Conv2d(in_channels=in_channel, out_channels=hidden_dim, kernel_size=1)
        self.pool = nn.AdaptiveAvgPool2d(1)

    def forward(self, x):
        x = self.conv(x)
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        return x

class global_CrossAttention(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super(global_CrossAttention, self).__init__()
        self.multihead_attn = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=num_heads)

    def forward(self, query, key, value):
        query = query.unsqueeze(1)
        key = key.unsqueeze(1)
        value = value.unsqueeze(1)

        query = query.transpose(0, 1)
        key = key.transpose(0, 1)
        value = value.transpose(0, 1)
        attn_output, _ = self.multihead_attn(query, key, value)

        attn_output = attn_output.transpose(0, 1)

        attn_output = attn_output.squeeze(1)

        return attn_output






class ChannelAttention(nn.Module):
    """Channel attention used in RCAN.
    Args:
        num_feat (int): Channel number of intermediate features.
        squeeze_factor (int): Channel squeeze factor. Default: 16.
    """

    def __init__(self, num_feat, squeeze_factor=16):
        super(ChannelAttention, self).__init__()
        self.attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(num_feat, num_feat // squeeze_factor, 1, padding=0),
            nn.ReLU(inplace=True),
            nn.Conv2d(num_feat // squeeze_factor, num_feat, 1, padding=0),
            nn.Sigmoid())

    def forward(self, x):
        y = self.attention(x)
        return x * y


class CAB(nn.Module):

    def __init__(self, num_feat, compress_ratio=3, squeeze_factor=12):
        super(CAB, self).__init__()

        self.cab = nn.Sequential(
            nn.Conv2d(num_feat, num_feat // compress_ratio, 3, 1, 1),
            nn.GELU(),
            nn.Conv2d(num_feat // compress_ratio, num_feat, 3, 1, 1),
            ChannelAttention(num_feat, squeeze_factor)
            )

    def forward(self, x):
        return self.cab(x)




class SS2D(nn.Module):
    def __init__(
            self,
            d_model,
            d_state=16,
            d_conv=3,
            expand=2,
            dt_rank="auto",
            dt_min=0.001,
            dt_max=0.1,
            dt_init="random",
            dt_scale=1.0,
            dt_init_floor=1e-4,
            dropout=0.,
            conv_bias=True,
            bias=False,
            device=None,
            dtype=None,
            **kwargs,
    ):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand
        self.d_inner = int(self.expand * self.d_model)
        self.dt_rank = math.ceil(self.d_model / 16) if dt_rank == "auto" else dt_rank

        self.in_proj = nn.Linear(self.d_model, self.d_inner * 2, bias=bias, **factory_kwargs)
        self.conv2d = nn.Conv2d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            groups=self.d_inner,
            bias=conv_bias,
            kernel_size=d_conv,
            padding=(d_conv - 1) // 2,
            **factory_kwargs,
        )
        self.act = nn.SiLU()

        self.x_proj = (
            nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs),
            nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs),
            nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs),
            nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs),
        )
        self.x_proj_weight = nn.Parameter(torch.stack([t.weight for t in self.x_proj], dim=0))  # (K=4, N, inner)
        del self.x_proj

        self.dt_projs = (
            self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor,
                         **factory_kwargs),
            self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor,
                         **factory_kwargs),
            self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor,
                         **factory_kwargs),
            self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor,
                         **factory_kwargs),
        )
        self.dt_projs_weight = nn.Parameter(torch.stack([t.weight for t in self.dt_projs], dim=0))  # (K=4, inner, rank)
        self.dt_projs_bias = nn.Parameter(torch.stack([t.bias for t in self.dt_projs], dim=0))  # (K=4, inner)
        del self.dt_projs

        self.A_logs = self.A_log_init(self.d_state, self.d_inner, copies=4, merge=True)  # (K=4, D, N)
        self.Ds = self.D_init(self.d_inner, copies=4, merge=True)  # (K=4, D, N)

        self.selective_scan = selective_scan_fn

        self.out_norm = nn.LayerNorm(self.d_inner)
        self.out_proj = nn.Linear(self.d_inner, self.d_model, bias=bias, **factory_kwargs)
        self.dropout = nn.Dropout(dropout) if dropout > 0. else None

    @staticmethod
    def dt_init(dt_rank, d_inner, dt_scale=1.0, dt_init="random", dt_min=0.001, dt_max=0.1, dt_init_floor=1e-4,
                **factory_kwargs):
        dt_proj = nn.Linear(dt_rank, d_inner, bias=True, **factory_kwargs)

        # Initialize special dt projection to preserve variance at initialization
        dt_init_std = dt_rank ** -0.5 * dt_scale
        if dt_init == "constant":
            nn.init.constant_(dt_proj.weight, dt_init_std)
        elif dt_init == "random":
            nn.init.uniform_(dt_proj.weight, -dt_init_std, dt_init_std)
        else:
            raise NotImplementedError

        # Initialize dt bias so that F.softplus(dt_bias) is between dt_min and dt_max
        dt = torch.exp(
            torch.rand(d_inner, **factory_kwargs) * (math.log(dt_max) - math.log(dt_min))
            + math.log(dt_min)
        ).clamp(min=dt_init_floor)
        # Inverse of softplus: https://github.com/pytorch/pytorch/issues/72759
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        with torch.no_grad():
            dt_proj.bias.copy_(inv_dt)
        # Our initialization would set all Linear.bias to zero, need to mark this one as _no_reinit
        dt_proj.bias._no_reinit = True

        return dt_proj

    @staticmethod
    def A_log_init(d_state, d_inner, copies=1, device=None, merge=True):
        # S4D real initialization
        A = repeat(
            torch.arange(1, d_state + 1, dtype=torch.float32, device=device),
            "n -> d n",
            d=d_inner,
        ).contiguous()
        A_log = torch.log(A)  # Keep A_log in fp32
        if copies > 1:
            A_log = repeat(A_log, "d n -> r d n", r=copies)
            if merge:
                A_log = A_log.flatten(0, 1)
        A_log = nn.Parameter(A_log)
        A_log._no_weight_decay = True
        return A_log

    @staticmethod
    def D_init(d_inner, copies=1, device=None, merge=True):
        # D "skip" parameter
        D = torch.ones(d_inner, device=device)
        if copies > 1:
            D = repeat(D, "n1 -> r n1", r=copies)
            if merge:
                D = D.flatten(0, 1)
        D = nn.Parameter(D)  # Keep in fp32
        D._no_weight_decay = True
        return D

    def forward_core(self, x: torch.Tensor):
        B, C, H, W = x.shape
        L = H * W
        K = 4

        x_hwwh = torch.stack([x.view(B, -1, L), torch.transpose(x, dim0=2, dim1=3).contiguous().view(B, -1, L)], dim=1).view(B, 2, -1, L)
        xs = torch.cat([x_hwwh, torch.flip(x_hwwh, dims=[-1])], dim=1) # (1, 4, 192, 3136)
        x_dbl = torch.einsum("b k d l, k c d -> b k c l", xs.view(B, K, -1, L), self.x_proj_weight)
        dts, Bs, Cs = torch.split(x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=2)
        dts = torch.einsum("b k r l, k d r -> b k d l", dts.view(B, K, -1, L), self.dt_projs_weight)

        xs = xs.float().view(B, -1, L)
        dts = dts.contiguous().float().view(B, -1, L) # (b, k * d, l)
        Bs = Bs.float().view(B, K, -1, L)
        Cs = Cs.float().view(B, K, -1, L) # (b, k, d_state, l)
        Ds = self.Ds.float().view(-1)
        As = -torch.exp(self.A_logs.float()).view(-1, self.d_state)
        dt_projs_bias = self.dt_projs_bias.float().view(-1) # (k * d)

        out_y = self.selective_scan(
            xs, dts,
            As, Bs, Cs, Ds, z=None,
            delta_bias=dt_projs_bias,
            delta_softplus=True,
            return_last_state=False,
        ).view(B, K, -1, L)
        assert out_y.dtype == torch.float

        inv_y = torch.flip(out_y[:, 2:4], dims=[-1]).view(B, 2, -1, L)
        wh_y = torch.transpose(out_y[:, 1].view(B, -1, W, H), dim0=2, dim1=3).contiguous().view(B, -1, L)
        invwh_y = torch.transpose(inv_y[:, 1].view(B, -1, W, H), dim0=2, dim1=3).contiguous().view(B, -1, L)

        return out_y[:, 0], inv_y[:, 0], wh_y, invwh_y

    def forward(self, x: torch.Tensor, **kwargs):
        B, H, W, C = x.shape

        xz = self.in_proj(x)
        x, z = xz.chunk(2, dim=-1)
        x = x.permute(0, 3, 1, 2).contiguous()
        x = self.act(self.conv2d(x))
        y1, y2, y3, y4 = self.forward_core(x)
        assert y1.dtype == torch.float32
        y = y1 + y2 + y3 + y4
        y = torch.transpose(y, dim0=1, dim1=2).contiguous().view(B, H, W, -1)
        y = self.out_norm(y)
        y = y * F.silu(z)
        out = self.out_proj(y)
        if self.dropout is not None:
            out = self.dropout(out)
        return out


class VSSBlock(nn.Module):
    def __init__(
            self,
            hidden_dim: int = 0,
            drop_path: float = 0,
            norm_layer: Callable[..., torch.nn.Module] = partial(nn.LayerNorm, eps=1e-6),
            attn_drop_rate: float = 0,
            d_state: int = 16,
            expand: float = 2.,
            **kwargs,
    ):
        super().__init__()
        self.ln_1 = norm_layer(hidden_dim)
        self.self_attention = SS2D(d_model=hidden_dim, d_state=d_state,expand=expand,dropout=attn_drop_rate, **kwargs)
        self.drop_path = DropPath(drop_path)
        self.skip_scale= nn.Parameter(torch.ones(hidden_dim))


        self.conv_blk = CAB(hidden_dim)
        self.ln_2 = nn.LayerNorm(hidden_dim)
        self.skip_scale2 = nn.Parameter(torch.ones(hidden_dim))

 

    def forward(self, input, x_size):
        B, L, C = input.shape
        input = input.view(B, *x_size, C).contiguous()  # [B,H,W,C]
        x = self.ln_1(input)
        x = input*self.skip_scale + self.drop_path(self.self_attention(x))
        x = x*self.skip_scale2 + self.conv_blk(self.ln_2(x).permute(0, 3, 1, 2).contiguous()).permute(0, 2, 3, 1).contiguous()
        x = x.view(B, -1, C).contiguous()
        return x



## Resizing modules
class Downsample(nn.Module):
    def __init__(self, n_feat):
        super(Downsample, self).__init__()

        self.body = nn.Sequential(nn.Conv2d(n_feat, n_feat // 2, kernel_size=3, stride=1, padding=1, bias=False),
                                  nn.PixelUnshuffle(2))

    def forward(self, x, H, W):
        x = rearrange(x, "b (h w) c -> b c h w", h=H, w=W).contiguous()
        x = self.body(x)
        x = rearrange(x, "b c h w -> b (h w) c").contiguous()
        return x
class Downsample_input(nn.Module):
    def __init__(self, n_feat):
        super(Downsample_input, self).__init__()

        self.body = nn.Sequential(nn.Conv2d(n_feat, n_feat // 2, kernel_size=3, stride=1, padding=1, bias=False),
                                  nn.PixelUnshuffle(2))

    def forward(self, x, H, W):
        x = rearrange(x, "b h w c -> b c h w", h=H, w=W).contiguous()
        x = self.body(x)
        x = rearrange(x, "b c h w -> b (h w) c").contiguous()
        return x

class Upsample(nn.Module):
    def __init__(self, n_feat):
        super(Upsample, self).__init__()

        self.body = nn.Sequential(nn.Conv2d(n_feat, n_feat * 2, kernel_size=3, stride=1, padding=1, bias=False),
                                  nn.PixelShuffle(2))

    def forward(self, x, H, W):
        x = rearrange(x, "b (h w) c -> b c h w", h=H, w=W).contiguous()
        x = self.body(x)
        x = rearrange(x, "b c h w -> b (h w) c").contiguous()
        return x



class AWMFuse(nn.Module):
    def __init__(self,
                 model_clip,
                 inp_channels=32,
                 out_channels=32,
                 dim=48,
                 num_blocks=[8, 10, 10, 12],
                 mlp_ratio=2.,
                 num_refinement_blocks=8,
                 drop_path_rate=0.,
                 bias=False,
                 dual_pixel_task=False  ## True for dual-pixel defocus deblurring only. Also set inp_channels=6
                 ):

        super(AWMFuse, self).__init__()
        self.model_clip = model_clip


        self.GTPM = GTPM(dim)
        self.LTPM_0 = LTPM(dim)
        self.LTPM_1 = LTPM(dim * 2 ** 1)
        self.LTPM_2 = LTPM(dim * 2 ** 2)
        self.LTPM_3 = LTPM(dim * 2 ** 3)
        self.LTPM_4 = LTPM(dim * 2 ** 2)
        self.LTPM_5 = LTPM(dim * 2 ** 1)
        self.LTPM_6 = LTPM(dim* 2 ** 1)

        self.Decoder_1 = Decoder(96,48)
        self.Decoder_2 = Decoder(48, 24)
        self.Decoder_3 = Decoder(24, 3)




        self.mlp_ratio = mlp_ratio

        base_d_state = 4
        self.encoder_level1 = nn.ModuleList([
            VSSBlock(
                hidden_dim=dim,
                drop_path=drop_path_rate,
                norm_layer=nn.LayerNorm,
                attn_drop_rate=0,
                expand=self.mlp_ratio,
                d_state=base_d_state,
            )
            for i in range(num_blocks[0])])
        self.down1_2 = Downsample(dim)  ## From Level 1 to Level 2
        self.encoder_level2 = nn.ModuleList([
            VSSBlock(
                hidden_dim=int(dim * 2 ** 1),
                drop_path=drop_path_rate,
                norm_layer=nn.LayerNorm,
                attn_drop_rate=0,
                expand=self.mlp_ratio,
                d_state=int(base_d_state * 2 ** 1),
            )
            for i in range(num_blocks[1])])

        self.down2_3 = Downsample(int(dim * 2 ** 1))  ## From Level 2 to Level 3
        self.encoder_level3 = nn.ModuleList([
            VSSBlock(
                hidden_dim=int(dim * 2 ** 2),
                drop_path=drop_path_rate,
                norm_layer=nn.LayerNorm,
                attn_drop_rate=0,
                expand=self.mlp_ratio,
                d_state=int(base_d_state * 2 ** 2),
            )
            for i in range(num_blocks[2])])

        self.down3_4 = Downsample(int(dim * 2 ** 2))  ## From Level 3 to Level 4
        self.latent = nn.ModuleList([
            VSSBlock(
                hidden_dim=int(dim * 2 ** 3),
                drop_path=drop_path_rate,
                norm_layer=nn.LayerNorm,
                attn_drop_rate=0,
                expand=self.mlp_ratio,
                d_state=int(base_d_state * 2 ** 3),
            )
            for i in range(num_blocks[3])])

        self.up4_3 = Upsample(int(dim * 2 ** 3))  ## From Level 4 to Level 3
        self.reduce_chan_level3 = nn.Conv2d(int(dim * 2 ** 3), int(dim * 2 ** 2), kernel_size=1, bias=bias)
        self.decoder_level3 = nn.ModuleList([
            VSSBlock(
                hidden_dim=int(dim * 2 ** 2),
                drop_path=drop_path_rate,
                norm_layer=nn.LayerNorm,
                attn_drop_rate=0,
                expand=self.mlp_ratio,
                d_state=int(base_d_state * 2 ** 2),
            )
            for i in range(num_blocks[2])])

        self.up3_2 = Upsample(int(dim * 2 ** 2))  ## From Level 3 to Level 2
        self.reduce_chan_level2 = nn.Conv2d(int(dim * 2 ** 2), int(dim * 2 ** 1), kernel_size=1, bias=bias)
        self.decoder_level2 = nn.ModuleList([
            VSSBlock(
                hidden_dim=int(dim * 2 ** 1),
                drop_path=drop_path_rate,
                norm_layer=nn.LayerNorm,
                attn_drop_rate=0,
                expand=self.mlp_ratio,
                d_state=int(base_d_state * 2 ** 1),
            )
            for i in range(num_blocks[1])])

        self.up2_1 = Upsample(int(dim * 2 ** 1))  ## From Level 2 to Level 1  (NO 1x1 conv to reduce channels)

        self.decoder_level1 = nn.ModuleList([
            VSSBlock(
                hidden_dim=int(dim * 2 ** 1),
                drop_path=drop_path_rate,
                norm_layer=nn.LayerNorm,
                attn_drop_rate=0,
                expand=self.mlp_ratio,
                d_state=int(base_d_state * 2 ** 1),
            )
            for i in range(num_blocks[0])])

        self.refinement_1 = nn.ModuleList([
            VSSBlock(
                hidden_dim=int(dim * 2 ** 1),
                drop_path=drop_path_rate,
                norm_layer=nn.LayerNorm,
                attn_drop_rate=0,
                expand=self.mlp_ratio,
                d_state=int(base_d_state * 2 ** 1),
            )
            for i in range(num_refinement_blocks)])
        
        self.refinement_2 = nn.ModuleList([
            VSSBlock(
                hidden_dim=int(dim),
                drop_path=drop_path_rate,
                norm_layer=nn.LayerNorm,
                attn_drop_rate=0,
                expand=self.mlp_ratio,
                d_state=int(base_d_state * 2 ** 1),
            )
            for i in range(num_refinement_blocks)])
        
        self.refinement_2 = nn.ModuleList([
            VSSBlock(
                hidden_dim=int(dim/2),
                drop_path=drop_path_rate,
                norm_layer=nn.LayerNorm,
                attn_drop_rate=0,
                expand=self.mlp_ratio,
                d_state=int(base_d_state * 2 ** 1),
            )
            for i in range(num_refinement_blocks)])

        ### For Dual-Pixel Defocus Deblurring Task ####
        self.dual_pixel_task = dual_pixel_task
        if self.dual_pixel_task:
            self.skip_conv = nn.Conv2d(dim, int(dim * 2 ** 1), kernel_size=1, bias=bias)
        ###########################

        self.output = nn.Conv2d(int(dim * 2 ** 1), out_channels, kernel_size=3, stride=1, padding=1, bias=bias)
        self.input_1 = nn.Conv2d(3, 48, kernel_size=3, stride=1, padding=1, bias=bias)
        self.input_2 = nn.Conv2d(1, 48, kernel_size=3, stride=1, padding=1, bias=bias)


    def forward(self, inp_img_ir, inp_img, clip, blip_text_ir, blip_text_vi):
        clip_text = self.get_text_feature(clip).to(inp_img_ir.dtype)
        inp = (inp_img_ir + inp_img) // 2
        ClipImage = self.get_image_feature(inp)
        ClipImage = ClipImage.float()
        _, _, H, W = inp_img.shape

        inp_img_ir = inp_img_ir[:, :1, :, :]

        blip_text = blip_text_ir + blip_text_vi
        blip_text = blip_text.squeeze(1)


        fusion_feature = self.GTPM(inp_img, inp_img_ir, ClipImage, clip_text)
        fusion_feature = (self.input_1(inp_img) + self.input_2(inp_img_ir)) * fusion_feature
        fusion_feature = rearrange(fusion_feature, "b c h w -> b (h w) c").contiguous()




        inp_enc_level1 = fusion_feature

    
        out_enc_level1 = inp_enc_level1
        for layer in self.encoder_level1:
            out_enc_level1 = layer(out_enc_level1, [H, W])

        local_TPM_0 = self.LTPM_0(ClipImage, out_enc_level1, blip_text, [H, W]) 

        out_enc_level1 = out_enc_level1 * local_TPM_0 + out_enc_level1
        inp_enc_level2 = self.down1_2(out_enc_level1, H, W)  # b, hw//4, 2c


        out_enc_level2 = inp_enc_level2
        for layer in self.encoder_level2:
            out_enc_level2 = layer(out_enc_level2, [H // 2, W // 2])

        local_TPM_1 = self.LTPM_1(ClipImage, out_enc_level2, blip_text, [H // 2, W // 2])
        out_enc_level2 = out_enc_level2 * local_TPM_1 + out_enc_level2
        inp_enc_level3 = self.down2_3(out_enc_level2, H // 2, W // 2)  # b, hw//16, 4c
        out_enc_level3 = inp_enc_level3 

        for layer in self.encoder_level3:
            out_enc_level3 = layer(out_enc_level3, [H // 4, W // 4])

        local_TPM_2 = self.LTPM_2(ClipImage, out_enc_level3, blip_text, [H // 4, W // 4])
        out_enc_level3 = out_enc_level3 * local_TPM_2 + out_enc_level3
        inp_enc_level4 = self.down3_4(out_enc_level3, H // 4, W // 4)  # b, hw//64, 8c
        latent = inp_enc_level4 

        for layer in self.latent:
            latent = layer(latent, [H // 8, W // 8])
        local_TPM_3 = self.LTPM_3(ClipImage, latent, blip_text, [H // 8, W // 8])
        latent = latent * local_TPM_3 + latent
        inp_dec_level3 = self.up4_3(latent, H // 8, W // 8)  # b, hw//16, 4c


        inp_dec_level3 = torch.cat([inp_dec_level3, out_enc_level3], 2)
        inp_dec_level3 = rearrange(inp_dec_level3, "b (h w) c -> b c h w", h=H // 4, w=W // 4).contiguous()
        inp_dec_level3 = self.reduce_chan_level3(inp_dec_level3)
        inp_dec_level3 = rearrange(inp_dec_level3, "b c h w -> b (h w) c").contiguous()  # b, hw//16, 4c
        out_dec_level3 = inp_dec_level3
        for layer in self.decoder_level3:
            out_dec_level3 = layer(out_dec_level3, [H // 4, W // 4])

        local_TPM_4 = self.LTPM_4(ClipImage, out_dec_level3, blip_text, [H // 4, W // 4])
        out_dec_level3 = out_dec_level3 * local_TPM_4 + out_dec_level3
        inp_dec_level2 = self.up3_2(out_dec_level3, H // 4, W // 4)  # b, hw//4, 2c        
        inp_dec_level2 = torch.cat([inp_dec_level2, out_enc_level2], 2)
        inp_dec_level2 = rearrange(inp_dec_level2, "b (h w) c -> b c h w", h=H // 2, w=W // 2).contiguous()
        inp_dec_level2 = self.reduce_chan_level2(inp_dec_level2)
        inp_dec_level2 = rearrange(inp_dec_level2, "b c h w -> b (h w) c").contiguous()  # b, hw//4, 2c
        out_dec_level2 = inp_dec_level2
        for layer in self.decoder_level2:
            out_dec_level2 = layer(out_dec_level2, [H // 2, W // 2])

        local_TPM_5 = self.LTPM_5(ClipImage, out_dec_level2, blip_text, [H // 2, W // 2])
        out_dec_level2 = out_dec_level2 * local_TPM_5 + out_dec_level2
        inp_dec_level1 = self.up2_1(out_dec_level2, H // 2, W // 2)  # b, hw, c

        inp_dec_level1 = torch.cat([inp_dec_level1, out_enc_level1], 2)
        out_dec_level1 = inp_dec_level1
        

        for layer in self.decoder_level1:
            out_dec_level1 = layer(out_dec_level1, [H, W])

        local_TPM_6 = self.LTPM_6(ClipImage, out_dec_level1, blip_text, [H, W]) 
        out_dec_level1 = out_dec_level1 * local_TPM_6 + out_dec_level1
        




        out_dec_level1 = rearrange(out_dec_level1, "b (h w) c -> b c h w", h=H, w=W).contiguous()


        ### For Dual-Pixel Defocus Deblurring Task ####
        if self.dual_pixel_task:
            out_dec_level1 = out_dec_level1 + self.skip_conv(inp_enc_level1)
            out_dec_level1 = self.output(out_dec_level1)
        ##########################################################
        out_dec_level1 = self.Decoder_1(out_dec_level1)
        out_dec_level1 = self.Decoder_2(out_dec_level1)
        out_dec_level1 = self.Decoder_3(out_dec_level1)


        return out_dec_level1


    @torch.no_grad()
    def get_text_feature(self, text):
        text_feature = self.model_clip.encode_text(text)
        return text_feature

    @torch.no_grad()
    def get_image_feature(self, image):
        if image.shape[1] == 1:
            image = image.repeat(1, 3, 1, 1)
        resize = torchvision.transforms.Resize((224, 224))
        image = resize(image)
        image_feature = self.model_clip.encode_image(image)
        return image_feature

