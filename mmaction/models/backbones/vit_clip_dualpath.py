from collections import OrderedDict
from typing import Tuple, Union
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
import clip

from mmengine.logging import MMLogger
# from mmaction.utils import get_root_logger
from einops import rearrange

from mmaction.registry import MODELS
from torchvision import transforms
import math

logger = MMLogger.get_current_instance()

class TemporalPositionalEmbedding(nn.Module):
    def __init__(self, channels):
        super(TemporalPositionalEmbedding, self).__init__()
        self.channels = channels
        self.inv_freq = 1.0 / (10000 ** (torch.arange(0, channels, 2).float() / channels))

    def forward(self, tensor):
        if len(tensor.shape) == 3:
            
            batch_size, N, channels = tensor.shape
            resize_shape = int(math.sqrt(N-1))
            num_frame_per_width = 4
            width = int(16 / num_frame_per_width)    # resized frame size
            temp = torch.zeros(batch_size, 16, width, width).to(tensor.device)
            for i in range(num_frame_per_width ** 2):
                temp[:, i, :, :] = i + 1
            temp = temp.reshape(batch_size, num_frame_per_width, num_frame_per_width, width, width)
            # batch_size, num_frame_per_width, , width, num_frame_per_width, width
            temp = temp.permute(0, 1, 3, 2, 4).reshape(batch_size, 16, 16)
            resize = transforms.Resize((resize_shape, resize_shape))
            temp = resize(temp)
            emb = temp.view(batch_size, -1)[0]
            emb = torch.cat([torch.tensor([0.0]).view(1).to(tensor.device), emb])
            emb = torch.einsum("i,j->ij", emb, self.inv_freq.to(tensor.device))   # [N, D]
            emb = torch.stack((emb.sin(), emb.cos()), dim = -1)
            emb = torch.flatten(emb, -2, -1)
            return emb.repeat(batch_size, 1, 1)
        else:
            # int(GB / self.num_frames), self.num_frames, N, D
            batch_size, Tt, N, channels = tensor.shape
            resize_shape = int(math.sqrt(N-1))
            # emb = torch.zeros(batch_size, N, channels)
            num_frame_per_width = 4
            width = int(16 / num_frame_per_width)    # resized frame size
            temp = torch.zeros(batch_size, 16 * Tt, width, width).to(tensor.device)
            for i in range((num_frame_per_width ** 2) * Tt):
                temp[:, i, :, :] = i + 1
            temp = temp.reshape(batch_size, Tt, num_frame_per_width, num_frame_per_width, width, width)
            temp = temp.permute(0, 1, 2, 4, 3, 5).reshape(batch_size, Tt, 16, 16)
            resize = transforms.Resize((resize_shape, resize_shape))
            temp = resize(temp) # [B, Tt, root(N), root(N)]
            emb = temp.view(batch_size, Tt, -1)[0]  # [B, Tt, N]
            emb = torch.cat([torch.tensor([[0.0]]*Tt).to(tensor.device), emb], dim = 1) #[Tt, N]
            emb = emb.view(-1)  # [TtxN]
            emb = torch.einsum("i,j->ij", emb, self.inv_freq.to(tensor.device))  #[TtxN, D]
            emb = torch.stack((emb.sin(), emb.cos()), dim = -1) # [TtxN,D/2,2]
            emb = torch.flatten(emb, -2, -1).reshape(Tt, N, -1) 
            return emb.repeat(batch_size, 1, 1, 1)

class Adapter(nn.Module):
    def __init__(self,
                d_model: int = 768,
                bottleneck: int = 128,
                dropout=0.0,
                init_option="lora",
                adapter_scalar="1.0",
                adapter_layernorm_option="out"):
        super().__init__()
        self.n_embd = d_model
        self.down_size = bottleneck

        #_before
        self.adapter_layernorm_option = adapter_layernorm_option

        self.adapter_layer_norm_before = None
        if adapter_layernorm_option in ["in", "out"]:
            self.adapter_layer_norm_before = nn.LayerNorm(self.n_embd)

        if adapter_scalar == "learnable_scalar":
            self.scale = nn.Parameter(torch.ones(1))
        else:
            self.scale = float(adapter_scalar)

        self.down_proj = nn.Linear(self.n_embd, self.down_size)
        self.non_linear_func = nn.GELU()
        self.up_proj = nn.Linear(self.down_size, self.n_embd)

        self.dropout = dropout
        if init_option == "bert":
            raise NotImplementedError
        elif init_option == "lora":
            with torch.no_grad():
                nn.init.kaiming_uniform_(self.down_proj.weight, a=math.sqrt(5))
                nn.init.zeros_(self.up_proj.weight)
                nn.init.zeros_(self.down_proj.bias)
                nn.init.zeros_(self.up_proj.bias)

    def forward(self, x, add_residual=False, residual=None):
        residual = x if residual is None else residual
        if self.adapter_layernorm_option == 'in':
            x = self.adapter_layer_norm_before(x)

        down = self.down_proj(x)
        down = self.non_linear_func(down)
        down = nn.functional.dropout(down, p=self.dropout, training=self.training)
        up = self.up_proj(down)

        up = up * self.scale

        if self.adapter_layernorm_option == 'out':
            up = self.adapter_layer_norm_before(up)

        return up + residual if add_residual else up

class LayerNorm(nn.LayerNorm):
    """Subclass torch's LayerNorm to handle fp16."""

    def forward(self, x: torch.Tensor):
        orig_type = x.dtype
        ret = super().forward(x.type(torch.float32))
        return ret.type(orig_type)


class QuickGELU(nn.Module):
    def forward(self, x: torch.Tensor):
        return x * torch.sigmoid(1.702 * x)


class ResidualAttentionBlock(nn.Module):
    def __init__(self, d_model: int, n_head: int, attn_mask: torch.Tensor = None, scale=0.25, num_frames=8, drop_path=0., adapter = None):
        super().__init__()
        
        self.attn = nn.MultiheadAttention(d_model, n_head)
        self.ln_1 = LayerNorm(d_model)
        self.mlp = nn.Sequential(OrderedDict([
            ("c_fc", nn.Linear(d_model, d_model * 4)),
            ("gelu", QuickGELU()),
            ("c_proj", nn.Linear(d_model * 4, d_model))
        ]))
        self.ln_2 = LayerNorm(d_model)
        self.attn_mask = attn_mask
        self.n_head = n_head

        self.adapter = adapter
        if self.adapter == 'w-adapter':
            # self.t_adapter_attn_b = Adapter(d_model=d_model, bottleneck=int(scale * d_model), dropout=0.1, adapter_layernorm_option=None)
            self.s_adapter_attn = Adapter(d_model=d_model, bottleneck=int(scale * d_model), dropout=0.1, adapter_layernorm_option=None)
            self.t_adapter_attn = Adapter(d_model=d_model, bottleneck=int(scale * d_model), dropout=0.1, adapter_layernorm_option=None)
            self.s_adapter_mlp = Adapter(d_model=d_model, bottleneck=int(scale * d_model), dropout=0.1, adapter_layernorm_option=None)
            self.t_adapter_mlp = Adapter(d_model=d_model, bottleneck=int(scale * d_model), dropout=0.1, adapter_layernorm_option=None)
        
        self.num_frames = num_frames
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def attention(self, x: torch.Tensor):
        self.attn_mask = self.attn_mask.to(dtype=x.dtype, device=x.device) if self.attn_mask is not None else None
        return self.attn(x, x, x, need_weights=False, attn_mask=self.attn_mask)[0]

    def forward(self, x: torch.Tensor):
        if self.adapter == None:
            x = x + self.attention(self.ln_1(x))
            x = x + self.mlp(self.ln_2(x))
            return x
        elif self.adapter == 'w-adapter':
            T = self.num_frames
            Ts = 8
            Tt = int(T - Ts)
            B, N, D = int(x.shape[1] / (T)), x.shape[0], x.shape[2]
            x_t_residual = x.reshape(N, B, T, D)[:, :, 0:Tt, :].reshape(N, -1, D)
            x_s_residual = x.reshape(N, B, T, D)[:, :, Tt:, :].reshape(N, -1, D)
            x = self.ln_1(x)
            x = x.reshape(N, B, T, D)
            x_s = x[:, :, Tt:, :].reshape(N, -1, D)  # [N+1, B*Ts, D]
            x_t = x[:, :, 0:Tt, :].reshape(N, -1, D) # [N+1, B*Tt, D]
            # x_t = self.t_adapter_attn_b(x_t, add_residual=False)
            x_t = self.attention(x_t)
            x_t = self.t_adapter_attn(x_t)
            x_t = x_t + x_t_residual
            x_t_residual2 = x_t
            x_t = self.ln_2(x_t)
            x_t = self.mlp(x_t)
            x_t = self.t_adapter_mlp(x_t) + x_t_residual2

            x_s_adapt = self.s_adapter_attn(x_s)
            x_s = self.attention(x_s) + x_s_adapt + x_s_residual
            x_s_residual2 = x_s
            x_s = self.ln_2(x_s)
            x_s_mlp = self.s_adapter_mlp(x_s)
            x_s = self.mlp(x_s) + x_s_mlp + x_s_residual2
            x = torch.cat([x_t.reshape(N, B, -1, D), x_s.reshape(N, B, -1, D)], dim=2).reshape(N, -1, D)
            return x


class Transformer(nn.Module):
    def __init__(self, num_frames, width: int, layers: int, heads: int, attn_mask: torch.Tensor = None,  adapter_scale=1., drop_path=0.1, adapter: str = None,):
        super().__init__()
        self.width = width
        self.layers = layers
        self.adapter = adapter
        dpr = [x.item() for x in torch.linspace(0, drop_path, self.layers)]
        self.resblocks = nn.Sequential(*[ResidualAttentionBlock(width, heads, attn_mask, adapter_scale,  num_frames, dpr[i], adapter = adapter) for i in range(layers)])

    def forward(self, x: torch.Tensor):
        return self.resblocks(x)


@MODELS.register_module()
class ViT_CLIP_DUALPATH(nn.Module):
    ## ViT definition in CLIP image encoder
    def __init__(self, input_resolution: int, num_frames: int, patch_size: int, width: int, layers: int, heads: int, drop_path_rate, adapter_scale=0.5, pretrained=None,
                adapter: str = 'w-adapter', output_dim: int = 512):
        super().__init__()
        self.input_resolution = input_resolution
        self.pretrained = pretrained
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=width, kernel_size=patch_size, stride=patch_size, bias=False)

        scale = width ** -0.5
        self.layers = layers
        self.class_embedding = nn.Parameter(scale * torch.randn(width))
        self.positional_embedding = nn.Parameter(scale * torch.randn((input_resolution // patch_size) ** 2 + 1, width))
        self.ln_pre = LayerNorm(width)

        self.num_frames = num_frames // 16 + 8

        # self.temporal_embedding = nn.Parameter(torch.zeros(1, num_frames, width))

        self.transformer = Transformer(self.num_frames, width, layers, heads, adapter_scale=adapter_scale, drop_path=drop_path_rate,adapter=adapter)

        self.ln_post = LayerNorm(width)



        adapter_list = [None, 'w-adapter', 'protuning', 'vpt', 'st-adapter', 'adaptformer', 'aim']
        if adapter not in adapter_list:
            raise ValueError("Warning: Check adapt method!")

        self.adapter = adapter
        self.temporal_positional_embedding = TemporalPositionalEmbedding(width)
        self.spatial_positional_embedding = nn.Parameter(scale * torch.randn((input_resolution // patch_size) ** 2 + 1, width))

        self.proj = nn.Parameter(scale * torch.randn(width, output_dim))
        if adapter == 'w-adapter':
            self.head = nn.Sequential(OrderedDict([
          ('linear1', nn.Linear(output_dim * 2, output_dim)),
        ('gelu', nn.GELU()),
        ]))
        
    def init_weights(self, pretrained=None):
        def _init_weights(m):
            if isinstance(m, nn.Linear):
                trunc_normal_(m.weight, std=.02)
                if isinstance(m, nn.Linear) and m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.bias, 0)
                nn.init.constant_(m.weight, 1.0)

        if pretrained:
            self.pretrained = pretrained
        if isinstance(self.pretrained, str):
            self.apply(_init_weights)
            # logger = get_root_logger()
            logger.info(f'load model from: {self.pretrained}')
            ## Load OpenAI CLIP pretrained weights
            if self.layers == 12:
                clip_model, preprocess = clip.load("ViT-B/16", device="cpu")
            else:
                clip_model, preprocess = clip.load("ViT-L/14", device="cpu")
            pretrain_dict = clip_model.visual.state_dict()
            del clip_model
            del pretrain_dict['proj']
            msg = self.load_state_dict(pretrain_dict, strict=False)
            logger.info('Missing keys: {}'.format(msg.missing_keys))
            logger.info('Unexpected keys: {}'.format(msg.unexpected_keys))
            logger.info(f"=> loaded successfully '{self.pretrained}'")
            
            ## freeze some parameters
            for name, param in self.named_parameters():
                param.requires_grad = False
                if name in msg.missing_keys:
                    param.requires_grad = True
                if 'class_embedding' in name:
                    param.requires_grad = True
                if 'spatial_positional_embedding' in name:
                    param.requires_grad = True
                if 'positional_embedding' in name:
                    param.requires_grad = True

            for name, param in self.named_parameters():
                logger.info(f'{name}: {param.requires_grad}')
            num_param = sum(p.numel() for p in self.parameters() if p.requires_grad)
            num_total_param = sum(p.numel() for p in self.parameters())
            logger.info(
                f'Number of total parameters: {num_total_param}, tunable parameters: {num_param}'
            )
            
            torch.cuda.empty_cache()
        elif self.pretrained is None:
            logger.info('No pretrained weights are loaded!!!!!')
            self.apply(_init_weights)
        else:
            raise TypeError('pretrained must be a str or None')

        # ## initialize S_Adapter
        # for n, m in self.transformer.named_modules():
        #     if 's_adapter_attn' in n:
        #         for n2, m2 in m.named_modules():
        #             if 'D_fc2' in n2:
        #                 if isinstance(m2, nn.Linear):
        #                     nn.init.constant_(m2.weight, 0)
        #                     nn.init.constant_(m2.bias, 0)

        # ## initialize T_Adapter
        # for n, m in self.transformer.named_modules():
        #     if 't_adapter_attn' in n:
        #         for n2, m2 in m.named_modules():
        #             if 'D_fc2' in n2:
        #                 if isinstance(m2, nn.Linear):
        #                     nn.init.constant_(m2.weight, 0)
        #                     nn.init.constant_(m2.bias, 0)

        # ## initialize MLP_Adapter
        # for n, m in self.transformer.named_modules():
        #     if 'MLP_Adapter' in n:
        #         for n2, m2 in m.named_modules():
        #             if 'D_fc2' in n2:
        #                 if isinstance(m2, nn.Linear):
        #                     nn.init.constant_(m2.weight, 0)
        #                     nn.init.constant_(m2.bias, 0)
        

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'absolute_pos_embed', 'temporal_embedding'}

    @torch.jit.ignore
    def no_weight_decay_keywords(self):
        return {'relative_position_bias_table', 'temporal_position_bias_table'}

    def forward(self, x: torch.Tensor):
        B, C, T, H, W = x.shape
        
        x= self.grid_like_frameset(x,T)
        
        # x = rearrange(x, 'b c t h w -> (b t) c h w')
        x = self.conv1(x)  
        x = x.reshape(x.shape[0], x.shape[1], -1) # bt d n
        x = x.permute(0, 2, 1)  # bt n d
        x = torch.cat([self.class_embedding.to(x.dtype) + torch.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device), x], dim=1)
        if self.adapter == 'w-adapter':
            GB, N, D = x.shape
            Ts = 8
            Tt = int(self.num_frames - 8)
            x = x.reshape(int(GB / self.num_frames), self.num_frames, N, D)
            # Temporal path : temporal_positional_embedding & spatial_positional_embedding
            x[:, 0:Tt, :, :] = x[:, 0:Tt, :, :] + self.temporal_positional_embedding(x[:, 0:Tt, :, :]) + self.spatial_positional_embedding.expand(int(GB / self.num_frames), Tt, -1, -1)
            # Spatial Path: positional_embedding
            x[:, Tt:, :, :] = (x[:, Tt:, :, :].reshape(-1, N, D) + self.positional_embedding.to(x.dtype)).reshape(int(GB / self.num_frames), 8, N, D)

            x = x.reshape(-1, N, D)
        else:
            x = x + self.positional_embedding.to(x.dtype)

        # n = x.shape[1]
        # x = rearrange(x, '(b t) n d -> (b n) t d', t=self.num_frames)
        # x = x + self.temporal_embedding
        # x = rearrange(x, '(b n) t d -> (b t) n d', n=n)
        
        x = self.ln_pre(x)

        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        
        # x_temp = x[:, 1:, :]    # [B*T, N, D]
        x = self.ln_post(x[:, 0, :])    # [B*T, 1, D]
        if self.proj is not None:
            x = x @ self.proj

        if self.adapter == 'w-adapter':
            x = x.reshape(-1, Ts + Tt, x.shape[-1])

            # x_temp = x_temp.reshape(-1, Ts + Tt, x_temp.shape[1], x_temp.shape[-1]) # [B, T, N, D]
            # x_temp = x_temp[:, 0:Tt, :, :].reshape(-1, x_temp.shape[2], x_temp.shape[-1])  # [B * Tt, N, D]
            # cls_t = x[:, 0:Tt, :].reshape(-1, 1, cls_t.shape[-1])   # [B * Tt, D]
            # attn_t = torch.bmm(x_temp, cls_t.permute(0, 2, 1)).reshape(-1, Tt, x_temp.shape[1])  # [B * Tt, N]
            # attn_t = attn_t.reshape(-1, Tt, int(math.sqrt(attn_t.shape[1])), int(math.sqrt(attn_t.shape[1])))
            # [b 2* output_dim]
            x = torch.cat([x[:, 0:Tt, :].mean(1), x[:, Tt:, :].mean(1)], dim = -1)
        else:
            x = x.mean(1)
        x=self.head(x)
        # x = self.ln_post(x)
        # x = x[:, 0]
        x = rearrange(x, '(b t) d -> b d t',b=B,t=1)
        
        x = x.unsqueeze(-1).unsqueeze(-1)  # BDTHW for I3D head

        return x
    
    def grid_like_frameset(self, x, num_frames):
        
        
        # samples : NxCxTxHxW
        # N
        batch_size = x.shape[0]
        # C
        in_channels = x.shape[1]
        # H & W
        frame_size = x.shape[3]

        # spatial sampling interval
        spatial_stride = int(num_frames / 8)
        # grid size
        block_width = 4

        # num of grid-like framesets
        num_temporal_frame = int(num_frames / (block_width ** 2))

        resize = transforms.Resize((frame_size, frame_size))
        samples_t = x
        # NxCx(T/spatial_stride)xHxW
        samples_s = x[:, :, 0::spatial_stride, :, :]

        # grid-like frameset transform
        # num_frames ->  num_temporal_frame x (block_width ** 2)                                                                                                                          
        samples_t = samples_t.reshape(batch_size, in_channels, num_temporal_frame, int(block_width ** 2), frame_size, frame_size)
        # int(block_width ** 2) -> block_width x block_width
        samples_t = samples_t.reshape(batch_size, in_channels, num_temporal_frame, block_width, block_width, frame_size, frame_size)
        # batch_size, in_channels, num_temporal_frame, block_width, frame_size, block_width, frame_size
        samples_t = samples_t.permute(0, 1, 2, 3, 5, 4, 6)
        # flatten 2D
        samples_t = samples_t.reshape(batch_size * in_channels * num_temporal_frame, block_width * frame_size, block_width * frame_size)
        # resize 
        samples_t = resize(samples_t).reshape(batch_size, in_channels, num_temporal_frame, frame_size, frame_size)

        x = torch.cat([samples_t, samples_s], dim=2)

        # samples = samples.to(device, non_blocking=True)

        if x.shape[2] != 1:
            # batch_size, in_channels, num_temporal_frame, frame_size, frame_size
            x = x.permute(0, 2, 1, 3, 4)
            x = x.reshape(
                batch_size * (8 + num_temporal_frame),
                in_channels,
                frame_size,
                frame_size,
            )
        else:
            x = x.squeeze()

        return x

        
    
    def train(self, mode: bool = True) -> None:
        """Convert the model into training mode while keep layers frozen."""
        super(ViT_CLIP_DUALPATH, self).train(mode)


if __name__ == '__main__':

    from mmengine.analysis import get_model_complexity_info

    input_shape = (3,32,224,224)
    model = ViT_CLIP_DUALPATH(
                        adapter_scale=0.25,
                        pretrained='openaiclip',
                        input_resolution=224,
                        patch_size=16,
                        num_frames=32,
                        width=768,
                        layers=12,
                        heads=12,
                        drop_path_rate=0.1,
                        adapter='w-adapter',
                        output_dim=512)

    analysis_results = get_model_complexity_info(model, input_shape)
    
    print(analysis_results['out_table'])

    # print(analysis_results['out_arch'])

    print(f"Model Flops:{analysis_results['flops_str']}")

    print(f"Model Parameters:{analysis_results['params_str']}")

    
    