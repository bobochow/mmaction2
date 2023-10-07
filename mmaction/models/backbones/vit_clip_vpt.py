from collections import OrderedDict
from typing import Dict, List, Optional, Sequence, Tuple, Union
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

# from ..builder import BACKBONES

logger = MMLogger.get_current_instance()

def init_generator(device: torch.device, fallback: torch.Generator=None):
    """
    Forks the current default random generator given device.
    """
    if device.type == "cpu":
        return torch.Generator(device="cpu").set_state(torch.get_rng_state())
    elif device.type == "cuda":
        return torch.Generator(device=device).set_state(torch.cuda.get_rng_state())
    else:
        if fallback is None:
            return init_generator(torch.device("cpu"))
        else:
            return fallback


class Adapter(nn.Module):
    def __init__(self, D_features, mlp_ratio=0.25, act_layer=nn.GELU, skip_connect=True):
        super().__init__()
        self.skip_connect = skip_connect
        D_hidden_features = int(D_features * mlp_ratio)
        self.act = act_layer()
        self.D_fc1 = nn.Linear(D_features, D_hidden_features)
        self.D_fc2 = nn.Linear(D_hidden_features, D_features)
        
    def forward(self, x):
        # x is (BT, HW+1, D)
        xs = self.D_fc1(x)
        xs = self.act(xs)
        xs = self.D_fc2(xs)
        if self.skip_connect:
            x = x + xs
        else:
            x = xs
        return x


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
    def __init__(self, d_model: int, n_head: int, attn_mask: torch.Tensor = None,
                scale=1., num_adapter=1, num_frames=8, drop_path=0.,

                shift: bool = False, 
                shift_type: str = 'psm',
                
                ):
        super().__init__()
        self.num_adapter = num_adapter
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


        self.d_model=d_model
        self.shift = shift
        self.shift_type = shift_type

        self.MLP_Adapter = Adapter(d_model, skip_connect=False)
        self.S_Adapter = Adapter(d_model)
        # self.S_Adapter_cross = Adapter(d_model)

        self.scale = scale
        self.T_Adapter = Adapter(d_model, skip_connect=False)

        if num_adapter == 2:
            self.S_Adapter_shift = Adapter(d_model)

        self.num_frames = num_frames
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        if self.shift:
            # self.shift_op=PatchShift( inv=False)

            if self.shift_type == 'psm':
                self.shift_op=PatchShift( inv=False)
                # self.shift_op = PatchShift(False)
                # self.shift_op_back = PatchShift(True)
            elif self.shift_type == 'r-psm':
                self.shift_op=Rand2dPatchShift( inv=False, sx=2, sy=2)
            # elif self.shift_type == 'tsm':
            #     self.shift_op = TemporalShift(8)


    def attention(
        self, x: torch.Tensor 
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        
        # input: HW+2, BT, D
        
        q = (x @ self.attn.in_proj_weight[:self.d_model].T
             ) + self.attn.in_proj_bias[:self.d_model]

        k = (x @ self.attn.in_proj_weight[self.d_model:-self.d_model].T
             ) + self.attn.in_proj_bias[self.d_model:-self.d_model]
        v = (x @ self.attn.in_proj_weight[-self.d_model:].T
             ) + self.attn.in_proj_bias[-self.d_model:]
        Tx, Ty, N = q.size(0), k.size(0), q.size(1)
        q = q.view(Tx, N, self.attn.num_heads,
                   self.attn.head_dim).permute(1, 2, 0, 3) # N num_heads Tx D_head
        k = k.view(Ty, N, self.attn.num_heads,
                   self.attn.head_dim).permute(1, 2, 0, 3) # N num_heads Ty D_head
        v = v.view(Ty, N, self.attn.num_heads,
                   self.attn.head_dim).permute(1, 2, 0, 3)
        
        aff = (q @ k.transpose(-2, -1) / (self.attn.head_dim**0.5)) # (N, num_heads, Tx, Ty)
        
        aff = aff.softmax(dim=-1)
        
        out = aff @ v  # N num_heads Tx D_head
        out = out.permute(2, 0, 1, 3).flatten(2)
        out = self.attn.out_proj(out) # N Tx D
        
        return out 
    
    def shift_attention(
        self, x: torch.Tensor ,y: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        
        # input: HW+2, BT, D
        
        # L, BT, C = x.shape
        # T = self.num_frames
        # head_dims = C // self.n_head
        # B = BT // self.num_frames
        # H = W = int((L-2)**0.5)
        
        q = (x @ self.attn.in_proj_weight[:self.d_model].T
            ) + self.attn.in_proj_bias[:self.d_model]

        k = (y @ self.attn.in_proj_weight[self.d_model:-self.d_model].T
            ) + self.attn.in_proj_bias[self.d_model:-self.d_model]
        v = (y @ self.attn.in_proj_weight[-self.d_model:].T
            ) + self.attn.in_proj_bias[-self.d_model:]
        Tx, Ty, N = q.size(0), k.size(0), q.size(1)
        q = q.view(Tx, N, self.attn.num_heads,
                    self.attn.head_dim).permute(1, 2, 0, 3) # N num_heads Tx D_head
        k = k.view(Ty, N, self.attn.num_heads,
                    self.attn.head_dim).permute(1, 2, 0, 3) # N num_heads Ty D_head
        v = v.view(Ty, N, self.attn.num_heads,
                    self.attn.head_dim).permute(1, 2, 0, 3)
        
        aff = (q @ k.transpose(-2, -1) / (self.attn.head_dim**0.5)) # (N, num_heads, Tx, Ty)
        
        aff = aff.softmax(dim=-1)
        
        out = aff @ v  # N num_heads Tx D_head
        out = out.permute(2, 0, 1, 3).flatten(2)
        out = self.attn.out_proj(out) # N Tx D
        
        return out 

    def forward(self, x: torch.Tensor):
        ## x shape [HW+1, BT, D]
        n, bt, d = x.shape
        # temporal adaptation
        
        class_token=x[:1,:,:] # 1, BT, D
        
        xt = rearrange(class_token, 'n (b t) d -> t (b n) d', t = self.num_frames + self.num_frames//8 )
        
        xt = self.T_Adapter(self.attention(self.ln_1(xt)))
        xt = rearrange(xt, 't (b n) d -> n (b t) d', n=1)
        
        x= torch.cat([x[:1,:,:], xt, x[1:,:,:]], dim=0)
        
        ## prompt tuning
        if self.shift:
            
            xln=self.ln_1(x)
            
            tmp_x=xln[2:, :, :].clone()
            # tmp_x=xln[1:, :, :].clone()
            
            # x = x + self.S_Adapter(self.attention(tmp_x,xln)[:L+2,:,:])
            
            x = x + self.S_Adapter(self.shift_attention(xln,tmp_x))
            
            # joint adaptation
            x= torch.cat([x[:1,:,:], x[2:,:,:]], dim=0) # [HW+2, BT, D]
            xn = self.ln_2(x)
            x = x + self.mlp(xn) + self.drop_path(self.scale * self.MLP_Adapter(xn))
            
        else:
            
            x = x + self.S_Adapter(self.attention(self.ln_1(x)))
            # joint adaptation
            x= torch.cat([x[:1,:,:], x[2:,:,:]], dim=0) # [HW+2, BT, D]
            xn = self.ln_2(x)
            x = x + self.mlp(xn) + self.drop_path(self.scale * self.MLP_Adapter(xn))
        
        return x


class Transformer(nn.Module):
    def __init__(self, num_frames, width: int, layers: int, heads: int, attn_mask: torch.Tensor = None, 
                num_adapter=1, scale=1., drop_path=0.1,
                ):
        super().__init__()
        self.width = width
        self.layers = layers

        dpr = [x.item() for x in torch.linspace(0, drop_path, self.layers)]
        
        self.resblocks = nn.Sequential(
            *[
                ResidualAttentionBlock(
                    width,
                    heads,
                    attn_mask,
                    scale,
                    num_adapter,
                    num_frames,
                    dpr[i],
                    shift= False,
                    shift_type='psm',
                )
                for i in range(layers)
            ]
        )
        
    def forward(self, x: torch.Tensor):
        return self.resblocks(x)


@MODELS.register_module()
class ViT_CLIP_VPT(nn.Module):
    ## ViT definition in CLIP image encoder
    def __init__(self, input_resolution: int, num_frames: int, patch_size: int, width: int, layers: int, heads: int, drop_path_rate,
                num_adapter=2,
                adapter_scale=0.5, 
                pretrained=None,
                abs_tpe=True,
                use_abs_pos_embed=True,
                
                ):
        super().__init__()
        self.input_resolution = input_resolution
        self.pretrained = pretrained
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=width, kernel_size=patch_size, stride=patch_size, bias=False)

        scale = width ** -0.5
        self.layers = layers
        self.class_embedding = nn.Parameter(scale * torch.randn(width))
        
        self.use_abs_pos_embed = use_abs_pos_embed
        if self.use_abs_pos_embed:
            self.positional_embedding = nn.Parameter(scale * torch.randn((input_resolution // patch_size) ** 2 + 1, width))
        self.ln_pre = LayerNorm(width)

        self.num_frames = num_frames
        patch_nums=input_resolution // patch_size
        self.input_size=(num_frames,patch_nums,patch_nums)
        
        self.abs_tpe=abs_tpe
        if self.abs_tpe:
            self.temporal_embedding = nn.Parameter(torch.zeros(1, num_frames, width))

        self.transformer = Transformer(num_frames, width, layers, heads, num_adapter=num_adapter, scale=adapter_scale, drop_path=drop_path_rate,
                                            
                                            )

        self.ln_post = LayerNorm(width)
        
        self.shift_op=PatchShift( inv=False)
        

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
            torch.cuda.empty_cache()
        elif self.pretrained is None:
            logger.info('No pretrained weights are loaded!!!!!')
            self.apply(_init_weights)
        else:
            raise TypeError('pretrained must be a str or None')

        ## initialize S_Adapter
        for n, m in self.transformer.named_modules():
            if 'S_Adapter' in n:
                for n2, m2 in m.named_modules():
                    if 'D_fc2' in n2:
                        if isinstance(m2, nn.Linear):
                            nn.init.constant_(m2.weight, 0)
                            nn.init.constant_(m2.bias, 0)
        
        ## initialize T_Adapter
        for n, m in self.transformer.named_modules():
            if 'T_Adapter' in n:
                for n2, m2 in m.named_modules():
                    if 'D_fc2' in n2:
                        if isinstance(m2, nn.Linear):
                            nn.init.constant_(m2.weight, 0)
                            nn.init.constant_(m2.bias, 0)

        ## initialize MLP_Adapter
        for n, m in self.transformer.named_modules():
            if 'MLP_Adapter' in n:
                for n2, m2 in m.named_modules():
                    if 'D_fc2' in n2:
                        if isinstance(m2, nn.Linear):
                            nn.init.constant_(m2.weight, 0)
                            nn.init.constant_(m2.bias, 0)

        
        ## freeze some parameters
        for name, param in self.named_parameters():
            if 'temporal_embedding' not in name and 'ln_post' not in name and 'cls_head' not in name and 'Adapter' not in name and 'rel_pos_t' not in name :
                param.requires_grad = False

        for name, param in self.named_parameters():
            logger.info(f'{name}: {param.requires_grad}')
        
        num_param = sum(p.numel() for p in self.parameters() if p.requires_grad)
        num_total_param = sum(p.numel() for p in self.parameters())
        logger.info(
            f'Number of total parameters: {(num_total_param/1.e6):6.2f}, tunable parameters: {(num_param/1.e6):6.2f}'
        )

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'absolute_pos_embed', 'temporal_embedding'}

    @torch.jit.ignore
    def no_weight_decay_keywords(self):
        return {'relative_position_bias_table', 'temporal_position_bias_table'}

    def forward(self, x: torch.Tensor):
        B, C, T, H, W = x.shape
        x = rearrange(x, 'b c t h w -> (b t) c h w')
        x = self.conv1(x)  
        x = x.reshape(x.shape[0], x.shape[1], -1) # bt d n
        x = x.permute(0, 2, 1)  # bt n d
        
        x = torch.cat([self.class_embedding.to(x.dtype) + torch.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device), x], dim=1)
        
        if self.use_abs_pos_embed:
            x = x + self.positional_embedding.to(x.dtype)

        if self.abs_tpe:
            n = x.shape[1]
            x = rearrange(x, '(b t) n d -> (b n) t d', t=self.num_frames)
            x = x + self.temporal_embedding
            x = rearrange(x, '(b n) t d -> (b t) n d', n=n)
        
        x = self.ln_pre(x)

        x = x.permute(1, 0, 2)  # NLD -> LND    [HW+1, NT, D]
        
        L, BT, D = x.shape
        H = W = int((L-1)**0.5)
        tmp_x = rearrange(x[1:,:,:], '(h w) (b t)  c -> b t h w c', b=B, t=T, h=H, w=W, c=D)
        tmp_x = self.shift_op(tmp_x)
        tmp_x = tmp_x[:, 4::8, :, :, :]
        tmp_x = rearrange(tmp_x, 'b t h w c -> b t (h w) c')
        
        x = rearrange(x,'l (b t) c -> b t l c', b=B, t=T)
        tmp_x= torch.cat([x[:, 4::8, 0, :].unsqueeze(-2),tmp_x], dim=2)
        
        x = torch.cat([x,tmp_x], dim=1)
        x = rearrange(x, 'b t l c -> l (b t) c', b=B)
        
        del tmp_x
        
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_post(x)
        x = x[:, 0]
        x = rearrange(x, '(b t) d -> b d t',b=B,t=T+T//8 )
        
        x = x.unsqueeze(-1).unsqueeze(-1)  # BDTHW for I3D head

        return x


class PatchShift(nn.Module):
    def __init__(self, inv=False, ratio=1):
        super(PatchShift, self).__init__()

        self.inv = inv

        if inv:
            print('=> Using inverse PatchShift,   tps')
        else:
            print('=> Using PatchShift,   tps')

    def forward(self, x):
        x = self.shift(x, inv=self.inv)
        return x #self.net(x)

    @staticmethod
    def shift(x, inv=False):
        B, T , H, W, c = x.size()
        feat = x
        # feat = feat.view(B, T,  H,  W, c)
        out = feat.clone()
        stride = 1
        multiplier = -1 if inv else 1
        ## Pattern C
        out[:, :,  0::3, 0::3,:] = torch.roll(feat[:, :,  0::3,0::3,:], shifts=-4*multiplier*stride, dims=1)
        out[:, :,  0::3, 1::3,:] = torch.roll(feat[:, :,  0::3,1::3,:], shifts=multiplier*stride, dims=1)
        out[:, :,  1::3, 0::3,:] = torch.roll(feat[:, :,  1::3,0::3,:], shifts=-multiplier*stride, dims=1)
        out[:, :,  0::3, 2::3,:] = torch.roll(feat[:, :,  0::3,2::3,:], shifts=2*multiplier*stride, dims=1)
        out[:, :,  2::3, 0::3,:] = torch.roll(feat[:, :,  2::3,0::3,:], shifts=-2*multiplier*stride, dims=1)
        out[:, :,  1::3, 2::3,:] = torch.roll(feat[:, :,  1::3,2::3,:], shifts=3*multiplier*stride, dims=1)
        out[:, :,  2::3, 1::3,:] = torch.roll(feat[:, :,  2::3,1::3,:], shifts=-3*multiplier*stride, dims=1)
        out[:, :,  2::3, 2::3,:] = torch.roll(feat[:, :,  2::3,2::3,:], shifts=4*multiplier*stride, dims=1) 

        # out = out.view(B, T, H, W, c)
        return out


class Rand2dPatchShift(nn.Module):
    def __init__(self, inv=False, sx=1,sy=1,w: int=14, h: int=14):
        super(Rand2dPatchShift, self).__init__()

        self.inv = inv
        # self.ratio = ratio
        self.sx = sx
        self.sy = sy
        
        self.w=w
        self.h=h

        if inv:
            print(f'=> Using inverse Rand2dPatchShift,   {sx}x{sy}-tps')
        else:
            print(f'=> Using Rand2dPatchShift,   {sx}x{sy}-tps')

    def forward(self, x):
        with torch.no_grad():
            generator=init_generator(x.device)
            
            x = self.shift(x, inv=self.inv,generator=generator,sx=self.sx,sy=self.sy,w=self.w,h=self.h)
        return x #self.net(x)

    @staticmethod
    def shift(x, inv=False,generator: torch.Generator = None, sx:int = 1, sy:int = 1, w:int = 14, h:int = 14):
        
        # b t h w c
        B, T, H, W, c = x.size()
        feat = x
        feat = feat.view(B, T, H*W, c)
        # out = feat.clone()
        stride = 1
        multiplier = -1 if inv else 1
        
        # rand pattern
        hsy, wsx = h // sy, w // sx
        rand_idx = torch.randint(sy*sx, size=(hsy, wsx, 1), device=generator.device, generator=generator).to(x.device)
        
        # The image might not divide sx and sy, so we need to work on a view of the top left if the idx buffer instead
        idx_buffer_view = torch.zeros(hsy, wsx, sy*sx, device=x.device, dtype=torch.int64)
        idx_buffer_view.scatter_(dim=2, index=rand_idx, src=-torch.ones_like(rand_idx, dtype=rand_idx.dtype))
        idx_buffer_view = idx_buffer_view.view(hsy, wsx, sy, sx).transpose(1, 2).reshape(hsy * sy, wsx * sx)

        # Image is not divisible by sx or sy so we need to move it into a new buffer
        if (hsy * sy) < h or (wsx * sx) < w:
            idx_buffer = torch.zeros(h, w, device=x.device, dtype=torch.int64)
            idx_buffer[:(hsy * sy), :(wsx * sx)] = idx_buffer_view
        else:
            idx_buffer = idx_buffer_view

        # We set dst tokens to be -1 and src to be 0, so an argsort gives us dst|src indices
        rand_idx = idx_buffer.reshape(-1).argsort(dim=0)

        # We're finished with these
        del idx_buffer, idx_buffer_view
        
        # rand_idx is currently dst|src, so split them
        num_dst = hsy * wsx
        shift_idx = rand_idx[:num_dst].cpu().numpy() # src
        # a_idx = rand_idx[:, num_dst:, :] # src
        # b_idx = rand_idx[:, :num_dst, :] # dst
        
        out=[]
        
        for i in range(shift_idx.shape[0]):
            
            w_idx=shift_idx[i]%7
            h_idx=shift_idx[i]//7
            
            if w_idx%3==0 and h_idx%3==0:
                out.append(torch.roll(feat[:, :,  shift_idx[i] ,:], shifts=-4*multiplier*stride, dims=1))
            elif w_idx%3==0 and h_idx%3==1:
                out.append(torch.roll(feat[:, :,  shift_idx[i] ,:], shifts=multiplier*stride, dims=1))
            elif w_idx%3==0 and h_idx%3==2:
                out.append(torch.roll(feat[:, :,  shift_idx[i] ,:], shifts=2*multiplier*stride, dims=1))
            elif w_idx%3==1 and h_idx%3==0:
                out.append(torch.roll(feat[:, :,  shift_idx[i] ,:], shifts=-multiplier*stride, dims=1))
            elif w_idx%3==1 and h_idx%3==1:
                
                if w_idx!=h_idx:
                    inv=-1
                
                out.append(torch.roll(feat[:, :,  shift_idx[i] ,:], shifts=inv*multiplier*stride, dims=1))
            elif w_idx%3==1 and h_idx%3==2:
                out.append(torch.roll(feat[:, :,  shift_idx[i] ,:], shifts=3*multiplier*stride, dims=1))
            elif w_idx%3==2 and h_idx%3==0:
                out.append(torch.roll(feat[:, :,  shift_idx[i] ,:], shifts=-2*multiplier*stride, dims=1))
            elif w_idx%3==2 and h_idx%3==1:
                out.append(torch.roll(feat[:, :,  shift_idx[i] ,:], shifts=-3*multiplier*stride, dims=1))
            elif w_idx%3==2 and h_idx%3==2:
                out.append(torch.roll(feat[:, :,  shift_idx[i] ,:], shifts=4*multiplier*stride, dims=1)) 
        # b t  L C
        out = torch.stack(out,dim=2)
        out = rearrange(out, 'b t (h w) C -> b t h w C',h=hsy,w=wsx)
        # out = out.view(B, T, H, W, c)
        return out

if __name__ == '__main__':

    from mmengine.analysis import get_model_complexity_info

    # backbone=dict(
    #     type='ViT_CLIP_TPS',
    #     pretrained='openaiclip',
    #     input_resolution=224,
    #     patch_size=16,
    #     num_frames=32,
    #     width=768,
    #     layers=12,
    #     heads=12,
    #     drop_path_rate=0.1),

    input_shape = (3,32,224,224)
    model = ViT_CLIP_VPT(pretrained='openaiclip',
                        input_resolution=224,
                        adapter_scale=0.5,
                        patch_size=16,
                        num_frames=32,
                        width=768,
                        layers=12,
                        heads=12,
                        drop_path_rate=0.1,
                        )

    analysis_results = get_model_complexity_info(model, input_shape)


    print(analysis_results['out_table'])

    # print(analysis_results['out_arch'])

    print(f"Model Flops:{analysis_results['flops_str']}")

    print(f"Model Parameters:{analysis_results['params_str']}")
