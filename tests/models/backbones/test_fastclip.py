import torch

from mmaction.models import ViT_CLIP_UTUNER,ViT_CLIP_FLASH
from mmaction.testing import generate_backbone_demo_inputs

from mmengine.runner.amp import autocast

# # @pytest.mark.parametrize("fused_mlp", [False, True])
# @pytest.mark.parametrize('fused_mlp', [False])
# # @pytest.mark.parametrize("optimized", [False, True])
# @pytest.mark.parametrize('optimized', [True])
def test_fast_clip():
    """Check that our implementation of ViT matches the timm's implementation:
    the output of our forward pass in fp16 should be around the same as
    timm' forward pass in fp16, when compared to timm's forward pass in fp32.
    """
    dtype = torch.float16
    device = "cuda"

    # kwargs = {}
    # if optimized:
    #     kwargs = dict(use_flash_attn=True, fused_bias_fc=True, fused_dropout_add_ln=True)
    # kwargs["fused_mlp"] = fused_mlp
    # model = flash_vit_base_patch16_224(**kwargs).to(device=device, dtype=dtype)

    # model_ref = vit_base_patch16_224(pretrained=True).to(device=device)
    # model_timm = vit_base_patch16_224(pretrained=True).to(device=device, dtype=dtype)

    # input_shape = (3, 32, 224, 224)
    model = ViT_CLIP_FLASH(pretrained='openaiclip',
                        input_resolution=224,
                        adapter_scale=0.5,
                        patch_size=16,
                        num_frames=4,
                        width=768,
                        layers=12,
                        heads=12,
                        drop_path_rate=0.1,
                        shift = False,
                        use_flash_attn=True,
                        ).to(device=device, dtype=dtype)
    model.init_weights()
    model.eval()
    
    model_ref = ViT_CLIP_UTUNER(pretrained='openaiclip',
                        input_resolution=224,
                        adapter_scale=0.5,
                        patch_size=16,
                        num_frames=4,
                        width=768,
                        layers=12,
                        heads=12,
                        drop_path_rate=0.1,
                        shift = False,
                        ).to(device=device)
    model_ref.init_weights()
    
    model_ref.eval()
    
    model_fp16 = ViT_CLIP_UTUNER(pretrained='openaiclip',
                        input_resolution=224,
                        adapter_scale=0.5,
                        patch_size=16,
                        num_frames=4,
                        width=768,
                        layers=12,
                        heads=12,
                        drop_path_rate=0.1,
                        shift = False,
                        ).to(device=device, dtype=dtype)
    model_fp16.init_weights()
    model_fp16.eval()

    torch.manual_seed(0)
    
    x = torch.randn(2, 3 , 4, 224, 224, device=device, dtype=dtype)
    
    # input_shape = (2, 3, 4, 224, 224)
    # x = generate_backbone_demo_inputs(input_shape)
    
    with autocast(enabled=True,device_type='cuda',dtype=torch.float16):
        print('start flash-attn!!!!!!!!!!!')
        out = model(x)
        print('start fp16 !!!!!!!!!!!')
        out_timm = model_fp16(x)
    print('start fp32 !!!!!!!!!!!')
    out_ref = model_ref(x.float())

    print(f"Output max diff: {(out - out_ref).abs().max().item()}")
    print(f"Output mean diff: {(out - out_ref).abs().mean().item()}")
    print(f"origin fp16 max diff: {(out_timm - out_ref).abs().max().item()}")
    print(f"origin fp16 mean diff: {(out_timm - out_ref).abs().mean().item()}")
    rtol = 2 
    assert (out - out_ref).abs().max().item() < rtol * (out_timm - out_ref).abs().max().item()
