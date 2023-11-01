# model settings
model = dict(
    type='Recognizer3D',
    backbone=dict(
        type='ViT_CLIP',
        pretrained='openaiclip',
        input_resolution=224,
        patch_size=16,
        num_frames=32,
        width=768,
        layers=12,
        heads=12,
        drop_path_rate=0.1),
    data_preprocessor=dict(
        type='ActionDataPreprocessor',
        mean=[108.3272985, 116.7460125, 104.09373615000001],
        std=[68.5005327, 66.6321579, 70.32316305],
        format_shape='NCTHW'),
    cls_head=dict(
        type='I3DHead',
        in_channels=768,
        num_classes=400,
        spatial_type='avg',
        dropout_ratio=0.5,
        average_clips='prob',
        ),
    )