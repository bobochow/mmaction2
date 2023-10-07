_base_ = [
    '../../_base_/models/vitclip_base.py', '../../_base_/default_runtime.py'
]
# model settings
model = dict(
    backbone=dict(drop_path_rate=0.2, adapter_scale=1, num_tadapter=2, num_frames=8),
    cls_head=dict(num_classes=174,label_smooth_eps=0.1),
    test_cfg=dict(max_testing_views=2), 
    )

# dataset settings
dataset_type = 'VideoDataset'
data_root = 'data/sthv2/videos'
data_root_val = 'data/sthv2/videos'
ann_file_train = 'data/sthv2/sthv2_train_list_videos.txt'
ann_file_val = 'data/sthv2/sthv2_val_list_videos.txt'
ann_file_test = 'data/sthv2/sthv2_val_list_videos.txt'

total_epochs = 50
work_dir = './work_dirs/vitclip_base_sthv2'

file_client_args = dict(io_backend='disk')
train_pipeline = [
    dict(type='DecordInit', **file_client_args),
    dict(type='SampleFrames', clip_len=8, frame_interval=2, num_clips=1, frame_uniform=True),
    dict(type='DecordDecode'),
    dict(type='Resize', scale=(-1, 256)),
    dict(type='RandomResizedCrop'),
    dict(type='Resize', scale=(224, 224), keep_ratio=False),
    dict(type='Flip', flip_ratio=0),
    dict(
        type='PytorchVideoWrapper',
        op='RandAugment',
        magnitude=7,
        num_layers=4),
    dict(type='RandomErasing', erase_prob=0.25, mode='rand'),
    dict(type='FormatShape', input_format='NCTHW'),
    dict(type='PackActionInputs')
]
val_pipeline = [
    dict(type='DecordInit', **file_client_args),
    dict(
        type='SampleFrames',
        clip_len=8,
        frame_interval=2,
        num_clips=1,
        frame_uniform=True,
        test_mode=True),
    dict(type='DecordDecode'),
    dict(type='Resize', scale=(-1, 256)),
    dict(type='CenterCrop', crop_size=224),
    dict(type='Flip', flip_ratio=0),
    dict(type='FormatShape', input_format='NCTHW'),
    dict(type='PackActionInputs')
]
test_pipeline = [
    dict(type='DecordInit', **file_client_args),
    dict(
        type='SampleFrames',
        clip_len=8,
        frame_interval=2,
        num_clips=1,
        frame_uniform=True,
        test_mode=True),
    dict(type='DecordDecode'),
    dict(type='Resize', scale=(-1, 224)),
    dict(type='ThreeCrop', crop_size=224),
    dict(type='Flip', flip_ratio=0),
    dict(type='FormatShape', input_format='NCTHW'),
    dict(type='PackActionInputs')
]

train_dataloader = dict(
    batch_size=32,
    num_workers=8,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    dataset=dict(
        type=dataset_type,
        ann_file=ann_file_train,
        data_prefix=dict(video=data_root),
        pipeline=train_pipeline))
val_dataloader = dict(
    batch_size=1,
    num_workers=1,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        ann_file=ann_file_val,
        data_prefix=dict(video=data_root_val),
        pipeline=val_pipeline,
        test_mode=True))
test_dataloader = dict(
    batch_size=1,
    num_workers=1,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        ann_file=ann_file_test,
        data_prefix=dict(video=data_root_val),
        pipeline=test_pipeline,
        test_mode=True))

val_evaluator = dict(type='AccMetric')
test_evaluator = val_evaluator

train_cfg = dict(
    type='EpochBasedTrainLoop', max_epochs=total_epochs, val_begin=1, val_interval=3)
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')

# optimizer
optim_wrapper = dict(
    type='AmpOptimWrapper',
    optimizer=dict(
        type='AdamW', lr=3e-4, betas=(0.9, 0.999), weight_decay=0.05),
    paramwise_cfg=dict(
        class_embedding=dict(decay_mult=0.),
        positional_embedding=dict(decay_mult=0.),
        ln_1=dict(decay_mult=0.),
        ln_2=dict(decay_mult=0.),
        ln_pre=dict(decay_mult=0.),
        ln_post=dict(decay_mult=0.),
        ))



# learning policy
param_scheduler = [
    dict(
        type='LinearLR',
        start_factor=0.1,
        by_epoch=True,
        begin=0,
        end=2.5,
        convert_to_iter_based=True),
    dict(
        type='CosineAnnealingLR',
        T_max=total_epochs,
        eta_min=0,
        by_epoch=True,
        begin=0,
        end=total_epochs)
]


# runtime settings

default_hooks = dict(
    checkpoint=dict(interval=3, max_keep_ckpts=3,save_best='auto'), 
    logger=dict(interval=100)
    )

visualizer = dict(
    type='ActionVisualizer',
    vis_backends=[
        dict(type='LocalVisBackend'),
        dict(type='TensorboardVisBackend', save_dir=f'{work_dir}/tensorboard'),
        dict(type='WandbVisBackend',init_kwargs=dict(project='vitclip_sthv2', name='vitclip_base_sthv2')),
    ],
)

auto_scale_lr = dict(enable=True, base_batch_size=64)