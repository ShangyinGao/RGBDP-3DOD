# model settings
model = dict(
    type='rgbdp',
    backbone_rgb=dict(
        type='resnet',
        # some other config can be also added
        ),
    backbone_depth=dict(),
    backbone_pointcloud=dict(
        type='Pointnet2MSG'
        ),
    neck=dict(
        type='rpn',
        ),
    head=dict(
        type='resnet',
        )
)

# dataset settings
dataset_type = 'CocoDataset'
data_root = 'data/coco/'

# optimizer
optimizer = dict(type='SGD', lr=0.01, momentum=0.9, weight_decay=0.0001)
optimizer_config = dict(grad_clip=dict(max_norm=35, norm_type=2))

# learning policy
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=500,
    warmup_ratio=1.0 / 3,
    step=[8, 11])
checkpoint_config = dict(interval=1)
# yapf:disable
log_config = dict(
    interval=50,
    hooks=[
        dict(type='TextLoggerHook'),
        # dict(type='TensorboardLoggerHook')
    ])
# yapf:enable

# runtime settings
total_epochs = 12
log_level = 'INFO'
batch_size = 1