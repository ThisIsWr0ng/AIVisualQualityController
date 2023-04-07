num_classes = 4

cfg = []
cfg.append('[net]\n')
cfg.append('batch=64\n')
cfg.append('subdivisions=16\n')
cfg.append('width=416\n')
cfg.append('height=416\n')
cfg.append('channels=3\n')
cfg.append('momentum=0.9\n')
cfg.append('decay=0.0005\n')
cfg.append('angle=0\n')
cfg.append('saturation = 1.5\n')
cfg.append('exposure = 1.5\n')
cfg.append('hue=.1\n')
cfg.append('learning_rate=0.001\n')
cfg.append('burn_in=1000\n')
cfg.append('max_batches = 500200\n')
cfg.append('policy=steps\n')
cfg.append('steps=400000,450000\n')
cfg.append('scales=.1,.1\n')

for i in range(num_classes):
    cfg.append(f'[convolutional]\n')
    cfg.append(f'batch_normalize=1\n')
    cfg.append(f'filters={32*(2**(i))}\n')
    cfg.append(f'size=3\n')
    cfg.append(f'stride=1\n')
    cfg.append(f'pad=1\n')
    if i > 0:
        cfg.append(f'route=previous\n')
    if i == num_classes-1:
        cfg.append(f'[yolo]\n')
        cfg.append(f'mask = 0,1,2\n')
        cfg.append(f'anchors = 10,14, 23,27, 37,58, 81,82, 135,169, 344,319\n')
        cfg.append(f'classes={num_classes}\n')
        cfg.append(f'num={i}\n')
        cfg.append(f'jitter=.3\n')
        cfg.append(f'rescale=1.3\n')
        cfg.append(f'ignore_thresh = .5\n')
        cfg.append(f'trainable=1\n')
        cfg.append(f'random=1\n')
    else:
        cfg.append(f'[convolutional]\n')
        cfg.append(f'batch_normalize=1\n')
        cfg.append(f'filters={32*(2**(i+1))}\n')
        cfg.append(f'size=3\n')
        cfg.append(f'stride=2\n')
        cfg.append(f'pad=1\n')

        for j in range(i):
            cfg.append(f'[convolutional]\n')
            cfg.append(f'batch_normalize=1\n')
            cfg.append(f'filters={32*(2**(i-j))}\n')
            cfg.append(f'size=1\n')
            cfg.append(f'stride=1\n')
            cfg.append(f'pad=1\n')
            
        cfg.append(f'[route]\n')
        cfg.append(f'layers=-1, {i-num_classes}\n')
        cfg.append(f'[reorg]\n')
        cfg.append(f'stride=2\n')

with open('yolov7-tiny-custom.cfg', 'w') as f:
    f.writelines(cfg)
