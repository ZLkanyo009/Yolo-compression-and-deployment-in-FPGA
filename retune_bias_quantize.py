from __future__ import division

import os
import random
import argparse
import time
import math
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn

from data import *
import tools

from utils.augmentations import SSDAugmentation
from utils.cocoapi_evaluator import COCOAPIEvaluator
from utils.vocapi_evaluator import VOCAPIEvaluator

os.environ["CUDA_VISIBLE_DEVICES"] = "1"

def parse_args():
    parser = argparse.ArgumentParser(description='YOLO Detection')
    parser.add_argument('-v', '--version', default='yolo_v2',
                        help='yolo_v2, yolo_v3, yolo_v3_spp, slim_yolo_v2, tiny_yolo_v3')
    parser.add_argument('-d', '--dataset', default='voc',
                        help='voc or coco')
    parser.add_argument('-hr', '--high_resolution', action='store_true', default=False,
                        help='use high resolution to pretrain.')  
    parser.add_argument('-ms', '--multi_scale', action='store_true', default=False,
                        help='use multi-scale trick')                  
    parser.add_argument('--batch_size', default=32, type=int, 
                        help='Batch size for training')
    parser.add_argument('--lr', default=1e-7, type=float, 
                        help='initial learning rate')
    parser.add_argument('-cos', '--cos', action='store_true', default=False,
                        help='use cos lr')
    parser.add_argument('-no_wp', '--no_warm_up', action='store_true', default=False,
                        help='yes or no to choose using warmup strategy to train')
    parser.add_argument('--wp_epoch', type=int, default=2,
                        help='The upper bound of warm-up')
    parser.add_argument('--start_epoch', type=int, default=0,
                        help='start epoch to train')
    parser.add_argument('-r', '--resume', default=None, type=str, 
                        help='keep training')
    parser.add_argument('--momentum', default=0.9, type=float, 
                        help='Momentum value for optim')
    parser.add_argument('--weight_decay', default=5e-4, type=float, 
                        help='Weight decay for SGD')
    parser.add_argument('--gamma', default=0.1, type=float, 
                        help='Gamma update for SGD')
    parser.add_argument('--num_workers', default=8, type=int, 
                        help='Number of workers used in dataloading')
    parser.add_argument('--eval_epoch', type=int,
                            default=10, help='interval between evaluations')
    parser.add_argument('--cuda', action='store_true', default=False,
                        help='use cuda.')
    parser.add_argument('--tfboard', action='store_true', default=False,
                        help='use tensorboard')
    parser.add_argument('--debug', action='store_true', default=False,
                        help='debug mode where only one image is trained')
    parser.add_argument('--save_folder', default='weights/', type=str, 
                        help='Gamma update for SGD')
    parser.add_argument('-q', '--quantize', action='store_true', default=False,
                        help='quantize the yolo network.')

    return parser.parse_args()

quantized_layers = []
def quantize_tensor(tensor,bitwidth,channel_level=False):
    if channel_level:
        _max = tensor.abs().view(tensor.size(0),-1).max(1)[0]
    else:
        _max = tensor.abs().max()
        
    scale = (2 ** (bitwidth - 1) - 1) / _max
    if tensor.dim() == 4:
        scale = scale.view(-1, 1, 1, 1)
    else:
        scale = scale.view(-1, 1)
    scale = 2**torch.floor(torch.log2(scale))
    new_tensor = torch.round(scale * tensor)
    return new_tensor, scale

def quantize_tensor_b(tensor,bitwidth,channel_level=False):
    if channel_level:
        _max = tensor.abs().view(tensor.size(0),-1).max(1)[0]
    else:
        _max = tensor.abs().max()
        
    scale = (2 ** (bitwidth - 1) - 1) / _max
    scale = 2**torch.floor(torch.log2(scale))
    new_tensor = torch.round(scale * tensor)
    return new_tensor, scale

def init_quantize_net(net,weight_bitwidth):
    for name,m in net.named_modules():
        if isinstance(m,nn.Conv2d) or isinstance(m,nn.Linear):
            if hasattr(m.weight,'weight_back') and hasattr(m.bias,'bias_back'):
                m.weight.weight_back=m.weight.data.clone()
                m.bias.bias_back=m.bias.data.clone()
                continue
            quantized_layers.append(m)

            m.weight.weight_back=m.weight.data.clone()
            m.bias.bias_back=m.bias.data.clone()

def quantize_layers(bitwidth,rescale=True):
    # count = 1
    for layer in quantized_layers:
        with torch.no_grad():
            quantized_w, scale_w=quantize_tensor(layer.weight.weight_back,bitwidth,False)
            quantized_b, scale_b=quantize_tensor_b(layer.bias.bias_back,bitwidth,False)
            
            layer.weight[...]= quantized_w/scale_w if rescale else quantized_w
            layer.bias[...]= quantized_b/scale_b if rescale else quantized_b

def weightsdistribute(model):
    print("================show every layer's weights distribute================")
    for key, value in model.named_parameters():
        print("=================key=================")
        print(key)
        unique, count = torch.unique(value.detach(), sorted=True, return_counts= True)
        print(unique,":", unique.shape)


# ===================== new add quantization 2 ============================
def train():
    args = parse_args()
    path_to_save = os.path.join(args.save_folder, args.dataset, args.version)
    os.makedirs(path_to_save, exist_ok=True)

    # use hi-res backbone
    if args.high_resolution:
        print('use hi-res backbone')
        hr = True
    else:
        hr = False
    
    # cuda
    if args.cuda:
        print('use cuda')
        cudnn.benchmark = True
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    # multi-scale
    if args.multi_scale:
        print('use the multi-scale trick ...')
        train_size = [640, 640]
        val_size = [416, 416]
    else:
        train_size = [240, 320]
        val_size = [240, 320]

    cfg = train_cfg
    # dataset and evaluator
    print("Setting Arguments.. : ", args)
    print("----------------------------------------------------------")
    print('Loading the dataset...')

    if args.dataset == 'voc':
        data_dir = VOC_ROOT
        num_classes = 20
        dataset = VOCDetection(root=data_dir, 
                                transform=SSDAugmentation(train_size)
                                )

        evaluator = VOCAPIEvaluator(data_root=data_dir,
                                    img_size=val_size,
                                    device=device,
                                    transform=BaseTransform(val_size),
                                    labelmap=VOC_CLASSES
                                    )

    elif args.dataset == 'coco':
        data_dir = coco_root
        num_classes = 80
        dataset = COCODataset(
                    data_dir=data_dir,
                    img_size=train_size[0],
                    transform=SSDAugmentation(train_size),
                    debug=args.debug)


        evaluator = COCOAPIEvaluator(
                        data_dir=data_dir,
                        img_size=val_size,
                        device=device,
                        transform=BaseTransform(val_size)
                        )
    
    elif args.dataset == 'mask':
        data_dir = VOC_ROOT
        num_classes = 2

        #VOCDetection内部对xmin xmax等做了归一化处理
        dataset = VOCDetection(root=data_dir, 
                                transform=SSDAugmentation(train_size)
                                )

        evaluator = VOCAPIEvaluator(data_root=data_dir,
                                    img_size=val_size,
                                    device=device,
                                    transform=BaseTransform(val_size),
                                    labelmap=VOC_CLASSES
                                    )
    
    else:
        print('unknow dataset !! Only support voc and coco !!')
        exit(0)
    
    print('Training model on:', dataset.name)
    print('The dataset size:', len(dataset))
    print("----------------------------------------------------------")

    # dataloader
    dataloader = torch.utils.data.DataLoader(
                    dataset, 
                    batch_size=args.batch_size, 
                    shuffle=True, 
                    collate_fn=detection_collate,
                    num_workers=args.num_workers,
                    pin_memory=True
                    )

    # build model
    if args.version == 'yolo_v2':
        from models.yolo_v2 import myYOLOv2
        anchor_size = ANCHOR_SIZE if args.dataset == 'voc' else ANCHOR_SIZE_COCO
    
        yolo_net = myYOLOv2(device, input_size=train_size, num_classes=num_classes, trainable=True, anchor_size=anchor_size, hr=hr)
        print('Let us train yolo_v2 on the %s dataset ......' % (args.dataset))

    elif args.version == 'yolo_v3':
        from models.yolo_v3 import myYOLOv3
        anchor_size = MULTI_ANCHOR_SIZE if args.dataset == 'voc' else MULTI_ANCHOR_SIZE_COCO
        
        yolo_net = myYOLOv3(device, input_size=train_size, num_classes=num_classes, trainable=True, anchor_size=anchor_size, hr=hr)
        print('Let us train yolo_v3 on the %s dataset ......' % (args.dataset))

    elif args.version == 'yolo_v3_spp':
        from models.yolo_v3_spp import myYOLOv3Spp
        anchor_size = MULTI_ANCHOR_SIZE if args.dataset == 'voc' else MULTI_ANCHOR_SIZE_COCO
        
        yolo_net = myYOLOv3Spp(device, input_size=train_size, num_classes=num_classes, trainable=True, anchor_size=anchor_size, hr=hr)
        print('Let us train yolo_v3_spp on the %s dataset ......' % (args.dataset))

    elif args.version == 'slim_yolo_v2':
        from models.slim_yolo_v2 import SlimYOLOv2
        anchor_size = ANCHOR_SIZE if args.dataset == 'voc' else ANCHOR_SIZE_COCO
        if args.dataset == 'mask':
            anchor_size = ANCHOR_SIZE_MASK
    
        yolo_net = SlimYOLOv2(device, input_size=train_size, num_classes=num_classes, trainable=True, anchor_size=anchor_size, hr=hr)
        print('Let us train slim_yolo_v2 on the %s dataset ......' % (args.dataset))
    
    elif args.version == 'slim_yolo_v2_q':
        from models.slim_yolo_v2 import SlimYOLOv2_quantize
        anchor_size = ANCHOR_SIZE if args.dataset == 'voc' else ANCHOR_SIZE_COCO
        if args.dataset == 'mask':
            anchor_size = ANCHOR_SIZE_MASK
    
        yolo_net = SlimYOLOv2_quantize(device, input_size=train_size, num_classes=num_classes, trainable=True, anchor_size=anchor_size, hr=hr)
        print('Let us train slim_yolo_v2 on the %s dataset ......' % (args.dataset))
    
    elif args.version == 'slim_yolo_v2_q_bf':
        from models.slim_yolo_v2 import SlimYOLOv2_quantize_bnfuse
        anchor_size = ANCHOR_SIZE if args.dataset == 'voc' else ANCHOR_SIZE_COCO
        if args.dataset == 'mask':
            anchor_size = ANCHOR_SIZE_MASK
    
        yolo_net = SlimYOLOv2_quantize_bnfuse(device, input_size=train_size, num_classes=num_classes, trainable=True, anchor_size=anchor_size, hr=hr)
        print('Let us train slim_yolo_v2 on the %s dataset ......' % (args.dataset))


    elif args.version == 'tiny_yolo_v3':
        from models.tiny_yolo_v3 import YOLOv3tiny
        anchor_size = TINY_MULTI_ANCHOR_SIZE if args.dataset == 'voc' else TINY_MULTI_ANCHOR_SIZE_COCO
    
        yolo_net = YOLOv3tiny(device, input_size=train_size, num_classes=num_classes, trainable=True, anchor_size=anchor_size, hr=hr)
        print('Let us train tiny_yolo_v3 on the %s dataset ......' % (args.dataset))

    else:
        print('Unknown version !!!')
        exit()

    model = yolo_net
    #weightsdistribute(model)
    model.to(device)
    #model = torch.nn.DataParallel(model)

    model.train()

    weight_bitwidth = 8

    # use tfboard
    if args.tfboard:
        print('use tensorboard')
        from torch.utils.tensorboard import SummaryWriter
        c_time = time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(time.time()))
        log_path = os.path.join('log/coco/', args.version, c_time)
        os.makedirs(log_path, exist_ok=True)

        writer = SummaryWriter(log_path)
    
    # keep training
    if args.resume is not None:
        print('keep training model: %s' % (args.resume))
        model.load_state_dict(torch.load(args.resume, map_location=device), strict = False)

    # optimizer setup
    base_lr = args.lr
    tmp_lr = base_lr
    optimizer = optim.SGD(model.parameters(), 
                            lr=args.lr, 
                            momentum=args.momentum,
                            weight_decay=args.weight_decay
                            )

    max_epoch = cfg['max_epoch']
    epoch_size = len(dataset) // args.batch_size

    # start training loop
    t0 = time.time()

    for epoch in range(args.start_epoch, max_epoch):

        for iter_i, (images, targets) in enumerate(dataloader):
            # WarmUp strategy for learning rate
            set_lr(optimizer, tmp_lr)
        
            # to device
            images = images.to(device)

            # multi-scale trick
            if iter_i % 10 == 0 and iter_i > 0 and args.multi_scale:
                # randomly choose a new size
                size = random.randint(10, 19) * 32
                train_size = [size, size]
                model.set_grid(train_size)
            if args.multi_scale:
                # interpolate
                images = torch.nn.functional.interpolate(images, size=train_size, mode='bilinear', align_corners=False)
            
            # make labels
            targets = [label.tolist() for label in targets]
            if args.version == 'yolo_v2' or args.version == 'slim_yolo_v2' or args.version == 'slim_yolo_v2_q' or args.version == 'slim_yolo_v2_q_bf':
                targets = tools.gt_creator(input_size=train_size, 
                                           stride=yolo_net.stride, 
                                           label_lists=targets, 
                                           anchor_size=anchor_size
                                           )
            else:
                targets = tools.multi_gt_creator(input_size=train_size, 
                                                 strides=yolo_net.stride, 
                                                 label_lists=targets, 
                                                 anchor_size=anchor_size
                                                 )
            targets = torch.tensor(targets).float().to(device)

            # forward and loss
            conf_loss, cls_loss, txtytwth_loss, total_loss = model(images, target=targets, quantization = args.quantize)

            if args.quantize == True: 
                init_quantize_net(model, weight_bitwidth)
                quantize_layers(weight_bitwidth)

                print("max calibrate %d" % iter_i)
                if args.batch_size * iter_i > 1000:
                    print("calibrate num: %d" % (args.batch_size * iter_i))
                    break

                continue

            # weightsdistribute(model)

            # backprop
            total_loss.backward()        
            optimizer.step()
            optimizer.zero_grad()

            # display
            if iter_i % 10 == 0:
                if args.tfboard:
                    # viz loss
                    writer.add_scalar('object loss', conf_loss.item(), iter_i + epoch * epoch_size)
                    writer.add_scalar('class loss', cls_loss.item(), iter_i + epoch * epoch_size)
                    writer.add_scalar('local loss', txtytwth_loss.item(), iter_i + epoch * epoch_size)
                
                t1 = time.time()
                print('[Epoch %d/%d][Iter %d/%d][lr %.6f]'
                    '[Loss: obj %.2f || cls %.2f || bbox %.2f || total %.2f || size %d || time: %.2f]'
                        % (epoch+1, max_epoch, iter_i, epoch_size, tmp_lr,
                            conf_loss.item(), cls_loss.item(), txtytwth_loss.item(), total_loss.item(), train_size[0], t1-t0),
                        flush=True)

                t0 = time.time()

        # evaluation

        model.trainable = False
        model.set_grid(val_size)
        model.eval()

        # evaluate
        evaluator.evaluate(model, quantization = args.quantize)

        # convert to training mode.
        model.trainable = True
        model.set_grid(train_size)
        model.train()

        # save model
        print('Saving state, epoch:', epoch + 1)
        if args.quantize == True:
            torch.save(model.state_dict(), os.path.join(path_to_save, 
                        args.version + '_retune_quantize' + repr(epoch + 1) + '.pth')
                        )  
            break
            
        else:
            torch.save(model.state_dict(), os.path.join(path_to_save, 
                    args.version + '_retune_' + repr(epoch + 1) + '.pth')
                    )  

def set_lr(optimizer, lr):
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


if __name__ == '__main__':
    train()
