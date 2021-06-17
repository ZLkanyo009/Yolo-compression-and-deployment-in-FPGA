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
from utils.vocapi_evaluator_mask import VOCAPIEvaluator_mask
import utils.bn_fuse
import utils.modules

os.environ["CUDA_VISIBLE_DEVICES"] = "2"

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
    parser.add_argument('--lr', default=1e-4, type=float, 
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

    return parser.parse_args()

# ===================== new add quantization 1 ============================
quantized_layers = []
def quantize_tensor(layer,tensor,bitwidth,channel_level=False):
    if channel_level:
        _max = tensor.abs().view(tensor.size(0),-1).max(1)[0]
    else:
        _max = tensor.abs().max()
    scale = (2 ** (bitwidth - 1) - 1) / _max
    if tensor.dim() == 4:
        scale = scale.view(-1, 1, 1, 1)
    else:
        scale = scale.view(-1, 1)
    new_tensor = torch.round(scale * tensor)
    return new_tensor, scale

def init_quantize_net(net,weight_bitwidth):
    for name,m in net.named_modules():
        if isinstance(m,nn.Conv2d) or isinstance(m,nn.Linear):
        # if isinstance(m,SpikeConv2d):
            if hasattr(m.weight,'weight_back'):
                continue
            quantized_layers.append(m)
            m.weight.weight_back=m.weight.data.clone()
            if m.bias is not None:
                raise NotImplementedError

def quantize_layers(bitwidth,rescale=True):
    for layer in quantized_layers:
        with torch.no_grad():
            quantized_w, scale=quantize_tensor(layer,layer.weight.weight_back,bitwidth,False)
            layer.weight[...]= quantized_w/scale if rescale else quantized_w

def weightsdistribute(model):
    print("================show every layer's weights distribute================")
    for key, value in model.named_parameters():
        unique, count = torch.unique(value.detach(), sorted=True, return_counts= True)
        print(unique,":", unique.shape)

def fuse(self):
    # Fuse Conv2d + BatchNorm2d layers throughout model
    fused_list = nn.ModuleList()

    for a in list(self.children())[0]:
        if isinstance(a, nn.Sequential):
            for i, b in enumerate(a):
                if isinstance(b, nn.modules.batchnorm.BatchNorm2d):
                    # fuse this bn layer with the previous conv2d layer
                    conv = a[i - 1]
                    fused = fuse_conv_and_bn(conv, b)
                    a = nn.Sequential(fused, *list(a.children())[i + 1:])
                    break
        fused_list.append(a)
    self.module_list = fused_list

def fuse_conv_and_bn(conv, bn):
    # https://tehnokv.com/posts/fusing-batchnorm-and-conv/
    with torch.no_grad():
        # init
        fusedconv = torch.nn.Conv2d(conv.in_channels,
                                    conv.out_channels,
                                    kernel_size=conv.kernel_size,
                                    stride=conv.stride,
                                    padding=conv.padding,
                                    bias=True)

        # prepare filters
        w_conv = conv.weight.clone().view(conv.out_channels, -1)
        w_bn = torch.diag(bn.weight.div(torch.sqrt(bn.eps + bn.running_var)))
        fusedconv.weight.copy_(torch.mm(w_bn, w_conv).view(fusedconv.weight.size()))

        # prepare spatial bias
        if conv.bias is not None:
            b_conv = conv.bias
        else:
            b_conv = torch.zeros(conv.weight.size(0))
        b_bn = bn.bias - bn.weight.mul(bn.running_mean).div(torch.sqrt(bn.running_var + bn.eps))
        fusedconv.bias.copy_(b_conv + b_bn)

        return fusedconv


# ===================== new add quantization 2 ============================
def trans():
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
        data_dir = VOC_ROOT_mask
        num_classes = 2

        #VOCDetection内部对xmin xmax等做了归一化处理
        dataset = VOCDetection_mask(root=data_dir, 
                                    transform=SSDAugmentation(train_size)
                                   )
        # for i in range(0, len(dataset)):
        #     dataset.pull_item(i)
        # assert False

        evaluator = VOCAPIEvaluator_mask(data_root=data_dir,
                                         img_size=val_size,
                                         device=device,
                                         transform=BaseTransform(val_size),
                                         labelmap=VOC_CLASSES_mask
                                        )
    
    else:
        print('unknow dataset !! Only support voc and coco !!')
        exit(0)
    
    print('Training model on:', dataset.name)
    print('The dataset size:', len(dataset))
    print("----------------------------------------------------------")


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

    elif args.version == 'tiny_yolo_v3':
        from models.tiny_yolo_v3 import YOLOv3tiny
        anchor_size = TINY_MULTI_ANCHOR_SIZE if args.dataset == 'voc' else TINY_MULTI_ANCHOR_SIZE_COCO
    
        yolo_net = YOLOv3tiny(device, input_size=train_size, num_classes=num_classes, trainable=True, anchor_size=anchor_size, hr=hr)
        print('Let us train tiny_yolo_v3 on the %s dataset ......' % (args.dataset))

    else:
        print('Unknown version !!!')
        exit()

    model = yolo_net
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
        model.load_state_dict(torch.load(args.resume, map_location=device))

    # Fuse Conv2d + BatchNorm2d layers throughout model
    fused_list = nn.ModuleList()

    for a in list(model.children()):
        if isinstance(a, utils.modules.Conv2d):
            for i, b in enumerate(a.convs):
                if isinstance(b, nn.modules.batchnorm.BatchNorm2d):
                    # fuse this bn layer with the previous conv2d layer
                    conv = a.convs[i - 1]
                    fused = fuse_conv_and_bn(conv, b)
                    a.convs = nn.Sequential(fused, *list(a.convs.children())[i + 1:])
                    break
        fused_list.append(a)
    print("===== after bn_fuse ======")
    for a in list(model.children()):
        print(a)
    
    model.to(device)


    model.trainable = False
    model.set_grid(val_size)
    model.eval()

    # evaluate
    evaluator.evaluate(model)

    # save model
    print('Saving conv bn fuse model')
    torch.save(model.state_dict(), os.path.join(path_to_save, 
                args.version + '_bnfuse'+ '.pth')
                )  


def set_lr(optimizer, lr):
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


if __name__ == '__main__':
    trans()
