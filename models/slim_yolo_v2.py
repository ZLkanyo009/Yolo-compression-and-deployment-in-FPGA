import torch
import torch.nn as nn
import torch.nn.functional as F
from utils import Conv2d, reorg_layer, Conv2d_fuse
from backbone import *
import numpy as np
import tools

class AveragedRangeTracker(nn.Module):
    def __init__(self, momentum=0.1):
        super().__init__()
        self.momentum = momentum
        self.register_buffer('scale', torch.zeros(1))
        self.register_buffer('first_a', torch.zeros(1))

    def quantize_activation(self, activation, bitwidth=8, rescale=True, 
                            quantization = False, freeze = False):
        if(quantization == False):
            return activation

        with torch.no_grad():
            _max = activation.abs().max()
            scale = (2 ** (bitwidth - 1) - 1) / _max

            if self.first_a == 0:
                self.first_a.add_(1)
                self.scale.add_(scale)
            elif freeze == True:
                self.scale = self.scale
            else:
                self.scale.mul_(1 - self.momentum).add_(scale * self.momentum)
        
            scale_log2 = 2**torch.floor(torch.log2(self.scale))

            quantized_a = torch.round(scale_log2 * activation)
            quantized_a.requires_grad = True

        return quantized_a/scale_log2 if rescale else quantized_a

class SlimYOLOv2_quantize_bnfuse(nn.Module):
    
    def __init__(self, device, input_size=None, num_classes=20, trainable=False, conf_thresh=0.01, nms_thresh=0.5, anchor_size=None, hr=False):
        super(SlimYOLOv2_quantize_bnfuse, self).__init__()
        self.device = device
        self.input_size = input_size
        self.num_classes = num_classes
        self.trainable = trainable
        self.conf_thresh = conf_thresh
        self.nms_thresh = nms_thresh
        self.anchor_size = torch.tensor(anchor_size)
        self.anchor_number = len(anchor_size)
        self.stride = 16
        # init set
        self.grid_cell, self.all_anchor_wh = self.create_grid(input_size)
        self.scale = np.array([[[input_size[1], input_size[0], input_size[1], input_size[0]]]])
        self.scale_torch = torch.tensor(self.scale.copy(), device=device).float()
        
        self.a_tracker_in = AveragedRangeTracker()
        self.conv1 = Conv2d_fuse(3, 16, 3, 1, leakyReLU=True)
        self.a_tracker1 = AveragedRangeTracker()
        self.pool1  = nn.MaxPool2d(2, 2)

        self.conv2 = Conv2d_fuse(16, 32, 3, 1, leakyReLU=True)
        self.a_tracker2 = AveragedRangeTracker()
        self.pool2  = nn.MaxPool2d(2, 2)

        self.conv3_1 = Conv2d_fuse(32, 64, 3, 1, leakyReLU=True)
        self.a_tracker3_1 = AveragedRangeTracker()
        self.conv3_2 = Conv2d_fuse(64, 64, 3, 1, leakyReLU=True)
        self.a_tracker3_2 = AveragedRangeTracker()
        self.pool3  = nn.MaxPool2d(2, 2)

        self.conv4_1 = Conv2d_fuse(64, 128, 3, 1, leakyReLU=True)
        self.a_tracker4_1 = AveragedRangeTracker()
        self.conv4_2 = Conv2d_fuse(128, 128, 3, 1, leakyReLU=True)
        self.a_tracker4_2 = AveragedRangeTracker()
        self.pool4  = nn.MaxPool2d(2, 2)

        self.conv5 = Conv2d_fuse(128, 256, 3, 1, leakyReLU=True)
        self.a_tracker5 = AveragedRangeTracker()
        self.conv6 = Conv2d_fuse(256, 256, 3, 1, leakyReLU=True)
        self.a_tracker6 = AveragedRangeTracker()
        self.conv7 = Conv2d_fuse(256, 256, 3, 1, leakyReLU=True)
        self.a_tracker7 = AveragedRangeTracker()
        
        # prediction layer
        self.pred = nn.Conv2d(256, self.anchor_number*(1 + 4 + self.num_classes), 3, 1, padding=1)

        self.a_tracker_pred = AveragedRangeTracker()

    def create_grid(self, input_size):
        w, h = input_size[1], input_size[0]
        # generate grid cells
        ws, hs = round(w / self.stride), round(h / self.stride)
        grid_y, grid_x = torch.meshgrid([torch.arange(hs), torch.arange(ws)])
        grid_xy = torch.stack([grid_x, grid_y], dim=-1).float()
        grid_xy = grid_xy.view(1, hs*ws, 1, 2).to(self.device)

        # generate anchor_wh tensor
        anchor_wh = self.anchor_size.repeat(hs*ws, 1, 1).unsqueeze(0).to(self.device)


        return grid_xy, anchor_wh

    def set_grid(self, input_size):
        self.input_size = input_size
        self.grid_cell, self.all_anchor_wh = self.create_grid(input_size)
        self.scale = np.array([[[input_size[1], input_size[0], input_size[1], input_size[0]]]])
        self.scale_torch = torch.tensor(self.scale.copy(), device=self.device).float()

    def decode_xywh(self, txtytwth_pred):
        """
            Input:
                txtytwth_pred : [B, H*W, anchor_n, 4] containing [tx, ty, tw, th]
            Output:
                xywh_pred : [B, H*W*anchor_n, 4] containing [xmin, ymin, xmax, ymax]
        """
        B, HW, ab_n, _ = txtytwth_pred.size()
        xy_pred = torch.sigmoid(txtytwth_pred[:, :, :, :2]) + self.grid_cell
        wh_pred = torch.exp(txtytwth_pred[:, :, :, 2:]) * self.all_anchor_wh
        # [H*W, anchor_n, 4] -> [H*W*anchor_n, 4]
        xywh_pred = torch.cat([xy_pred, wh_pred], -1).view(B, HW*ab_n, 4) * self.stride

        return xywh_pred
    
    def decode_boxes(self, txtytwth_pred, requires_grad=False):
        """
            Input:
                txtytwth_pred : [B, H*W, anchor_n, 4] containing [tx, ty, tw, th]
            Output:
                x1y1x2y2_pred : [B, H*W*anchor_n, 4] containing [xmin, ymin, xmax, ymax]
        """
        # [H*W*anchor_n, 4]
        xywh_pred = self.decode_xywh(txtytwth_pred)

        # [center_x, center_y, w, h] -> [xmin, ymin, xmax, ymax]
        x1y1x2y2_pred = torch.zeros_like(xywh_pred, requires_grad=requires_grad)
        x1y1x2y2_pred[:, :, 0] = (xywh_pred[:, :, 0] - xywh_pred[:, :, 2] / 2)
        x1y1x2y2_pred[:, :, 1] = (xywh_pred[:, :, 1] - xywh_pred[:, :, 3] / 2)
        x1y1x2y2_pred[:, :, 2] = (xywh_pred[:, :, 0] + xywh_pred[:, :, 2] / 2)
        x1y1x2y2_pred[:, :, 3] = (xywh_pred[:, :, 1] + xywh_pred[:, :, 3] / 2)
        
        return x1y1x2y2_pred

    def nms(self, dets, scores):
        """"Pure Python NMS baseline."""
        x1 = dets[:, 0]  #xmin
        y1 = dets[:, 1]  #ymin
        x2 = dets[:, 2]  #xmax
        y2 = dets[:, 3]  #ymax

        areas = (x2 - x1) * (y2 - y1)                 # the size of bbox
        order = scores.argsort()[::-1]                        # sort bounding boxes by decreasing order

        keep = []                                             # store the final bounding boxes
        while order.size > 0:
            i = order[0]                                      #the index of the bbox with highest confidence
            keep.append(i)                                    #save it to keep
            xx1 = np.maximum(x1[i], x1[order[1:]])
            yy1 = np.maximum(y1[i], y1[order[1:]])
            xx2 = np.minimum(x2[i], x2[order[1:]])
            yy2 = np.minimum(y2[i], y2[order[1:]])

            w = np.maximum(1e-28, xx2 - xx1)
            h = np.maximum(1e-28, yy2 - yy1)
            inter = w * h

            # Cross Area / (bbox + particular area - Cross Area)
            ovr = inter / (areas[i] + areas[order[1:]] - inter)
            #reserve all the boundingbox whose ovr less than thresh
            inds = np.where(ovr <= self.nms_thresh)[0]
            order = order[inds + 1]

        return keep

    def postprocess(self, all_local, all_conf):
        """
        bbox_pred: (HxW*anchor_n, 4), bsize = 1
        prob_pred: (HxW*anchor_n, num_classes), bsize = 1
        """
        bbox_pred = all_local
        prob_pred = all_conf

        cls_inds = np.argmax(prob_pred, axis=1)
        prob_pred = prob_pred[(np.arange(prob_pred.shape[0]), cls_inds)]
        scores = prob_pred.copy()
        
        # threshold
        keep = np.where(scores >= self.conf_thresh)
        bbox_pred = bbox_pred[keep]
        scores = scores[keep]
        cls_inds = cls_inds[keep]

        # NMS
        keep = np.zeros(len(bbox_pred), dtype=np.int)
        for i in range(self.num_classes):
            inds = np.where(cls_inds == i)[0]
            if len(inds) == 0:
                continue
            c_bboxes = bbox_pred[inds]
            c_scores = scores[inds]
            c_keep = self.nms(c_bboxes, c_scores)
            keep[inds[c_keep]] = 1

        keep = np.where(keep > 0)
        bbox_pred = bbox_pred[keep]
        scores = scores[keep]
        cls_inds = cls_inds[keep]

        return bbox_pred, scores, cls_inds

    def forward(self, x, target=None, quantization = False, find = False):
        
        bitwidth = 8
        freeze = not self.trainable
        rescale = True

        output = self.a_tracker_in.quantize_activation(x, bitwidth, rescale, quantization, freeze)

        output = self.conv1(output)
        
        if find:
            if(output.abs().max() >= 2**(16-1)): #32768
                print("too high!!!")
                print(output.abs().max())
                assert False
            output = output / 2**11

        output = self.a_tracker1.quantize_activation(output, bitwidth, rescale, quantization, freeze)
        
        output = self.pool1(output)

        output = self.conv2(output)

        if find:
            if(output.abs().max() >= 2**(16-1)): #32768
                print("too high!!!")
                print(output.abs().max())
                assert False
            output = output / 2**10

        output = self.a_tracker2.quantize_activation(output, bitwidth, rescale, quantization, freeze)
        
        output = self.pool2(output)

        output = self.conv3_1(output)

        if find:
            if(output.abs().max() >= 2**(16-1)): #32768
                print("too high!!!")
                print(output.abs().max())
                assert False
            output = output / 2**10
        
        output = self.a_tracker3_1.quantize_activation(output, bitwidth, rescale, quantization, freeze)
        output = self.conv3_2(output)

        if find:
            if(output.abs().max() >= 2**(16-1)): #32768
                print("too high!!!")
                print(output.abs().max())
                assert False
            output = output / 2**11
        
        output = self.a_tracker3_2.quantize_activation(output, bitwidth, rescale, quantization, freeze)
        output = self.pool3(output)

        output = self.conv4_1(output)

        if find:
            if(output.abs().max() >= 2**(16-1)): #32768
                print("too high!!!")
                print(output.abs().max())
                assert False
            output = output / 2**11
        
        output = self.a_tracker4_1.quantize_activation(output, bitwidth, rescale, quantization, freeze)
        output = self.conv4_2(output)
        
        if find:
            if(output.abs().max() >= 2**(16-1)): #32768
                print("too high!!!")
                print(output.abs().max())
                assert False
            output = output / 2**10

        output = self.a_tracker4_2.quantize_activation(output, bitwidth, rescale, quantization, freeze)
        output = self.pool4(output)

        output = self.conv5(output)

        if find:
            if(output.abs().max() >= 2**(16-1)): #32768
                print("too high!!!")
                print(output.abs().max())
                assert False
            output = output / 2**11
        
        output = self.a_tracker5.quantize_activation(output, bitwidth, rescale, quantization, freeze)
        output = self.conv6(output)

        if find:
            if(output.abs().max() >= 2**(16-1)): #32768
                print("too high!!!")
                print(output.abs().max())
                assert False
            output = output / 2**11

        output = self.a_tracker6.quantize_activation(output, bitwidth, rescale, quantization, freeze)
        output = self.conv7(output)

        if find:
            if(output.abs().max() >= 2**(16-1)): #32768
                print("too high!!!")
                print(output.abs().max())
                assert False
            output = output / 2**11
        
        output = self.a_tracker7.quantize_activation(output, bitwidth, rescale, quantization, freeze)
        
        prediction = self.pred(output)
        if find:
            if(prediction.abs().max() >= 2**(16-1)): #32768
                print("too high!!!")
                print(prediction.abs().max())
                assert False
            prediction = prediction / 2**10
        prediction = self.a_tracker_pred.quantize_activation(prediction, bitwidth, rescale, quantization, freeze)

        B, abC, H, W = prediction.size()

        # [B, anchor_n * C, N, M] -> [B, N, M, anchor_n * C] -> [B, N*M, anchor_n*C]
        prediction = prediction.permute(0, 2, 3, 1).contiguous().view(B, H*W, abC)

        # Divide prediction to conf_pred, txtytwth_pred and cls_pred   
        # [B, H*W*anchor_n, 1]
        conf_pred = prediction[:, :, :1 * self.anchor_number].contiguous().view(B, H*W*self.anchor_number, 1)
        # [B, H*W, anchor_n, num_cls]
        cls_pred = prediction[:, :, 1 * self.anchor_number : (1 + self.num_classes) * self.anchor_number].contiguous().view(B, H*W*self.anchor_number, self.num_classes)
        # [B, H*W, anchor_n, 4]
        txtytwth_pred = prediction[:, :, (1 + self.num_classes) * self.anchor_number:].contiguous()
        
        # test
        if not self.trainable:
            txtytwth_pred = txtytwth_pred.view(B, H*W, self.anchor_number, 4)
            with torch.no_grad():
                # batch size = 1                
                all_obj = torch.sigmoid(conf_pred)[0]           # 0 is because that these is only 1 batch.
                all_bbox = torch.clamp((self.decode_boxes(txtytwth_pred) / self.scale_torch)[0], 0., 1.)
                all_class = (torch.softmax(cls_pred[0, :, :], 1) * all_obj)
                # separate box pred and class conf
                all_obj = all_obj.to('cpu').numpy()
                all_class = all_class.to('cpu').numpy()
                all_bbox = all_bbox.to('cpu').numpy()

                bboxes, scores, cls_inds = self.postprocess(all_bbox, all_class)

                return bboxes, scores, cls_inds

        else:
            txtytwth_pred = txtytwth_pred.view(B, H*W, self.anchor_number, 4)
            # decode bbox, and remember to cancel its grad since we set iou as the label of objectness.
            with torch.no_grad():
                x1y1x2y2_pred = (self.decode_boxes(txtytwth_pred) / self.scale_torch).view(-1, 4)

            txtytwth_pred = txtytwth_pred.view(B, H*W*self.anchor_number, 4)

            x1y1x2y2_gt = target[:, :, 7:].view(-1, 4)

            # compute iou
            iou = tools.iou_score(x1y1x2y2_pred, x1y1x2y2_gt).view(B, H*W*self.anchor_number, 1)

            # we set iou between pred bbox and gt bbox as conf label. 
            # [obj, cls, txtytwth, x1y1x2y2] -> [conf, obj, cls, txtytwth]
            target = torch.cat([iou, target[:, :, :7]], dim=2)

            conf_loss, cls_loss, txtytwth_loss, total_loss = tools.loss(pred_conf=conf_pred, pred_cls=cls_pred,
                                                                        pred_txtytwth=txtytwth_pred,
                                                                        label=target,
                                                                        num_classes=self.num_classes)

            return conf_loss, cls_loss, txtytwth_loss, total_loss


class SlimYOLOv2(nn.Module):
    def __init__(self, device, input_size=None, num_classes=20, trainable=False, conf_thresh=0.01, nms_thresh=0.5, anchor_size=None, hr=False):
        super(SlimYOLOv2, self).__init__()
        self.device = device
        self.input_size = input_size
        self.num_classes = num_classes
        self.trainable = trainable
        self.conf_thresh = conf_thresh
        self.nms_thresh = nms_thresh
        self.anchor_size = torch.tensor(anchor_size)
        self.anchor_number = len(anchor_size)
        self.stride = 16
        # init set
        self.grid_cell, self.all_anchor_wh = self.create_grid(input_size)
        self.scale = np.array([[[input_size[1], input_size[0], input_size[1], input_size[0]]]])
        self.scale_torch = torch.tensor(self.scale.copy(), device=device).float()
        
        # backbone
        self.conv1 = Conv2d(3, 16, 3, 1, leakyReLU=True)
        self.pool1  = nn.MaxPool2d(2, 2)

        self.conv2 = Conv2d(16, 32, 3, 1, leakyReLU=True)
        self.pool2  = nn.MaxPool2d(2, 2)

        self.conv3_1 = Conv2d(32, 64, 3, 1, leakyReLU=True)
        self.conv3_2 = Conv2d(64, 64, 3, 1, leakyReLU=True)
        self.pool3  = nn.MaxPool2d(2, 2)

        self.conv4_1 = Conv2d(64, 128, 3, 1, leakyReLU=True)
        self.conv4_2 = Conv2d(128, 128, 3, 1, leakyReLU=True)
        self.pool4  = nn.MaxPool2d(2, 2)

        self.conv5 = Conv2d(128, 256, 3, 1, leakyReLU=True)
        self.conv6 = Conv2d(256, 256, 3, 1, leakyReLU=True)
        self.conv7 = Conv2d(256, 256, 3, 1, leakyReLU=True)
        
        # prediction layer
        self.pred = nn.Conv2d(256, self.anchor_number*(1 + 4 + self.num_classes), 3, 1, padding=1)

    def create_grid(self, input_size):
        w, h = input_size[1], input_size[0]
        # generate grid cells
        ws, hs = round(w / self.stride), round(h / self.stride)

        grid_y, grid_x = torch.meshgrid([torch.arange(hs), torch.arange(ws)])
        grid_xy = torch.stack([grid_x, grid_y], dim=-1).float()
        grid_xy = grid_xy.view(1, hs*ws, 1, 2).to(self.device)

        # generate anchor_wh tensor
        anchor_wh = self.anchor_size.repeat(hs*ws, 1, 1).unsqueeze(0).to(self.device)


        return grid_xy, anchor_wh

    def set_grid(self, input_size):
        self.input_size = input_size
        self.grid_cell, self.all_anchor_wh = self.create_grid(input_size)
        self.scale = np.array([[[input_size[1], input_size[0], input_size[1], input_size[0]]]])
        self.scale_torch = torch.tensor(self.scale.copy(), device=self.device).float()

    def decode_xywh(self, txtytwth_pred):
        """
            Input:
                txtytwth_pred : [B, H*W, anchor_n, 4] containing [tx, ty, tw, th]
            Output:
                xywh_pred : [B, H*W*anchor_n, 4] containing [xmin, ymin, xmax, ymax]
        """
        # b_x = sigmoid(tx) + gride_x,  b_y = sigmoid(ty) + gride_y
        B, HW, ab_n, _ = txtytwth_pred.size()
        
        xy_pred = torch.sigmoid(txtytwth_pred[:, :, :, :2]) + self.grid_cell
        # b_w = anchor_w * exp(tw),     b_h = anchor_h * exp(th)
        wh_pred = torch.exp(txtytwth_pred[:, :, :, 2:]) * self.all_anchor_wh
        # [H*W, anchor_n, 4] -> [H*W*anchor_n, 4]
        xywh_pred = torch.cat([xy_pred, wh_pred], -1).view(B, HW*ab_n, 4) * self.stride

        return xywh_pred
    
    def decode_boxes(self, txtytwth_pred, requires_grad=False):
        """
            Input:
                txtytwth_pred : [B, H*W, anchor_n, 4] containing [tx, ty, tw, th]
            Output:
                x1y1x2y2_pred : [B, H*W*anchor_n, 4] containing [xmin, ymin, xmax, ymax]
        """
        # [H*W*anchor_n, 4]
        xywh_pred = self.decode_xywh(txtytwth_pred)

        # [center_x, center_y, w, h] -> [xmin, ymin, xmax, ymax]
        x1y1x2y2_pred = torch.zeros_like(xywh_pred, requires_grad=requires_grad)
        x1y1x2y2_pred[:, :, 0] = (xywh_pred[:, :, 0] - xywh_pred[:, :, 2] / 2)
        x1y1x2y2_pred[:, :, 1] = (xywh_pred[:, :, 1] - xywh_pred[:, :, 3] / 2)
        x1y1x2y2_pred[:, :, 2] = (xywh_pred[:, :, 0] + xywh_pred[:, :, 2] / 2)
        x1y1x2y2_pred[:, :, 3] = (xywh_pred[:, :, 1] + xywh_pred[:, :, 3] / 2)
        
        return x1y1x2y2_pred

    def nms(self, dets, scores):
        """"Pure Python NMS baseline."""
        x1 = dets[:, 0]  #xmin
        y1 = dets[:, 1]  #ymin
        x2 = dets[:, 2]  #xmax
        y2 = dets[:, 3]  #ymax

        areas = (x2 - x1) * (y2 - y1)                 # the size of bbox
        order = scores.argsort()[::-1]                        # sort bounding boxes by decreasing order

        keep = []                                             # store the final bounding boxes
        while order.size > 0:
            i = order[0]                                      #the index of the bbox with highest confidence
            keep.append(i)                                    #save it to keep
            xx1 = np.maximum(x1[i], x1[order[1:]])
            yy1 = np.maximum(y1[i], y1[order[1:]])
            xx2 = np.minimum(x2[i], x2[order[1:]])
            yy2 = np.minimum(y2[i], y2[order[1:]])

            w = np.maximum(1e-28, xx2 - xx1)
            h = np.maximum(1e-28, yy2 - yy1)
            inter = w * h

            # Cross Area / (bbox + particular area - Cross Area)
            ovr = inter / (areas[i] + areas[order[1:]] - inter)
            #reserve all the boundingbox whose ovr less than thresh
            inds = np.where(ovr <= self.nms_thresh)[0]
            order = order[inds + 1]

        return keep

    def postprocess(self, all_local, all_conf):
        """
        bbox_pred: (HxW*anchor_n, 4), bsize = 1
        prob_pred: (HxW*anchor_n, num_classes), bsize = 1
        """
        bbox_pred = all_local
        prob_pred = all_conf

        cls_inds = np.argmax(prob_pred, axis=1)
        prob_pred = prob_pred[(np.arange(prob_pred.shape[0]), cls_inds)]
        scores = prob_pred.copy()
        
        # threshold
        keep = np.where(scores >= self.conf_thresh)
        bbox_pred = bbox_pred[keep]
        scores = scores[keep]
        cls_inds = cls_inds[keep]

        # NMS
        keep = np.zeros(len(bbox_pred), dtype=np.int)
        for i in range(self.num_classes):
            inds = np.where(cls_inds == i)[0]
            if len(inds) == 0:
                continue
            c_bboxes = bbox_pred[inds]
            c_scores = scores[inds]
            c_keep = self.nms(c_bboxes, c_scores)
            keep[inds[c_keep]] = 1

        keep = np.where(keep > 0)
        bbox_pred = bbox_pred[keep]
        scores = scores[keep]
        cls_inds = cls_inds[keep]

        return bbox_pred, scores, cls_inds

    def forward(self, x, target=None, quantization = False, find = False):
        
        output = self.conv1(x)
        output = self.pool1(output)

        output = self.conv2(output)
        output = self.pool2(output)

        output = self.conv3_1(output)
        output = self.conv3_2(output)
        output = self.pool3(output)

        output = self.conv4_1(output)
        output = self.conv4_2(output)
        output = self.pool4(output)

        output = self.conv5(output)
        output = self.conv6(output)
        output = self.conv7(output)
        
        prediction = self.pred(output)

        B, abC, H, W = prediction.size()

        # [B, anchor_n * C, N, M] -> [B, N, M, anchor_n * C] -> [B, N*M, anchor_n*C]
        prediction = prediction.permute(0, 2, 3, 1).contiguous().view(B, H*W, abC)

        # Divide prediction to conf_pred, txtytwth_pred and cls_pred   
        # [B, H*W*anchor_n, 1]
        conf_pred = prediction[:, :, :1 * self.anchor_number].contiguous().view(B, H*W*self.anchor_number, 1)
        # [B, H*W, anchor_n, num_cls]
        cls_pred = prediction[:, :, 1 * self.anchor_number : (1 + self.num_classes) * self.anchor_number].contiguous().view(B, H*W*self.anchor_number, self.num_classes)
        # [B, H*W, anchor_n, 4]
        txtytwth_pred = prediction[:, :, (1 + self.num_classes) * self.anchor_number:].contiguous()
        
        # test
        if not self.trainable:
            txtytwth_pred = txtytwth_pred.view(B, H*W, self.anchor_number, 4)
            with torch.no_grad():
                # batch size = 1                
                all_obj = torch.sigmoid(conf_pred)[0]           # 0 is because that these is only 1 batch.
                all_bbox = torch.clamp((self.decode_boxes(txtytwth_pred) / self.scale_torch)[0], 0., 1.)
                all_class = (torch.softmax(cls_pred[0, :, :], 1) * all_obj)
                # separate box pred and class conf
                all_obj = all_obj.to('cpu').numpy()
                all_class = all_class.to('cpu').numpy()
                all_bbox = all_bbox.to('cpu').numpy()

                bboxes, scores, cls_inds = self.postprocess(all_bbox, all_class)

                return bboxes, scores, cls_inds

        else:
            txtytwth_pred = txtytwth_pred.view(B, H*W, self.anchor_number, 4)
            # decode bbox, and remember to cancel its grad since we set iou as the label of objectness.
            with torch.no_grad():
                x1y1x2y2_pred = (self.decode_boxes(txtytwth_pred) / self.scale_torch).view(-1, 4)

            txtytwth_pred = txtytwth_pred.view(B, H*W*self.anchor_number, 4)

            x1y1x2y2_gt = target[:, :, 7:].view(-1, 4)

            # compute iou
            iou = tools.iou_score(x1y1x2y2_pred, x1y1x2y2_gt).view(B, H*W*self.anchor_number, 1)

            # we set iou between pred bbox and gt bbox as conf label. 
            # [obj, cls, txtytwth, x1y1x2y2] -> [conf, obj, cls, txtytwth]
            target = torch.cat([iou, target[:, :, :7]], dim=2)

            conf_loss, cls_loss, txtytwth_loss, total_loss = tools.loss(pred_conf=conf_pred, pred_cls=cls_pred,
                                                                        pred_txtytwth=txtytwth_pred,
                                                                        label=target,
                                                                        num_classes=self.num_classes)

            return conf_loss, cls_loss, txtytwth_loss, total_loss