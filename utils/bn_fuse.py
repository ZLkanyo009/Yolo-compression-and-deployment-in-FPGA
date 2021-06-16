import torch
import torch.nn as nn

def fuse(self):
    # Fuse Conv2d + BatchNorm2d layers throughout model
    fused_list = nn.ModuleList()

    for a in list(self.children())[0]:
        if isinstance(a, nn.Sequential):
            for i, b in enumerate(a):
                if isinstance(b, nn.modules.batchnorm.BatchNorm2d):
                    # fuse this bn layer with the previous conv2d layer
                    conv = a[i - 1]
                    print(conv)
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
