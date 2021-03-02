import torch
import torch.nn as nn
import warnings
warnings.filterwarnings("ignore")
from einops import rearrange

def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super(BasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError('BasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    # Bottleneck in torchvision places the stride for downsampling at 3x3 convolution(self.conv2)
    # while original implementation places the stride at the first 1x1 convolution(self.conv1)
    # according to "Deep residual learning for image recognition"https://arxiv.org/abs/1512.03385.
    # This variant is also known as ResNet V1.5 and improves accuracy according to
    # https://ngc.nvidia.com/catalog/model-scripts/nvidia:resnet_50_v1_5_for_pytorch.

    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super(Bottleneck, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.)) * groups
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv1x1(inplanes, width)
        self.bn1 = norm_layer(width)
        self.conv2 = conv3x3(width, width, stride, groups, dilation)
        self.bn2 = norm_layer(width)
        self.conv3 = conv1x1(width, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.shortcut = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.shortcut is not None:
            identity = self.shortcut(x)

        out += identity
        out = self.relu(out)

        return out


class ResNet(nn.Module):

    def __init__(self, block, layers, num_classes=1000, zero_init_residual=False,
                 groups=1, width_per_group=64, replace_stride_with_dilation=None,
                 norm_layer=None):
        super(ResNet, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

        self.inplanes = 64
        self.dilation = 1
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError("replace_stride_with_dilation should be None "
                             "or a 3-element tuple, got {}".format(replace_stride_with_dilation))
        self.groups = groups
        self.base_width = width_per_group
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)



        self.block1 = self._make_layer(block, 64, layers[0],stride =1)
        self.pool = self.__sample(block,64,1,stride=2)

        self.block2 = self._make_layer(block, 128, layers[1], stride=2,
                                       dilate=True)
        self.block3 = self._make_layer(block, 256, layers[2], stride=2,
                                       dilate=True)
        
        self.squeeze = nn.Sequential(conv3x3(1024,256),nn.BatchNorm2d(256),nn.ReLU())

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block, planes, blocks, stride=1, dilate=False):
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        
        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, self.groups,
                            self.base_width,self.dilation, norm_layer))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation,
                                norm_layer=norm_layer))

        return nn.Sequential(*layers)

    def __sample(self, block, planes, blocks, stride=1, dilate=False):
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.MaxPool2d(kernel_size=2,stride=2,padding =0)

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, self.groups,
                            self.base_width, previous_dilation, norm_layer))

        return nn.Sequential(*layers)

    def _forward_impl(self, x):
        # See note [TorchScript super()]
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.block1(x)
        x = self.pool(x)

        x = self.block2(x)

        x = self.block3(x)
        x = self.squeeze(x)
        return x

    def forward(self, x):
        return self._forward_impl(x)


def _resnet(arch, block, layers, pretrained, progress, **kwargs):
    model = ResNet(block, layers, **kwargs)
    return model


def resnet18(pretrained=False, progress=True, **kwargs):
    r"""ResNet-18 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet('resnet18', BasicBlock, [2, 2, 2, 2], pretrained, progress,
                   **kwargs)


def resnet34(pretrained=False, progress=True, **kwargs):
    r"""ResNet-34 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet('resnet34', BasicBlock, [3, 4, 6, 3], pretrained, progress,
                   **kwargs)


def resnet50(pretrained=False, progress=True, **kwargs):
    r"""ResNet-50 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet('resnet50', Bottleneck, [2, 4, 6], pretrained, progress,
                   **kwargs)


def resnet101(pretrained=False, progress=True, **kwargs):
    r"""ResNet-101 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet('resnet101', Bottleneck, [3, 4, 23, 3], pretrained, progress,
                   **kwargs)


def resnet152(pretrained=False, progress=True, **kwargs):
    r"""ResNet-152 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet('resnet152', Bottleneck, [3, 8, 36, 3], pretrained, progress,
                   **kwargs)


def load_model_from_tensorflow():


    import tensorflow as tf
    from tensorflow.python import pywrap_tensorflow

    model_reader = pywrap_tensorflow.NewCheckpointReader("../weights/minimal_hand/model/detnet/detnet.ckpt")

    value = var_dict = model_reader.get_variable_to_shape_map()
    keys = value.keys()
    keys = sorted(keys)
    conv7x7 = map={
        'conv1.bn.bias':'bn1.bias',
        'conv1.bn.running_mean':'bn1.running_mean',
        'conv1.bn.running_var':'bn1.running_var',
        'conv1.bn.weight':'bn1.weight',
        'conv1.conv2d.kernel':'conv1.weight',
    }
    block_1_shortcut_map = {
        'block1.0.shortcut.conv2d.kernel':'block1.0.shortcut.0.weight',
        'block1.0.shortcut.bn.bias':'block1.0.shortcut.1.bias',
        'block1.0.shortcut.bn.weight':'block1.0.shortcut.1.weight',
        'block1.0.shortcut.bn.running_mean':'block1.0.shortcut.1.running_mean',
        'block1.0.shortcut.bn.running_var':'block1.0.shortcut.1.running_var'
    
    }

    block_2_shortcut_map = {
        'block2.0.shortcut.conv2d.kernel':'block2.0.shortcut.0.weight',
        'block2.0.shortcut.bn.bias':'block2.0.shortcut.1.bias',
        'block2.0.shortcut.bn.weight':'block2.0.shortcut.1.weight',
        'block2.0.shortcut.bn.running_mean':'block2.0.shortcut.1.running_mean',
        'block2.0.shortcut.bn.running_var':'block2.0.shortcut.1.running_var'
    }
    block_3_shortcut_map = {
        'block3.0.shortcut.conv2d.kernel':'block3.0.shortcut.0.weight',
        'block3.0.shortcut.bn.bias':'block3.0.shortcut.1.bias',
        'block3.0.shortcut.bn.weight':'block3.0.shortcut.1.weight',
        'block3.0.shortcut.bn.running_mean':'block3.0.shortcut.1.running_mean',
        'block3.0.shortcut.bn.running_var':'block3.0.shortcut.1.running_var'
    }
    squeeze_map = {
        'squeeze.conv2d.kernel':'squeeze.0.weight',
        'squeeze.bn.bias':'squeeze.1.bias',
        'squeeze.bn.running_mean' : 'squeeze.1.running_mean',
        'squeeze.bn.running_var':  'squeeze.1.running_var',
        'squeeze.bn.weight' : 'squeeze.1.weight',
    }


    import os 
    detnet_static_infos = {}
    
    for key in keys:
        if 'Adam' not in key:
            if 'resnet' not in key:
                continue
            
            transfer_key = key.split('resnet/')[-1]
            if 'unit' in transfer_key:
                transfer_key = transfer_key.replace('unit','').split('/')
                transfer_key[1] = str(int(transfer_key[1])-1)
                transfer_key = os.path.join(*transfer_key)
            
            transfer_key = transfer_key.replace('/','.')
            transfer_key = transfer_key.replace('moving_mean','running_mean').replace('moving_variance','running_var')
            transfer_key = transfer_key.replace('gamma','weight').replace('beta','bias').replace('batch_normalization','bn')

            #process conv7x7
            if transfer_key in  conv7x7:
                transfer_key = conv7x7[transfer_key]

            #process block_1
            if 'block1' in transfer_key:
                if transfer_key in block_1_shortcut_map:
                    transfer_key = block_1_shortcut_map[transfer_key]
                else:
                    if 'bn' in transfer_key:
                        key_element = transfer_key.split('.')
                        layer_index = key_element[2].split('conv')[-1]
                        key_element.pop(2)
                        key_element[2]+=layer_index
                        transfer_key = os.path.join(*key_element).replace('/','.')
                    else:
                        transfer_key = transfer_key.replace('.conv2d','').replace('kernel','weight')
                #transfer
                if 'block1.2' in transfer_key:
                    transfer_key = transfer_key.replace('block1.2','pool.0')
            elif 'block2' in transfer_key:
                if transfer_key in block_2_shortcut_map:
                    transfer_key = block_2_shortcut_map[transfer_key]
                else:
                    if 'bn' in transfer_key:
                        key_element = transfer_key.split('.')
                        layer_index = key_element[2].split('conv')[-1]
                        key_element.pop(2)
                        key_element[2]+=layer_index
                        transfer_key = os.path.join(*key_element).replace('/','.')
                    else:
                        transfer_key = transfer_key.replace('.conv2d','').replace('kernel','weight')
            elif 'block3' in transfer_key:
                if transfer_key in block_3_shortcut_map:
                    transfer_key = block_3_shortcut_map[transfer_key]
                else:
                    if 'bn' in transfer_key:
                        key_element = transfer_key.split('.')
                        layer_index = key_element[2].split('conv')[-1]
                        key_element.pop(2)
                        key_element[2]+=layer_index
                        transfer_key = os.path.join(*key_element).replace('/','.')
                    else:
                        transfer_key = transfer_key.replace('.conv2d','').replace('kernel','weight')
            elif 'squeeze' in transfer_key:
                transfer_key = squeeze_map[transfer_key]

            detnet_static_infos[transfer_key] = model_reader.get_tensor(key)

    return detnet_static_infos

if __name__ == '__main__':

    import numpy as np 
    inp  = np.load("./input.npy")
    output = np.load('./output.npy')
    output = torch.from_numpy(output)
    output = rearrange(output,'b h w c -> b c h w')




    inp = torch.from_numpy(inp)
    inp = rearrange(inp,'b h w c -> b c h w')

    model = resnet50()
    model_params = model.state_dict() 
    model  = model.eval()

    # r = model(inp)
    print(torch.sum(model(inp)-output))



    torch_keys = sorted(model_params.keys())  
    remain_torch_keys = []
    for key in torch_keys:
        if 'num_batches_tracked' not in key:
            remain_torch_keys.append(key)

    detnet_tensorflow = load_model_from_tensorflow()
    for key in detnet_tensorflow.keys():
        detnet_tensorflow[key] = torch.from_numpy(detnet_tensorflow[key])
        if len(detnet_tensorflow[key].shape)>1:
            detnet_tensorflow[key] = rearrange(detnet_tensorflow[key],"h w ic oc -> oc ic h w" )
    result = model.load_state_dict(detnet_tensorflow)

    torch.save(model.state_dict(),"./weights/resnet.pth")

    b,c,w,h = model(inp).shape
    loss  = model(inp)-output
    print(torch.max(loss),torch.sum(loss), torch.sum(loss)/(b*c*w*h))


    



