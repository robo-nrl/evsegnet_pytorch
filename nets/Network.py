from tokenize import group
from PIL import Image
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy
import matplotlib.pyplot as plt
import os
import argparse
import json
from requests.exceptions import ConnectionError, ReadTimeout, TooManyRedirects, MissingSchema, InvalidURL
import logging
import random
import glob
import torchvision
import torchvision.transforms as tvt
import ast
import requests
import numpy as np
from torch.utils.data import DataLoader, Dataset
import pdb

#TO ENSURE CODE REPRODUCIBILITY
seed = 0
random.seed(seed)
torch.manual_seed(seed)
numpy.random.seed(seed)
torch.backends.cudnn.deterministic=True
torch.backends.cudnn.benchmarks=False
os.environ['PYTHONHASHSEED'] = str(seed)

#START XCEPTION
import math
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.model_zoo as model_zoo
from torch.nn import init
import torch

__all__ = ['xception']

model_urls = {
    'xception':'https://www.dropbox.com/s/1hplpzet9d7dv29/xception-c0a72b38.pth.tar?dl=1'
}

class SeparableConv2d(nn.Module):
    def __init__(self,in_channels,out_channels,kernel_size=1,stride=1,padding=0,dilation=1,bias=False):
        super(SeparableConv2d,self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding= padding
        self.dilation =dilation
        self.bias = bias

        self.conv1 = nn.Conv2d(in_channels=in_channels,out_channels=in_channels, kernel_size=kernel_size,stride=stride,padding=padding,dilation=dilation,groups=in_channels,bias=bias)
        self.pointwise = nn.Conv2d(in_channels,out_channels,1,1,0,1,1,bias=bias)
        
 
    def forward(self,x):
        #print("shape1", x.shape)
        x = self.conv1(x)
        x = self.pointwise(x)
        return x

class Block(nn.Module):
    def __init__(self,in_filters,out_filters,reps,stride=1,start_with_relu=True,grow_first=True):
        super(Block, self).__init__()

        self.in_filters = in_filters
        self.out_filters = out_filters
        self.reps = reps
        self.stride = stride
        self.start_with_relu = start_with_relu
        self.grow_first = grow_first

        if out_filters != in_filters or stride!=1:
            self.skip = nn.Conv2d(in_filters,out_filters,1,stride=stride, bias=False)
            self.skipbn = nn.BatchNorm2d(out_filters)
        else:
            self.skip=None
        
        self.relu = nn.ReLU(inplace=True)
        rep=[]

        filters=in_filters
        if grow_first:
            rep.append(self.relu)
            rep.append(SeparableConv2d(in_filters,out_filters,3,stride=1,padding=1,bias=False))
        
            rep.append(nn.BatchNorm2d(out_filters))
            filters = out_filters

        for i in range(reps-1):
            rep.append(self.relu)
            rep.append(SeparableConv2d(filters,filters,3,stride=1,padding=1,bias=False))
            #if block_no==2:  
            rep.append(nn.BatchNorm2d(filters))
        
        if not grow_first:
            rep.append(self.relu)
            rep.append(SeparableConv2d(in_filters,out_filters,3,stride=1,padding=1,bias=False))
            rep.append(nn.BatchNorm2d(out_filters))

        if not start_with_relu:
            rep = rep[1:]
        else:
            rep[0] = nn.ReLU(inplace=False)

        if stride != 1:        
            rep.append(nn.MaxPool2d(3,stride,1))
        self.rep = nn.Sequential(*rep)

    def forward(self,inp):
        #print("rep inp:", inp.shape)
        x = self.rep(inp)

        if self.skip is not None:
            skip = self.skip(inp)
            skip = self.skipbn(skip)
        else:
            skip = inp

        x+=skip
        return x

features={}
def get_features(name):
    def hook(model, input, output):
        features[name] = output.detach()
    return hook

class Xception(nn.Module):
    """
    Xception optimized for the ImageNet dataset, as specified in
    https://arxiv.org/pdf/1610.02357.pdf
    """
    def __init__(self, num_classes):
        """ Constructor
        Args:
            num_classes: number of classes
        """
        super(Xception, self).__init__()

        
        self.num_classes = num_classes

        self.conv1 = nn.Conv2d(6, 32, 3,2, 0, bias=False)
        self.bn1 = nn.BatchNorm2d(32)
        self.relu = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(32,64,3,bias=False)
        self.bn2 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)

        self.block1=Block(64,128,2,2,start_with_relu=False,grow_first=True)
        self.block2=Block(128,256,2,2,start_with_relu=True,grow_first=True)
        self.block3=Block(256,728,2,2,start_with_relu=True,grow_first=True)

        self.block4=Block(728,728,3,1,start_with_relu=True,grow_first=True)
        self.block5=Block(728,728,3,1,start_with_relu=True,grow_first=True)
        self.block6=Block(728,728,3,1,start_with_relu=True,grow_first=True)
        self.block7=Block(728,728,3,1,start_with_relu=True,grow_first=True)

        self.block8=Block(728,728,3,1,start_with_relu=True,grow_first=True)
        self.block9=Block(728,728,3,1,start_with_relu=True,grow_first=True)
        self.block10=Block(728,728,3,1,start_with_relu=True,grow_first=True)
        self.block11=Block(728,728,3,1,start_with_relu=True,grow_first=True)

        self.block12=Block(728,728,2,2,start_with_relu=True,grow_first=False)

        #block13
        self.conv3 = nn.Conv2d(728,1024,(3,3), stride=(2, 2),padding=1)
        self.bn3 = nn.BatchNorm2d(1024)
        self.conv4 = SeparableConv2d(728,728,(3, 3), padding=1)
        self.bn4 = nn.BatchNorm2d(728)
        self.conv5 = SeparableConv2d(728,1024,(3, 3), padding=1)
        self.bn5 = nn.BatchNorm2d(1024)

        self.relu = nn.ReLU(inplace=True)

        #block14
        self.conv6 = SeparableConv2d(1024,1536,3,1,1)
        self.bn6 = nn.BatchNorm2d(1536)
        self.conv7 = SeparableConv2d(1536,2048,3,1,1)
        self.bn7 = nn.BatchNorm2d(2048)

        self.fc = nn.Linear(2048, num_classes)



        #------- init weights --------
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
        #-----------------------------


    def forward(self, x):
        #print(x.shape)

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.block5(x)
        x = self.block6(x)
        x = self.block7(x)
        x = self.block8(x)
        x = self.block9(x)
        x = self.block10(x)
        x = self.block11(x)
        x = self.block12(x)
        
        residual = self.conv3(x)
        residual = self.bn3(residual)
  
        x = self.relu(x)
        x = self.conv4(x)
        x = self.bn4(x)
        x = self.relu(x)
        x = self.conv5(x)
        x = self.bn5(x)
        m = nn.MaxPool2d(3,(2,2),1)
        x=m(x)
        x = x+residual

        x = self.conv6(x)
        x = self.bn6(x)
        x = self.relu(x)
        x = self.conv7(x)
        x = self.bn7(x)
        x = self.relu(x)

        x = F.adaptive_avg_pool2d(x, (1, 1))
        #x = a(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x




def xception_out(inputs, pretrained=False):
    """
    Construct Xception.
    """

    model = Xception(num_classes = 6)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['xception']))
    
    features_0= model.block2.rep[5].register_forward_hook(get_features('block2_sepconv2_bn'))
    features_1= model.block3.rep[5].register_forward_hook(get_features('block3_sepconv2_bn'))
    features_2= model.block4.rep[4].register_forward_hook(get_features('block4_sepconv2_bn'))
    features_3= model.bn5.register_forward_hook(get_features('block13_sepconv2_bn'))
    features_4= model.bn7.register_forward_hook(get_features('block14_sepconv2_bn'))

    outputs=model(inputs)

    #feats = [features_0, features_1, features_2, features_3, features_4]

    feats = []

    feats.append(features['block2_sepconv2_bn'])
    feats.append(features['block3_sepconv2_bn'])
    feats.append(features['block4_sepconv2_bn'])
    feats.append(features['block13_sepconv2_bn'])
    feats.append(features['block14_sepconv2_bn'])

    features_0.remove()
    features_1.remove()
    features_2.remove()
    features_3.remove()
    features_4.remove()

    return feats


#END XCEPTION
        

#Network_Model
class Segception_small(nn.Module):
    def __init__(self, num_classes=6, input_shape=(3, None, None), weights='imagenet'):
        super(Segception_small, self).__init__()
        self.num_classes=num_classes
        #note: filters=out_channels
        self.adap_encoder_1 = EncoderAdaption(in_channels=256, out_channels=256, kernel_size=3, dilation=1)
        self.adap_encoder_2 = EncoderAdaption(in_channels=728, out_channels=256, kernel_size=3, dilation=1)
        self.adap_encoder_3 = EncoderAdaption(in_channels=728, out_channels=128, kernel_size=3, dilation=1)
        self.adap_encoder_4 = EncoderAdaption(in_channels=1024, out_channels=64, kernel_size=3, dilation=1)
        self.adap_encoder_5 = EncoderAdaption(in_channels=2048, out_channels=32, kernel_size=3, dilation=1)

        self.decoder_conv_1 = FeatureGeneration(in_channels=256, out_channels=128, kernel_size=3, dilation=1, blocks=3)
        self.decoder_conv_2 = FeatureGeneration(in_channels=128, out_channels=64, kernel_size=3, dilation=1, blocks=3)
        self.decoder_conv_3 = FeatureGeneration(in_channels=64, out_channels=32, kernel_size=3, dilation=1, blocks=3)
        self.decoder_conv_4 = FeatureGeneration(in_channels=32, out_channels=32, kernel_size=3, dilation=1, blocks=1)
        self.aspp = ASPP_2(in_channels=32, out_channels=32, kernel_size=3)

        self.conv_logits = conv(in_channels= 32, out_channels=num_classes, kernel_size=1, stride=1, bias=True)

    def forward(self, inputs, mask=None, aux_loss=False):
        print(inputs.shape)
        #inputs = np.transpose(inputs, (0,3,2,1))
        #print(inputs.shape)
        #print('here:', inputs.shape)

        outputs = xception_out(inputs)
        # add activations to the ourputs of the model
        for i in range(len(outputs)):
            outputs[i] = nn.LeakyReLU(negative_slope=0.3)(outputs[i])
            
        print("output[0].shape:", outputs[0].shape)
        print("output[1].shape:", outputs[1].shape)
        print("output[2].shape:", outputs[2].shape)
        print("output[3].shape:", outputs[3].shape)
        print("output[4].shape:", outputs[4].shape)

        #pdb.set_trace()

        x = self.adap_encoder_1(outputs[0]#, training=training
        )
        #print("x.shape:", x.shape)
        x = upsampling(x, scale=2)
        x += reshape_into(self.adap_encoder_2(outputs[1]#, training=training
        ), x)  # 512
        x = self.decoder_conv_1(x
        #, training=training
        )  # 256

        #print("here:", x.shape)

        x = upsampling(x, scale=2)
        #print("here:", x.shape)
        x += reshape_into(self.adap_encoder_3(outputs[2]), x)  # 256
        x = self.decoder_conv_2(x
        #, training=training
        )  # 256

        x = upsampling(x, scale=2)
        x += reshape_into(self.adap_encoder_4(outputs[3]#, training=training
        ), x)  # 128
        x = self.decoder_conv_3(x#, training=training
        )  # 128

        x = self.aspp(x, #training=training, 
        operation='sum')  # 128

        x = upsampling(x, scale=2)
        x += reshape_into(self.adap_encoder_5(outputs[4], #training=training
        ), x)  # 64
        x = self.decoder_conv_4(x#, training=training
        )  # 64
        x = self.conv_logits(x)
        x = upsampling(x, scale=2)

        if aux_loss:
            return x, x
        else:
            return x

class EncoderAdaption(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, dilation=1):
        super(EncoderAdaption, self).__init__()

        self.in_channel=in_channels
        #x = self.conv(x)
        #return x

        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.dilation =dilation

        self.conv1 = Conv_BN(in_channels, out_channels, kernel_size=1)
        self.conv2 = ShatheBlock(out_channels, out_channels, kernel_size=kernel_size, dilation=dilation)

    def forward(self, inputs#, training=None
    ):
        x = self.conv1(inputs#, training=training
        )
        x = self.conv2(x#, training=training
        )
        return x


class FeatureGeneration(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, dilation=1,  blocks=3):
        super(FeatureGeneration, self).__init__()

        self.in_channel=in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.blocks = blocks
        self.dilation =dilation

        self.conv0 = Conv_BN( in_channels, out_channels, kernel_size=1)
        self.blocks = []
        for n in range(blocks):
            self.blocks = self.blocks + [
                ShatheBlock(out_channels, self.out_channels, kernel_size=kernel_size, dilation=dilation)]

    def forward(self, inputs#, training=None
    ):

        x = self.conv0(inputs#, training=training
        )
        for block in self.blocks:
            x = block(x#, training=training
            )

        return x

class ShatheBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size,  dilation=1, bottleneck=2):
        super(ShatheBlock, self).__init__()
        self.in_channel=in_channels
        self.bottleneck = bottleneck
        self.out_channels = out_channels * bottleneck
        self.kernel_size = kernel_size

        #print("shathe:",out_channels)
        #print("self:",ch)

        self.conv = DepthwiseConv_BN(in_channels, self.out_channels, kernel_size=kernel_size, dilation=dilation)
        self.conv1 = DepthwiseConv_BN(in_channels= self.out_channels, out_channels=self.out_channels, kernel_size=kernel_size, dilation=dilation)
        self.conv2 = DepthwiseConv_BN(self.out_channels, self.out_channels, kernel_size=kernel_size, dilation=dilation)
        self.conv3 = Conv_BN( self.out_channels, out_channels, kernel_size=1)

    def forward(self, inputs#, training=None
    ):
        x = self.conv(inputs#, training=training
        )
        x = self.conv1(x#, training=training
        )
        x = self.conv2(x#, training=training
        )
        x = self.conv3(x#, training=training
        )
        return x + inputs

class Conv_BN(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1):
        super(Conv_BN, self).__init__()

        self.in_channel=in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride

        self.conv = conv(in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride)
        self.bn = nn.BatchNorm2d(out_channels, eps=1e-3, momentum=0.993)

    def forward(self, inputs,# training=None,
    activation=True):
        import pdb
        #pdb.set_trace()
        x = self.conv(inputs)
        x = self.bn(x#, training=training
        )
        if activation:
            x = nn.LeakyReLU(negative_slope=0.3)(x)

        return x

class DepthwiseConv_BN(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, dilation=1):
        super(DepthwiseConv_BN, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.dilation =dilation

        #print("dep:",out_channels)

        self.conv = separableConv(in_channels, out_channels, kernel_size=kernel_size, stride=stride,dilation=dilation)
        self.bn = nn.BatchNorm2d(num_features=out_channels, eps=1e-3, momentum=0.993)

    def forward(self, inputs#, training=None
    ):
        x = self.conv(inputs)
        x = self.bn(x#, training=training
        )
        x = nn.LeakyReLU(negative_slope=0.3)(x)

        return x

class ASPP_2(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size):
        super(ASPP_2, self).__init__()
        self.in_channel=in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size

        self.conv1 = DepthwiseConv_BN(in_channels, out_channels, kernel_size=1, dilation=1)
        self.conv2 = DepthwiseConv_BN(out_channels, out_channels,kernel_size=kernel_size, dilation=4)
        self.conv3 = DepthwiseConv_BN(out_channels, out_channels,kernel_size=kernel_size, dilation=8)
        self.conv4 = DepthwiseConv_BN(out_channels, out_channels,kernel_size=kernel_size, dilation=16)
        self.conv6 = DepthwiseConv_BN(out_channels, out_channels, kernel_size=kernel_size, dilation=(2, 8))
        self.conv7 = DepthwiseConv_BN(out_channels, out_channels, kernel_size=kernel_size, dilation=(6, 3))
        self.conv8 = DepthwiseConv_BN(out_channels, out_channels, kernel_size=kernel_size, dilation=(8, 2))
        self.conv9 = DepthwiseConv_BN(out_channels, out_channels, kernel_size=kernel_size, dilation=(3, 6))
        self.conv5 = Conv_BN(out_channels, out_channels, kernel_size=1)

    def forward(self, inputs,# training=None, 
    operation='concat'):
        feature_map_size = inputs.shape
        image_features = torch.mean(inputs, [2, 3], True)
        #pdb.set_trace()
        image_features = self.conv1(image_features#, training=training
        )
        image_features=F.interpolate(image_features, size=(feature_map_size[2], feature_map_size[3]), mode='bilinear')
        x1 = self.conv2(inputs#, training=training
        )
        x2 = self.conv3(inputs#, training=training
        )
        x3 = self.conv4(inputs#, training=training
        )
        x4 = self.conv6(inputs#, training=training
        )
        x5 = self.conv7(inputs#, training=training
        )
        x4 = self.conv8(inputs#, training=training
        ) + x4
        x5 = self.conv9(inputs#, training=training
        ) + x5
        if 'concat' in operation:
            x = self.conv5(torch.cat((image_features, x1, x2, x3,x4,x5, inputs), 3)#, training=training
            )
        else:
            x = self.conv5(image_features + x1 + x2 + x3+x5+x4#, training=training
            ) + inputs

        return x

def upsampling(inputs, scale):
    return F.interpolate(inputs, scale_factor =2,  mode = 'bilinear', align_corners=True)


def reshape_into(inputs, input_to_copy):
    return F.interpolate(inputs, size=[input_to_copy.shape[2], input_to_copy.shape[3]], mode = 'bilinear', align_corners=True)


# convolution
def conv(in_channels, out_channels, kernel_size, stride=1, dilation=1, bias=False):
    return nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding_mode='zeros', bias=bias, dilation=dilation)



class separableConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, dilation=1, bias=False):
        super(separableConv, self).__init__()
        self.in_channel=in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.dilation =dilation
        self.stride = stride
        self.bias = bias

        #print("hi:",out_channels)

        self.depthwise = nn.Conv2d(in_channels, in_channels, kernel_size=kernel_size, stride=stride, groups=in_channels, bias=bias, padding='same')
        self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=bias, padding='same')




    def forward(self, x):
        import pdb
        #pdb. set_trace()
        x = self.depthwise(x)
        x = self.pointwise(x)
        return x

