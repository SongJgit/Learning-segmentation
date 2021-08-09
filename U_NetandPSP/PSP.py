'''
Descripttion: Learn PSP
version: 1.0
Author: SongJ
Date: 2021-07-29 19:43:47
LastEditors: SongJ
LastEditTime: 2021-08-09 11:16:20
'''

import numpy as np
import paddle.fluid as fluid
import paddle
from paddle.fluid.dygraph import to_variable
from paddle.fluid.dygraph import Layer
from paddle.fluid.dygraph import Conv2D
from paddle.fluid.dygraph import BatchNorm
from paddle.fluid.dygraph import Pool2D
from paddle.fluid.dygraph import Conv2DTranspose
from paddle.fluid.dygraph import Dropout
from resnet_dilated import ResNet50

class PSPModule(Layer):
    def __init__(self, num_channels,bin_size_list):
        super(PSPModule, self).__init__()
        self.bin_size_list = bin_size_list
        num_filters =num_channels // len(bin_size_list) # C/4
        self.features = []
        for i in range(len(bin_size_list)):
            self.features.append(
                fluid.dygraph.Sequential(
                    Conv2D(num_channels,num_filters,1),
                    BatchNorm(num_filters,act='relu')
                )
            )
        
    def forward(self, inputs):
        out = [inputs]
        for idx, f in enumerate(self.features):
            # 输入变成1,2,3,6、2048C的feature map
            x = fluid.layers.adaptive_pool2d(inputs,self.bin_size_list[idx])
            x = f(x)
            x = fluid.layers.interpolate(x,inputs.shape[2::], align_corners=True)
            out.append(x)
        out = fluid.layers.concat(out, axis=1)

        return out




class PSPNet(Layer):
    def __init__(self, num_classes=59, backbone='resnet50'):
        super(PSPNet, self).__init__()
        res = ResNet50(pretrained=False)
        self.layer0 = fluid.dygraph.Sequential(
            res.conv,
            res.pool2d_max
        )
        self.layer1 = res.layer1
        self.layer2 = res.layer2
        self.layer3 = res.layer3
        self.layer4 = res.layer4

        # pspmodule,2048 → 2048*2对应图的输入和输出C
        num_channels = 2048
        self.pspmodule = PSPModule(num_channels,[1,2,3,6])
        num_channels *=2

        # cls: 2048*2 → 512 → cnum_classes
        self.classifier = fluid.dygraph.Sequential(
            Conv2D(num_channels=num_channels, num_filters=512, filter_size=3, padding=1),
            BatchNorm(512,act='relu'),
            Dropout(0,1),
            Conv2D(num_channels=512, num_filters=num_classes, filter_size=1)
        )

        # aux

    def forward(self,inputs):
        x = self.layer0(inputs)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.pspmodule(x)
        x = self.classifier(x)

        x= fluid.layers.interpolate(x,inputs.shape[2::],align_corners=True)

        return x,1
def main():
    with fluid.dygraph.guard(fluid.CPUPlace()):
        model = PSPNet(num_classes=59)
        x_data = np.random.rand(1,3,473,473).astype(np.float32)
        model.train()
        inputs = to_variable(x_data)
        pred,aux = model(inputs)
        
        print(pred.shape,aux.shape)

if __name__ =='__main__': 
    main()