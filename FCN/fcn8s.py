'''
Descripttion: Learn FCN
version: 1.0
Author: SongJ
Date: 2021-07-28 20:12:15
LastEditors: SongJ
LastEditTime: 2021-08-08 09:42:18
'''

import numpy as np
import paddle
import paddle.fluid as fluid
from paddle.fluid.dygraph import to_variable
from paddle.fluid.dygraph import Conv2D
from paddle.fluid.dygraph import Conv2DTranspose
from paddle.fluid.dygraph import Dropout
from paddle.fluid.dygraph import BatchNorm
from paddle.fluid.dygraph import Pool2D
from paddle.fluid.dygraph import Linear
from vgg import VGG16BN


class FCN8s(fluid.dygraph.Layer):
    def __init__(self, num_classes=59):
        super(FCN8s, self).__init__()
        backbone = VGG16BN(pretrained=False)

        self.layer1 = backbone.layer1
        self.layer1[0].conv._padding=[100, 100]
        self.pool1 = Pool2D(pool_size=2, pool_stride=2, ceil_mode=True)
        self.layer2 = backbone.layer2
        self.pool2 = Pool2D(pool_size=2, pool_stride=2, ceil_mode=True)
        self.layer3 = backbone.layer3
        self.pool3 = Pool2D(pool_size=2, pool_stride=2, ceil_mode=True)
        self.layer4 = backbone.layer4
        self.pool4 = Pool2D(pool_size=2, pool_stride=2, ceil_mode=True)
        self.layer5 = backbone.layer5
        self.pool5 = Pool2D(pool_size=2, pool_stride=2, ceil_mode=True)


        self.fc6 = Conv2D(512, 4096, 7, act='relu')
        self.fc7 = Conv2D(4096, 4096, 1, act='relu')
        self.drop6 = Dropout()
        self.drop7 = Dropout()
        
        self.score = Conv2D(4096, num_classes, 1)
        self.score_pool3 = Conv2D(256, num_classes, 1)
        self.score_pool4 = Conv2D(512, num_classes, 1)
        
        # 图中FC7的采样 之后尺寸变为1/16
        self.up_output = Conv2DTranspose(num_channels=num_classes,
                                        num_filters=num_classes,
                                        filter_size=4,
                                        stride=2, 
                                        bias_attr=False)
        
        # 图中
        self.up_pool4 = Conv2DTranspose(num_channels=num_classes,
                                        num_filters=num_classes,
                                        filter_size=4,
                                        stride=2, 
                                        bias_attr=False)
        
        # 8倍上采样
        self.up_final = Conv2DTranspose(num_channels=num_classes,
                                        num_filters=num_classes,
                                        filter_size=16,
                                        stride=8, 
                                        bias_attr=False)

    def forward(self, inputs):
        x = self.layer1(inputs)
        x = self.pool1(x)   # 1/2
        x = self.layer2(x) 
        x = self.pool2(x)   # 1/4
        x = self.layer3(x)
        x = self.pool3(x)   # 1/8
        pool3 = x
        x = self.layer4(x)
        x = self.pool4(x)   # 1/16
        pool4 = x
        x = self.layer5(x) 
        x = self.pool5(x)   # 1/32

        x = self.fc6(x)
        x = self.drop6(x)
        x = self.fc7(x)
        x = self.drop7(x)

        # 图中FC7的第一次拼接
        x = self.score(x) # 对应网络结构的FC7 upscore
        x = self.up_output(x) # 上采样
        up_output = x  # 1/16  
        x = self.score_pool4(pool4) # 获取pool4
        x = x[:, :, 5:5+up_output.shape[2], 5:5+up_output.shape[3]] # 调整尺寸,来源于论文
        up_pool4 = x
        x = up_pool4 + up_output # 拼接

        # 图中的第二次拼接
        x = self.up_pool4(x)
        up_output = x  # 1/8
        x = self.score_pool3(pool3)
        x = x[:, :, 9:9+up_output.shape[2], 9:9+up_output.shape[3]] # 调整尺寸,来源于论文
        up_pool3 = x
        x = up_pool3+up_output

        x = self.up_final(x)
        x = x[:, :, 31:31+inputs.shape[2], 31:31+inputs.shape[3]] # 调整尺寸,来源于论文


        return x     

def main():

    with fluid.dygraph.guard():
        x_data = np.random.rand(2, 3, 512, 512).astype(np.float32)
        x = to_variable(x_data)
        model =FCN8s(num_classes=59)
        model.eval()
        pred = model(x)
        print(pred.shape)


if __name__=='__main__':
    main()

