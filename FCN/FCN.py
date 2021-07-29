
import numpy as np
import paddle
import paddle.fluid as fluid
from paddle.fluid.dygraph import to_variable
from paddle.fluid.dygraph import Conv2D
from paddle.fluid.dygraph import Conv2DTranspose
from paddle.fluid.dygraph import Dropout
from paddle.fluid.dygraph import BatchNorm
from paddle.fluid.dygraph import pool2d
from paddle.fluid.dygraph import Linear
from vgg import VGG16BN


class FCN8s(fluid.dygraph.Layer):
    def __init__(self, num_classes=59):
        super(FCN8s, self).__init__()
        backbone = VGG16BN(pretrained=False)

        self.layer1 = backbone.layer1
        self.layer1[0].conv._padding=[100, 100]
        self.pool1 = pool2d(pool_size=2, pool_stride=2, ceil_mode=True)
        self.layer2 = backbone.layer2
        self.pool2 = pool2d(pool_size=2, pool_stride=2, ceil_mode=True)
        self.layer3 = backbone.layer3
        self.pool3 = pool2d(pool_size=2, pool_stride=2, ceil_mode=True)
        self.layer4 = backbone.layer4
        self.pool4 = pool2d(pool_size=2, pool_stride=2, ceil_mode=True)
        self.layer5 = backbone.layer5
        self.pool5 = pool2d(pool_size=2, pool_stride=2, ceil_mode=True)


        self.fc6 = Conv2D(512, 4096, 7, act='relu')
        self.fc7 = Conv2D(512, 4096, 1, act='relu')
        self.drop6 = Dropout()
        self.drop7 = Dropout()
        
    def forward(self, inputs):
        x = self.layer1(inputs)
        x = self.pool1(x)   # 1/2
        x = self.layer2(x) 
        x = self.pool2(x)   # 1/4
        x = self.layer3(x)
        x = self.pool3(x)   # 1/8
        x = self.layer4(x)
        x = self.pool4(x)   # 1/16
        x = self.layer5(x) 
        x = self.pool5(x)   # 1/32

        x = self.fc6(x)
        x = self.drop6(x)
        x = self.fc7(x)
        x = self.drop7(x)

        return x     

def main():
    with fluid.dygraph.guard():
        x_data = np.random.rand(2, 3, 512, 512).astype(np.float32)
        x = to_variable(x_data)
        model =FCN8s(num_classes=59)
        model.eval()
        pred = model(x)
        print(pred.shape)




