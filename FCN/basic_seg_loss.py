import paddle
import paddle.fluid as fluid
import numpy as np
import cv2

eps = 1e-8

def Basic_SegLoss(preds, labels, ignore_index=255):
    n, c, h, w = preds.shape

    # TODO: create softmax_with_cross_entropy criterion
    criterion = fluid.layers.softmax_with_cross_entropy

    # TODO: transpose preds to NxHxWxC
    preds = fluid.layers.transpose(preds, (0,2,3,1))
    #print(preds.shape)
    
    mask = labels!=ignore_index
    mask = fluid.layers.cast(mask, 'float32')

    # TODO: call criterion and compute loss
    loss = criterion(preds, labels, axis=3)
    
    loss = loss * mask
    avg_loss = fluid.layers.mean(loss) / (fluid.layers.mean(mask) + eps)

    return avg_loss

def main():
    label = cv2.imread('work/dummy_data/GroundTruth_trainval_png/2008_000007.png')
    #print("label.shape = ", label.shape) [375,500,3]
    label = cv2.cvtColor(label, cv2.COLOR_BGR2GRAY).astype(np.int64) #转成了单通道
    pred = np.random.uniform(0, 1, (1, 59, label.shape[0], label.shape[1])).astype(np.float32)
    label = label[:,:,np.newaxis]
    label = label[np.newaxis, :, :, :]

    with fluid.dygraph.guard(fluid.CPUPlace()):
        pred = fluid.dygraph.to_variable(pred)
        label = fluid.dygraph.to_variable(label)
        loss = Basic_SegLoss(pred, label)
        print(pred.shape)
        print(label.shape)
        
        print(loss)

if __name__ == "__main__":
    main()

