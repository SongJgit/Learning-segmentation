'''
Descripttion: FCN train
version: 1.0
Author: SongJ
Date: 2021-07-29 15:38:53
LastEditors: SongJ
LastEditTime: 2021-07-29 15:38:54
'''
import os
import paddle
import paddle.fluid as fluid
from paddle.fluid.optimizer import AdamOptimizer
import numpy as np
import argparse
from utils import AverageMeter
from fcn8s import FCN8s
from basic_seg_loss import Basic_SegLoss
from basic_data_preprocessing import TrainAugmentation
from basic_dataloader import BasicDataLoader


parser = argparse.ArgumentParser()
parser.add_argument('--net', type=str, default='basic')
parser.add_argument('--lr', type=float, default=0.001)
parser.add_argument('--num_epochs', type=int, default=10)
parser.add_argument('--batch_size', type=int, default=4)
parser.add_argument('--image_folder', type=str, default='./FCN/dummy_data')
parser.add_argument('--image_list_file', type=str, default='./FCN/dummy_data/list.txt')
parser.add_argument('--checkpoint_folder', type=str, default='./FCN/output')
parser.add_argument('--save_freq', type=int, default=2)

args = parser.parse_args()


def train(dataloader, 
                model, 
                criterion,
                 optimizer, 
                 epoch,
                 total_batch):
    model.train()
    train_loss_meter = AverageMeter()
    for batch_id,data in enumerate(dataloader):
        image = data[0]
        label =data[1]
        image =fluid.layers.transpose(image, (0,3,1,2)) # NHWC to NCHW

        pred = model(image)
        loss = criterion(pred, label)
        loss.backward()
        optimizer.minimize(loss)
        model.clear_gradients()
        n = image.shape[0]

        train_loss_meter.update(loss.numpy()[0],n)
        print(f"Epoch[{epoch:03d}/{args.num_epochs:03d}], " +
        f"Step[{batch_id:04d}/{total_batch:04d}], " +
        f"Average Loss: {train_loss_meter.avg:4f}")

    return train_loss_meter.avg

def main():
    place = paddle.fluid.CUDAPlace(0)
    with fluid.dygraph.guard(place):
        basic_augmentation = TrainAugmentation(image_size=256)
        basic_dataloader = BasicDataLoader(image_folder=args.image_folder,
                                image_list_file=args.image_list_file,
                                transform=basic_augmentation,                                       
                                shuffle=True)

        train_dataloader = fluid.io.DataLoader.from_generator(capacity=10, use_multiprocess=True)
        train_dataloader.set_sample_generator(basic_dataloader,
                                            batch_size=args.batch_size,
                                            places=place)
        total_batch = int(len(basic_dataloader) / args.batch_size)

        if args.net == "basic":
            #TODO: create basicmodel
             #model = BasicModel()
             model = FCN8s()

        else:   
            raise NotImplementedError(f"args.net: {args.net} is not Supported!")

        
        criterion = Basic_SegLoss

        
        optimizer = AdamOptimizer(learning_rate=args.lr, parameter_list=model.parameters())

        
        for epoch in range(1, args.num_epochs+1):
            train_loss = train(train_dataloader,
                               model,
                               criterion,
                               optimizer,
                               epoch,
                               total_batch)
            print(f"----- Epoch[{epoch}/{args.num_epochs}] Train Loss: {train_loss:.4f}")

            if epoch % args.save_freq == 0 or epoch == args.num_epochs:
                model_path = os.path.join(args.checkpoint_folder, f"{args.net}-Epoch-{epoch}-Loss-{train_loss}")

                # TODO: save model and optmizer states
                model_dict = model.state_dict()
                fluid.save_dygraph(model_dict, model_path)
                optimizer_dict = optimizer.state_dict()
                fluid.save_dygraph(optimizer_dict, model_path)
                print(f'---- Save model: {model_path}.pdparams')
                print(f'---- Save optimizer: {model_path}.pdopt')


                print(f'----- Save model: {model_path}.pdparams')
                print(f'----- Save optimizer: {model_path}.pdopt')







if __name__ == "__main__":
    main()
