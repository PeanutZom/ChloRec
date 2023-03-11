import dataset
import model
from torch.utils.data import dataloader
import torch.nn as nn
import torch.optim as optim
import torch
import utils
import torchvision.transforms as transforms
import os
from torch.utils.tensorboard import SummaryWriter
import sys

train_areas = ["EastSea", "EastCoast", "Florida"]
# train_areas = ["EastCoast"]
val_areas = ["EastSea", "EastCoast", "Florida"]
test_areas = ["SouthSea"]

batch_size = 32
channel = 12

use_elevation = sys.argv[1].lower() == 'true'
use_refine = True
use_coor = sys.argv[2].lower() == 'true'

if use_elevation:
    name1 = "elevation_"
else:
    name1 = "no_elevation_"
    
if use_refine:
    name2 = "refine"
else:
    name2 = "single"
    
if use_coor:
    name3 = "coor_"
else:
    name3 = "no_coor_"
folder_name = name1 + name3 + name2


# 定义数据集的transform
ln_transform = transforms.Lambda(lambda y: torch.log(y))
nan_transform = transforms.Lambda(lambda y: torch.where(torch.isnan(y), torch.full_like(y, 0), y))
train_transform = transforms.Compose([ln_transform, nan_transform])
target_transform = transforms.Compose([ln_transform, nan_transform])

# 定义dataset和dataloader

train_datasets = []
train_dataloaders=[]

for train_area in train_areas:
    train_data_dir = os.path.join(os.path.dirname(os.getcwd()), 'data', train_area, 'train')
    train_dataset = dataset.DINCAEDataset(
        data_dir=train_data_dir, transform=train_transform, target_transform=target_transform, use_elevation=use_elevation, use_coor=use_coor)
    train_dataloader = dataloader.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True,
                                num_workers=2,
                                pin_memory=True)
    train_datasets.append(train_dataset)
    train_dataloaders.append(train_dataloader)


val_datasets = []
val_dataloaders = []

for val_area in val_areas:
    val_data_dir = os.path.join(os.path.dirname(os.getcwd()), 'data', val_area, 'val')
    val_dataset = dataset.DINCAEDataset(
        data_dir=val_data_dir, transform=train_transform, target_transform=target_transform, use_elevation=use_elevation, use_coor=use_coor)
    val_dataloader = dataloader.DataLoader(val_dataset, batch_size=batch_size, shuffle=False, drop_last=True,
                                num_workers=2,
                                pin_memory=True)
    val_datasets.append(val_dataset)
    val_dataloaders.append(val_dataloader)



mask_train_datasets = []
mask_train_dataloaders = []

for train_area in train_areas:
    mask_data_dir = os.path.join(os.path.dirname(os.getcwd()), 'data', train_area, 'mask')
    mask_train_dataset = dataset.MaskDataset(mask_dir=mask_data_dir)
    mask_train_dataloader = dataloader.DataLoader(mask_train_dataset, batch_size=batch_size, shuffle=True, drop_last=True,
                                num_workers=2,
                                pin_memory=True)
    mask_train_datasets.append(mask_train_dataset)
    mask_train_dataloaders.append(mask_train_dataloader)


    
mask_val_datasets = []
mask_val_dataloaders = []

for val_area in val_areas:
    mask_data_dir = os.path.join(os.path.dirname(os.getcwd()), 'data', val_area, 'mask')
    mask_val_dataset = dataset.MaskDataset(mask_dir=mask_data_dir)
    mask_val_dataloader = dataloader.DataLoader(mask_val_dataset, batch_size=batch_size, shuffle=False, drop_last=True,
                                num_workers=2,
                                pin_memory=True)
    mask_val_datasets.append(mask_val_dataset)
    mask_val_dataloaders.append(mask_val_dataloader)

# 定义网络
coarse_net = model.DINCAEModel(in_channels=channel)
refine_net = model.DINCAEModel(in_channels=channel+1)

criterion = nn.MSELoss()
optimizer = optim.Adam([{'params': coarse_net.parameters()}, {'params': refine_net.parameters()}])

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
coarse_net = coarse_net.to(device)
refine_net = refine_net.to(device)

print("device:", device)

# Tensorboard设置
writer = SummaryWriter("/root/tf-logs/" + folder_name + "/tensorboard")

# 开始训练
min_val_loss = 999.90

len_train = 0
len_val = 0

for train_dataloader in train_dataloaders:
    len_train += len(train_dataloader)
for val_dataloader in val_dataloaders:
    len_val += len(val_dataloaders)
    


for epoch in range(301):  # loop over the dataset multiple times

    running_loss = 0.0
    iters_train = [iter(train_dataloader) for train_dataloader in train_dataloaders]
    loaders_mask = [mask_train_dataloader for mask_train_dataloader in mask_train_dataloaders]
    # 对模型训练
    for i in range(len_train):
        data = None
        iter_mask = None
        while data is None:     
            try:
                iter_train = iters_train.pop(0)
                loader_mask = loaders_mask.pop(0)
                data = next(iter_train)
                iter_mask = iter(loader_mask)
                iters_train.append(iter_train)
                loaders_mask.append(loader_mask)
            except StopIteration:
                pass
        # get the inputs; data is a list of [inputs, labels]
        inputs, missing, labels = data[0], data[1], data[2]

        
        # 给输入施加一个mask,施加在三天的chlo-anomaly上
        mask1 = next(iter_mask)
        mask2 = next(iter_mask)
        mask3 = next(iter_mask)
        inputs[:, 0:1] *= mask1
        inputs[:, 1:2] *= mask2
        inputs[:, 2:3] *= mask3
        inputs[:, 9:10] *= mask1
        inputs[:, 10:11] *= mask2
        inputs[:, 11:12] *= mask3

        # 将tensor移动到gpu上
        inputs = inputs.to(device)
        labels = labels.to(device)
        missing = missing.to(device)

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = coarse_net(inputs.float())

        inputs_refine = torch.cat((inputs, outputs), 1)
        outputs_refine = refine_net(inputs_refine.float())
        
        # 损失函数不计算missed
        loss1 = criterion(outputs * missing, labels * missing)/missing.sum()*missing.numel()
        loss2 = criterion(outputs_refine * missing, labels * missing)/missing.sum()*missing.numel()
        loss = 0.3 * loss1 + 0.7 * loss2
        loss.backward()
        optimizer.step()
        
        # print statistics
        running_loss += loss2.item()
        if i == 19:
            writer.add_scalar('Training loss',
                              (running_loss/20)**0.5,
                              epoch + 1)
        if i % 20 == 19:    # print every 20 mini-batches
            print('[%d, %5d] loss: %.5f' %
                  (epoch + 1, i + 1, (running_loss/20)**0.5))
            running_loss = 0.0

    # 对模型进行validation
    if epoch % 5 == 4:
        val_epoch_loss = 0
        iters_val = [iter(val_dataloader) for val_dataloader in val_dataloaders]
        loaders_mask = [mask_val_dataloader for mask_val_dataloader in mask_val_dataloaders]
        for i in range(len_val):
            data = None
            iter_mask = None
            while data is None:     
                try:
                    iter_val = iters_val.pop(0)
                    loader_mask = loaders_mask.pop(0)
                    data = next(iter_val)
                    iter_mask = iter(loader_mask)
                    iters_val.append(iter_val)
                    loaders_mask.append(loader_mask)
                except:
                    pass
            # get the inputs; data is a list of [inputs, labels]
            inputs, missing, labels = data[0], data[1], data[2]
            
            # 给输入施加一个mask,施加在三天的chlo-anomaly上
            mask1 = next(iter_mask)
            mask2 = next(iter_mask)
            mask3 = next(iter_mask)
            inputs[:, 0:1] *= mask1
            inputs[:, 1:2] *= mask2
            inputs[:, 2:3] *= mask3
            inputs[:, 9:10] *= mask1
            inputs[:, 10:11] *= mask2
            inputs[:, 11:12] *= mask3
            # 将tensor移动到gpu上
            inputs = inputs.to(device)
            labels = labels.to(device)
            missing = missing.to(device)

            # forward + backward + optimize
            outputs = coarse_net(inputs.float())

            inputs_refine = torch.cat((inputs, outputs), 1)
            outputs_refine = refine_net(inputs_refine.float())

            # 损失函数不计算missed
            loss2 = criterion(outputs_refine * missing, labels * missing) / missing.sum() * missing.numel()

            # print statistics
            val_epoch_loss += loss2.item()

        val_loss = (val_epoch_loss/len_val) ** 0.5

        if val_loss < min_val_loss:
            min_val_loss = val_loss
            torch.save(coarse_net, "/root/tf-logs/" +folder_name + "/coarse_" + str(epoch) + '.pth')
            torch.save(refine_net, "/root/tf-logs/" +folder_name + "/refine_" + str(epoch) + '.pth')
        writer.add_scalar('Validation loss',
                          val_loss,
                          epoch + 1)
        print('[%d] loss: %.5f' %
              (epoch + 1, val_loss))





# i = 0
# utils.show(labels[i][0])
# utils.show(mask[i][0])
# utils.show(inputs[i][1])

# utils.show(outputs[i][0])
# utils.show(outputs_refine[i][0])

# a = (outputs_refine * missing - labels * missing)[i][0]
# utils.show(a)
# percent = a/labels[i][0]
# utils.show(percent)