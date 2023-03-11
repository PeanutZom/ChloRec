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


train_data_dir = os.path.join(os.path.dirname(os.getcwd()), 'data', 'ECS', 'train')
val_data_dir = os.path.join(os.path.dirname(os.getcwd()), 'data', 'ECS', 'val')
test_data_dir = os.path.join(os.path.dirname(os.getcwd()), 'data', 'ECS', 'test')
mask_data_dir = os.path.join(os.path.dirname(os.getcwd()), 'data', 'ECS', 'mask')

batch_size = 32
channel = 9
training_mean = -2.2366745
training_std = 0.81302816
use_elevation = True
use_refine = False
if use_elevation:
    name1 = "elevation_"
else:
    name1 = "no_elevation_"
if use_refine:
    name2 = "refine"
else:
    name2 = "single"
folder_name = name1 + name2


# 定义数据集的transform
ln_transform = transforms.Lambda(lambda y: torch.log(y))
nan_transform = transforms.Lambda(lambda y: torch.where(torch.isnan(y), torch.full_like(y, 0), y))
train_transform = transforms.Compose([ln_transform, transforms.Normalize(training_mean, training_std), nan_transform])
target_transform = transforms.Compose([ln_transform, transforms.Normalize(training_mean, training_std), nan_transform])

# 定义dataset和dataloader
train_dataset = dataset.DINCAEDataset(
    data_dir=train_data_dir, transform=train_transform, target_transform=target_transform, use_elevation=use_elevation)
train_dataloader = dataloader.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True,
                            num_workers=2,
                            pin_memory=True)

val_dataset = dataset.DINCAEDataset(
    data_dir=val_data_dir, transform=train_transform, target_transform=target_transform, use_elevation=use_elevation)
val_dataloader = dataloader.DataLoader(val_dataset, batch_size=batch_size, shuffle=False, drop_last=True,
                            num_workers=2,
                            pin_memory=True)

test_dataset = dataset.DINCAEDataset(
    data_dir=test_data_dir, transform=train_transform, target_transform=target_transform, use_elevation=use_elevation)
test_dataloader = dataloader.DataLoader(test_dataset, batch_size=batch_size, shuffle=False, drop_last=True,
                            num_workers=2,
                            pin_memory=True)

mask_dataset = dataset.MaskDataset(mask_dir=mask_data_dir)

mask_train_dataloader = dataloader.DataLoader(mask_dataset, batch_size=batch_size, shuffle=True, drop_last=True,
                            num_workers=2,
                            pin_memory=True)
mask_val_dataloader = dataloader.DataLoader(mask_dataset, batch_size=batch_size, shuffle=False, drop_last=True,
                            num_workers=2,
                            pin_memory=True)
mask_test_dataloader = dataloader.DataLoader(mask_dataset, batch_size=batch_size, shuffle=False, drop_last=True,
                            num_workers=2,
                            pin_memory=True)


# 定义网络
coarse_net = model.DINCAEModel(in_channels=channel)

criterion = nn.MSELoss()
optimizer = optim.Adam([{'params': coarse_net.parameters()}])

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
coarse_net = coarse_net.to(device)

print("device:", device)

# Tensorboard设置
writer = SummaryWriter("/root/tf-logs/" + folder_name + "/tensorboard")

# 开始训练
min_val_loss = 999.90
for epoch in range(501):  # loop over the dataset multiple times

    running_loss = 0.0
    iter_mask = iter(mask_train_dataloader)


    # 对模型训练
    for i, data in enumerate(train_dataloader, 0):

        # get the inputs; data is a list of [inputs, labels]
        inputs, missing, labels = data[0], data[1], data[2]

        # 给输入施加一个mask,施加在三天的chlo-anomaly上
        try:
            mask = next(iter_mask)
        except StopIteration:
            iter_mask = iter(mask_train_dataloader)
            mask = next(iter_mask)

        inputs[:, 1:2] *= mask
        inputs[:, 0:1] *= next(iter_mask)
        inputs[:, 2:3] *= next(iter_mask)

        # 将tensor移动到gpu上
        inputs = inputs.to(device)
        labels = labels.to(device)
        missing = missing.to(device)

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = coarse_net(inputs.float())


        # 损失函数不计算missed
        loss1 = criterion(outputs * missing, labels * missing)/missing.sum()*missing.numel()
        loss = loss1
        loss.backward()
        optimizer.step()

        # print statistics
        running_loss += loss1.item()
        if i == 19:
            writer.add_scalar('Training loss',
                              (running_loss/20)**0.5*training_std,
                              epoch + 1)
        if i % 20 == 19:    # print every 20 mini-batches
            print('[%d, %5d] loss: %.5f' %
                  (epoch + 1, i + 1, (running_loss/20)**0.5*training_std))
            running_loss = 0.0

    # 对模型进行validation
    if epoch % 5 == 4:
        val_epoch_loss = 0
        iter_mask = iter(mask_val_dataloader)
        for i, data in enumerate(val_dataloader, 0):

            # get the inputs; data is a list of [inputs, labels]
            inputs, missing, labels = data[0], data[1], data[2]

            # 给输入施加一个mask,施加在三天的chlo-anomaly上
            try:
                mask = next(iter_mask)
            except StopIteration:
                iter_mask = iter(mask_val_dataloader)
                mask = next(iter_mask)

            inputs[:, 1:2] *= mask
            inputs[:, 0:1] *= next(iter_mask)
            inputs[:, 2:3] *= next(iter_mask)
            # 将tensor移动到gpu上
            inputs = inputs.to(device)
            labels = labels.to(device)
            missing = missing.to(device)

            # forward + backward + optimize
            outputs = coarse_net(inputs.float())

            # 损失函数不计算missed
            loss2 = criterion(outputs * missing, labels * missing) / missing.sum() * missing.numel()

            # print statistics
            val_epoch_loss += loss2.item()

        val_loss = val_epoch_loss/len(val_dataloader) ** 0.5 * training_std

        if val_loss < min_val_loss:
            min_val_loss = val_loss
            torch.save(coarse_net, "/root/tf-logs/" +folder_name + "/coarse_" + str(epoch) + '.pth')
        writer.add_scalar('Validation loss',
                          val_loss,
                          epoch + 1)
        print('[%d] loss: %.5f' %
              (epoch + 1, val_loss))





# i = 0
# utils.show(labels[i][0])
# utils.show(mask[i][0])
# utils.show(inputs[i][1])
# utils.show(inputs[i][0])
# utils.show(inputs[i][2])
# utils.show(outputs[i][0])

# a = (outputs * missing - labels * missing)[i][0]
# utils.show(a)
# percent = a/labels[i][0]
# utils.show(percent)