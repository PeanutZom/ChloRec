import dataset
import model
from torch.utils.data import dataloader
import torch.nn as nn
import torch.optim as optim
import torch
import utils

# 初始化数据集，数据加载器， 模型， 损失函数， 优化器
batch_size = 32


train_dataset = dataset.ChloDataset()
train_dataloader = dataloader.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

test_dataset = dataset.ChloDataset()
test_dataloader = dataloader.DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

net = model.DINCAEModel()
criterion = nn.MSELoss()
optimizer = optim.Adam(net.parameters())
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
net = net.to(device)
print("device:", device)
# 开始训练
for epoch in range(50):  # loop over the dataset multiple times

    running_loss = 0.0
    for i, data in enumerate(train_dataloader, 0):

        # get the inputs; data is a list of [inputs, labels]
        inputs, missing, labels = data[0], data[1], data[2]

        # 给输入施加一个mask,施加在三天的chlo-anomaly上
        mask = utils.get_mask(train_dataset, batch=batch_size)
        inputs_masked = inputs * mask
        missing_masked = missing * mask

        # 将tensor移动到gpu上
        inputs_masked = inputs_masked.to(device)
        missing_masked = missing_masked.to(device)
        labels = labels.to(device)
        missing = missing.to(device)

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = net(inputs_masked.float())
        # 损失函数不计算missed
        loss = criterion(outputs * missing[:, 1:2], labels * missing[:, 1:2])/missing[:, 1:2].sum()*missing[:, 1:2].numel()
        loss.backward()
        optimizer.step()

        # print statistics
        running_loss += loss.item()
        if i % 20 == 19:    # print every 20 mini-batches
            print('[%d, %5d] loss: %.5f' %
                  (epoch + 1, i + 1, (running_loss/20)**0.5*train_dataset.std.item()))
            running_loss = 0.0

print('Finished Training')

MSE_Loss = (criterion(outputs * missing[:, 1:2], labels * missing[:, 1:2])/missing[:, 1:2].sum()*missing[:, 1:2].numel()).sqrt()*train_dataset.std.item()

i = 8
utils.show(inputs[i][1])
utils.show(inputs_masked[i][1])
utils.show(outputs[i][0])
utils.show(labels[i][0])
utils.show(missing[i, 1])
a = (outputs * missing[:, 1:2]-labels * missing[:, 1:2])[i][0]
utils.show(a)
percent = a/labels[i][0]
utils.show(percent)
percent_np = percent.cpu().detach().numpy()
