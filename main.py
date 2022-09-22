import xarray as xr
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import matplotlib.pyplot as plt


class ChloDataset(Dataset):
    # Initiate the dataset with an array
    def __init__(self, chlo_map_set):
        self.chlo_map_set = chlo_map_set.filled(0)
        self.missed_set = 1 - chlo_map_set.mask

    def __len__(self):
        return len(self.chlo_map_set)

    def __getitem__(self, idx):
        chlo_map = torch.from_numpy(self.chlo_map_set[idx])
        chlo_map = chlo_map.unsqueeze(0)
        missed_pos = torch.from_numpy(self.missed_set[idx])
        missed_pos = missed_pos.unsqueeze(0)
        # missed_pos is the missing mask in the original data, while mask is the random made-up mask
        mask = np.round(np.random.random_sample(chlo_map.shape) > 0.5)
        mask_tensor = torch.from_numpy(mask)
        input = torch.mul(chlo_map, mask_tensor)
        input_posited = torch.cat([input, missed_pos, mask_tensor], 0)
        # output = torch.subtract(chlo_map, input)
        return input_posited, chlo_map


class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        # 1 input image channel, 6 output channels, 5x5 square convolution
        # kernel
        self.conv1 = nn.Conv2d(3, 4, 5, padding=2)
        self.conv2 = nn.Conv2d(3, 4, 9, padding=4)
        self.conv3 = nn.Conv2d(3, 4, 11, padding=5)
        self.conv4 = nn.Conv2d(12, 12, 5, padding=2)
        self.conv5 = nn.Conv2d(12, 4, 11, padding=5)
        self.conv6 = nn.Conv2d(4, 1, 5, padding=2)
        self.bn1 = nn.BatchNorm2d(12)
        self.bn2 = nn.BatchNorm2d(12)
        self.bn3 = nn.BatchNorm2d(4)

    def forward(self, x):
        # Max pooling over a (2, 3) window
        x_0 = x.clone()[:, 0].unsqueeze(1)
        x_missing = x.clone()[:, 1].unsqueeze(1)
        x = F.leaky_relu(self.bn1(torch.cat([self.conv1(x), self.conv2(x), self.conv3(x)], 1)))
        x = F.leaky_relu(self.bn2(self.conv4(x)))
        x = F.leaky_relu(self.bn3(self.conv5(x)))
        x = F.relu(self.conv6(x))
        x = x + x_0
        x = x * x_missing
        # x = F.relu(x)
        return x


with xr.open_dataset("data//chlo_data.nc") as daily:
    chlo_mean = daily.CHL1_mean.variable.values

masked_chlo_mean = np.ma.masked_invalid(chlo_mean)
training_data = ChloDataset(chlo_map_set=masked_chlo_mean[:720])
test_data = ChloDataset(chlo_map_set=np.ma.masked_invalid(masked_chlo_mean[:720].filled(0)))

train_dataloader = DataLoader(training_data, batch_size=24, shuffle=True)
test_dataloader = DataLoader(test_data, batch_size=24, shuffle=False)


net = Net()

criterion = nn.MSELoss()
optimizer = optim.Adam(net.parameters())

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
net = net.to(device)

for epoch in range(2):  # loop over the dataset multiple times

    running_loss = 0.0
    for i, data in enumerate(train_dataloader, 0):
        # get the inputs; data is a list of [inputs, labels]
        inputs, labels = data[0].to(device), data[1].to(device)

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # print statistics
        running_loss += loss.item()
        if i % 20 == 19:    # print every 200 mini-batches
            print('[%d, %5d] loss: %.3f' %
                  (epoch + 1, i + 1, running_loss/20))
            running_loss = 0.0

print('Finished Training')


data_iter = iter(test_dataloader)
to_predict, true_value = data_iter.next()

to_predict = to_predict.to(device)
true_value = true_value.to(device)

prediction = net(to_predict)

testMSE = criterion(prediction, true_value)

print("MSE in test data: %.3f" % (testMSE.item()))

to_predict_np = to_predict.detach().cpu().numpy()
true_np = true_value.cpu().numpy()
prediction_np = prediction.detach().cpu().numpy()


def draw():
    with xr.open_dataset("data\816964734\L3m_20190102__816964734_4_GSM-MODVIR_CHL1_DAY_00.nc") as form:
        a = form
    for i in range(1, 10):
        a["CHL1_mean"].values = prediction_np[i][0]
        a["CHL1_mean"].plot(x='lon', y='lat')
        plt.show()

        a["CHL1_mean"].values = true_np[i][0]
        for j in range(193):
            for k in range(241):
                if a["CHL1_mean"].values[j, k] == 0:
                    a["CHL1_mean"].values[j, k] = np.nan
        a["CHL1_mean"].plot(x='lon', y='lat')
        plt.show()

        a["CHL1_mean"].values = to_predict_np[i][0]
        for j in range(193):
            for k in range(241):
                if a["CHL1_mean"].values[j, k] == 0:
                    a["CHL1_mean"].values[j, k] = np.nan

        a["CHL1_mean"].plot(x='lon', y='lat')
        plt.show()

