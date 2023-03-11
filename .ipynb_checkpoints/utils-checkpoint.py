import random
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import os
import dataset
from torch.utils.data import dataloader
from torchvision import transforms
import torch
import cartopy.crs as ccrs


# def get_mask(train_dataset, mode=None, batch=32, channel=8):
#     # 获取一个云的array(mask)
#     if mode == 'random':
#         return np.round(np.random.random_sample((batch, channel, 192, 240)) > 0.9)
#     else:
#         mask = np.zeros((batch, channel, 192, 240))
#         for i in range(batch):
#             mask[i] = train_dataset[random.randint(0, len(train_dataset)) - 1][1]
#         return mask

def get_mask(train_dataset, inputs):
    # 获取一个云的array(mask)
    batch = inputs.shape[0]
    channel = inputs.shape[1]
    mask = np.zeros((batch, channel, inputs.shape[2], inputs.shape[3]))
    mask = 1 - mask
    for i in range(batch):
        mask[i][1] = train_dataset[random.randint(0, len(train_dataset)) - 1][1][1]
    return mask


def remove_time_mean(x):
    return x - x.mean(dim="time")


def get_anomaly(x):
    return np.log(x).groupby("time.month").map(remove_time_mean)


def get_normalized(x):
    x_normalized = (x - x.mean()) / x.std()
    return x_normalized


def get_denormalized(x):
    training_mean = -2.2366745
    training_std = 0.81302816
    x_denormalized = np.exp(x * training_std + training_mean)
    return x_denormalized


def get_log(x):
    return np.log(x)


def show(output, format_file="data_format.nc", name="Chlo", color=None, robust=True, mask_land=True, vmax=None, vmin=None):
    format_dir = os.path.join(os.path.dirname(os.getcwd()), 'data', format_file)
    with xr.open_dataset(format_dir) as file:
        data_format = file
    land_mask = 1

    if mask_land:
        land_mask = (1 - np.round(data_format.CHL1_flags % 16 >= 8))
        land_mask = np.where(land_mask == 0, np.nan, land_mask)

    output_np = output.detach().cpu().numpy() * land_mask
    output_da = xr.DataArray(output_np,
                             dims=data_format.dims,
                             coords=data_format.coords,
                             name=name,
                             attrs=data_format.attrs)

    pic = output_da.plot(cmap=color, robust=robust, transform=ccrs.PlateCarree(), subplot_kws=dict(projection=ccrs.PlateCarree()), vmax=vmax, vmin=vmin)
    pic.axes.coastlines()
    plt.show()


def get_stat():
    print('----')
    train_data_dir = os.path.join(os.path.dirname(os.getcwd()), 'data', 'ECS', 'train')
    statDataset = dataset.DINCAEDataset(data_dir=train_data_dir, transform=None, target_transform=None)
    statDataloader = dataloader.DataLoader(statDataset, batch_size=1000, shuffle=True)
    data = next(iter(statDataloader))
    return data[2]
