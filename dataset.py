import torch
from torch.utils.data import Dataset
import xarray as xr
import numpy as np
import utils
import math
import os
import pandas as pd


class DINCAEDataset(Dataset):
    def __init__(self, data_dir=None, transform=None, target_transform=None, use_elevation=True, use_coor=True):
        """
        @param data_dir: 数据集目录名
        根据目录名获取相对路径，访问对应文件夹
        """
        self.transform = transform
        self.target_transform = target_transform
        self.data_dir = data_dir
        self.sample_dirs = os.listdir(data_dir)
        with xr.open_dataset(os.path.join(os.path.dirname(data_dir), 'bathymetry.nc')) as bathymetry:
            self.elevation = bathymetry.elevation.to_numpy()
        self.elevation = np.float_power(abs(self.elevation), 1.0/5)*np.sign(self.elevation)/5
        #self.elevation = self.elevation/5000
        self.use_elevation = use_elevation
        self.use_coor = use_coor

    def __getitem__(self, item):
        """
        @param item:
        @return: 返回第item个input和第item个mask
        """
        date = self.sample_dirs[item]

        sample_dir = os.path.join(self.data_dir, date)
        with xr.open_dataset(os.path.join(sample_dir, 'prev.nc')) as prev:
            prev = prev
        with xr.open_dataset(os.path.join(sample_dir, 'cur.nc')) as cur:
            cur = cur
        with xr.open_dataset(os.path.join(sample_dir, 'next.nc')) as next:
            next = next

        channel = 12
        input_np = np.zeros((channel, cur.CHL1_mean.shape[0], cur.CHL1_mean.shape[1]))
        missing_mask = np.zeros((1, cur.CHL1_mean.shape[0], cur.CHL1_mean.shape[1]))
        lat = cur.lat
        lon = cur.lon
        # 设置输入input_np, 为1-3层为chlo
        input_np[0] = prev.CHL1_mean
        input_np[1] = cur.CHL1_mean
        input_np[2] = next.CHL1_mean
        if self.use_elevation:
            # input_np[3] = np.round(cur.CHL1_flags % 16 >= 8)
            input_np[8] = self.elevation               
        if self.use_coor:
            input_np[5] = np.expand_dims((lon)/(360) * 2 - 1, 0)
            input_np[4] = np.expand_dims((lat)/(90) * 2 - 1, 1)

        input_np[6] = np.sin(pd.to_datetime(date).dayofyear / 366 * math.pi * 2)
        input_np[7] = np.cos(pd.to_datetime(date).dayofyear / 366 * math.pi * 2)
        input_np[9] = 1 - prev.CHL1_mean.to_masked_array().mask
        input_np[10] = 1 - cur.CHL1_mean.to_masked_array().mask
        input_np[11] = 1 - next.CHL1_mean.to_masked_array().mask
        # 数据的mask
        missing_mask[0] = 1 - cur.CHL1_mean.to_masked_array().mask
        # target是当天的数据
        target = input_np[1:2]

        # 数据的transform
        missing_mask_ts = torch.from_numpy(missing_mask)
        input_ts = torch.from_numpy(input_np)
        target_ts = torch.from_numpy(target)

        if self.transform is not None:
            input_ts = torch.cat([self.transform(input_ts[0:3]), input_ts[3:]], 0)

        if self.target_transform is not None:
            target_ts = self.target_transform(target_ts)

        return input_ts, missing_mask_ts, target_ts

    def __len__(self):
        return len(self.sample_dirs)





class MaskDataset(Dataset):
    def __init__(self, mask_dir):
        """
        @param mask_dir: 数据集目录名
        根据目录名获取相对路径，访问对应文件夹
        """
        self.mask_dir = mask_dir
        self.mask_names = os.listdir(mask_dir)

    def __getitem__(self, item):
        """
        @param item:
        @return: 返回第item个mask
        """
        mask_path = os.path.join(self.mask_dir, self.mask_names[item])
        with xr.open_dataset(mask_path) as mask:
            mask = mask
        return torch.from_numpy(np.expand_dims(1 - mask.CHL1_mean.to_masked_array().mask, 0))

    def __len__(self):
        return len(self.mask_names)
