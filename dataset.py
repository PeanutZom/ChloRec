import torch
from torch.utils.data import Dataset
import xarray as xr
import numpy as np
import utils
import math
import os

class ChloDataset(Dataset):
    # 模型输入为三天的chlo数据，Nan值被覆盖为0
    # cost function只计算该数据有target的地方
    def __init__(self, data_filename="training_chlo.nc"):
        data_path = os.path.join(os.path.dirname(os.getcwd()), "data", data_filename)
        with xr.open_dataset(data_path) as chlo_data:
            self.chlo_data = chlo_data
            # 数据为log值的normalized
        self.chlo_anomaly_log = utils.get_normalized(self.chlo_data.CHL1_mean)
        self.std = np.log(self.chlo_data.CHL1_mean).std()
        self.landscape = np.round(self.chlo_data.CHL1_flags[0] % 16 >= 8)  # 记录地形为陆地的
        self.lon = self.chlo_data.lon
        self.lon = (self.lon - self.lon.min())/(self.lon.max() - self.lon.min()) * 2 - 1
        self.lat = self.chlo_data.lat
        self.lat = (self.lat - self.lat.min())/(self.lat.max() - self.lat.min()) * 2 - 1

    def __getitem__(self, item):
        # 输入的array,待转为tensor
        input_np = np.zeros((8, 192, 240))
        missing_mask = np.zeros((8, 192, 240))
        # DataArray数据转为三天的masked array
        chlo_ma_prev = self.chlo_anomaly_log[item].to_masked_array()
        chlo_ma_cur = self.chlo_anomaly_log[item + 1].to_masked_array()
        chlo_ma_next = self.chlo_anomaly_log[item + 2].to_masked_array()

        # 设置输入input_np, 为1-3层为chlo
        input_np[0] = chlo_ma_prev.filled(0)
        input_np[1] = chlo_ma_cur.filled(0)
        input_np[2] = chlo_ma_next.filled(0)
        input_np[3] = self.landscape
        input_np[4] = np.expand_dims(self.lat, 1)
        input_np[5] = np.expand_dims(self.lon, 0)
        input_np[6] = np.sin(self.chlo_data.time[item + 1].dt.dayofyear/366 * math.pi * 2)
        input_np[7] = np.cos(self.chlo_data.time[item + 1].dt.dayofyear/366 * math.pi * 2)
        # 数据本身的缺失值
        missing_mask[0] = 1 - chlo_ma_prev.mask
        missing_mask[1] = 1 - chlo_ma_cur.mask
        missing_mask[2] = 1 - chlo_ma_next.mask
        missing_mask[3] = 1
        missing_mask[4] = 1
        missing_mask[5] = 1
        missing_mask[6] = 1
        missing_mask[7] = 1
        # target是当天的数据
        target = input_np[1:2]
        return torch.from_numpy(input_np), torch.from_numpy(missing_mask), torch.from_numpy(target)

    def __len__(self):
        return len(self.chlo_data.coords["time"])-2


