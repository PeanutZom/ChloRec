import xarray as xr
import numpy as np
import os
import pandas as pd


def load_data():
    # 使用os获取文件夹下的所有文件名称并使用xarray进行读取并合并为一个DataArray
    dir_path = 'D:\Final Project\data\chlo daily'
    filenames = os.listdir(dir_path)
    chlo_mean = []

    for filename in filenames:
        with xr.open_dataset(os.path.join(dir_path, filename)) as daily:
            daily = daily.expand_dims("time")
            daily.coords["time"] = pd.to_datetime([filename[4:12]])
            chlo_mean.append(daily)

    merged = xr.concat(chlo_mean, dim="time")
    return merged


def merge():
    merged = load_data()
    merged = merged.isel(lat=slice(1, None), lon=slice(1, None))
    merged.to_netcdf("D://Final Project//data//chlo_data.nc")


