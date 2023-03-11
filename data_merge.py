import xarray as xr
import numpy as np
import os
import pandas as pd


def load_data(datatype):
    # 使用os获取文件夹下的所有文件名称并使用xarray进行读取并合并为一个DataArray
    dir_path = os.path.join(os.path.dirname(os.getcwd()), 'data', datatype)
    filenames = os.listdir(dir_path)
    chlo_mean = []

    for filename in filenames:
        with xr.open_dataset(os.path.join(dir_path, filename)) as daily:
            print(filename)
            daily = daily.expand_dims("time")
            daily.coords["time"] = pd.to_datetime([filename[4:12]])
            chlo_mean.append(daily)

    merged = xr.concat(chlo_mean, dim="time")
    return merged


def merge(datatype):
    merged = load_data(datatype)
    merged = merged.isel(lat=slice(1, None), lon=slice(1, None))
    dir_path = os.path.join(os.path.dirname(os.getcwd()), 'data')
    filename = datatype + ".nc"
    merged.to_netcdf(os.path.join(dir_path, filename))




