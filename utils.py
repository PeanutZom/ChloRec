import random
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import os


def get_mask(train_dataset, mode=None, batch=32, channel=8):
    # 获取一个云的array(mask)
    if mode == 'random':
        return np.round(np.random.random_sample((batch, channel, 192, 240)) > 0.9)
    else:
        mask = np.zeros((batch, channel, 192, 240))
        for i in range(batch):
            mask[i] = train_dataset[random.randint(0, len(train_dataset)) - 1][1]
        return mask


def remove_time_mean(x):
    return x - x.mean(dim="time")


def get_anomaly(x):
    return np.log(x).groupby("time.month").map(remove_time_mean)


def get_normalized(x):
    x_log = np.log(x)
    x_normalized = (x_log - x_log.mean()) / x_log.std()
    return x_normalized


def get_denormalized(x, x_raw):
    x_denormalized = x * x_raw.std() + x.mean()
    return x_denormalized


def get_log(x):
    return np.log(x)


def show(output, format_file="data_format.nc"):
    format_dir = os.path.join(os.path.dirname(os.getcwd()), 'data', format_file)
    with xr.open_dataset(format_dir) as file:
        data_format = file
    output_np = output.detach().cpu().numpy()
    output_da = xr.DataArray(output_np,
                             dims=data_format.dims,
                             coords=data_format.coords,
                             name="log(Chlo)",
                             attrs=data_format.attrs)
    output_da.plot(cmap="RdBu_r", robust=True)
    plt.show()
