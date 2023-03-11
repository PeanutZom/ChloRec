import os
import random
import shutil
from shutil import copy2
from  shutil import copytree


def data_combine(data_folder, data_root):
    image_names = os.listdir(data_folder)
    combined_dir = os.path.join(data_root, 'combined')
    os.mkdir(combined_dir, mode=777)
    image_num = len(image_names)
    for i in range(image_num - 2):
        image_name = image_names[i + 1]
        sample_date = image_name[4:12]
        dst = os.path.join(combined_dir, sample_date)
        os.mkdir(dst, mode=777)

        copy2(os.path.join(data_folder, image_names[i]), dst)
        copy2(os.path.join(data_folder, image_names[i + 1]), dst)
        copy2(os.path.join(data_folder, image_names[i + 2]), dst)
        os.rename(os.path.join(dst, image_names[i]), os.path.join(dst, 'prev.nc'))
        os.rename(os.path.join(dst, image_names[i + 1]), os.path.join(dst, 'cur.nc'))
        os.rename(os.path.join(dst, image_names[i + 2]), os.path.join(dst, 'next.nc'))



def data_split(data_folder, data_root, train_scales = 0.8,val_scales = 0.1,test_scales = 0.1):
    sample_names = os.listdir(data_folder)

    train_folder = os.path.join(data_root, 'train')       #分割后的训练数据集路径
    val_folder = os.path.join(data_root, 'val')
    test_folder = os.path.join(data_root, 'test')

    # os.mkdir(train_folder, mode=777)
    # os.mkdir(val_folder, mode=777)
    # os.mkdir(test_folder, mode=777)

    sample_num = len(sample_names)
    index_list = list(range(sample_num))
    random.shuffle(index_list)

    train_stop_flag = sample_num * train_scales
    val_stop_flag = sample_num * (train_scales + val_scales)

    train_num = 0
    val_num = 0
    test_num = 0

    for i in range(sample_num):
        if i <= train_stop_flag:
            copytree(os.path.join(data_folder, sample_names[index_list[i]]), os.path.join(train_folder, sample_names[index_list[i]]))
            train_num += 1
        elif i <= val_stop_flag:
            copytree(os.path.join(data_folder, sample_names[index_list[i]]), os.path.join(val_folder, sample_names[index_list[i]]))
            val_num += 1
        else:
            copytree(os.path.join(data_folder, sample_names[index_list[i]]), os.path.join(test_folder, sample_names[index_list[i]]))
            test_num += 1

    print('训练集', train_num)
    print('验证集', val_num)
    print('测试集', test_num)


if __name__ == '__main__':
    # data_root = os.path.join(os.path.dirname(os.getcwd()), 'data', 'ECS')
    # data_folder = os.path.join(data_root, 'raw')  # 数据源文件地址
    # data_combine(data_folder, data_root)
    data_root = os.path.join(os.path.dirname(os.getcwd()), 'data', 'ECS')
    data_folder = os.path.join(data_root, 'combined')  # 数据源文件地址
    data_split(data_folder, data_root)
