import numpy as np
from PIL import Image
import torch.utils.data as data
import torch
import os


class TestData(data.Dataset):
    def __init__(self, test_img, test_label, transform=None, img_size=(144, 288)):
        self.test_image = test_img
        self.test_label = test_label
        self.transform = transform

    def __getitem__(self, index):
        img1, target1 = self.test_image[index], self.test_label[index]
        img1 = self.transform(img1)
        return img1, target1

    def __len__(self):
        return len(self.test_image)


# class TestDataOld(data.Dataset):
#     def __init__(self, data_dir, test_img_file, test_label, transform=None, img_size = (144,288)):
#
#         test_image = []
#         for i in range(len(test_img_file)):
#             img = Image.open(data_dir + test_img_file[i])
#             img = img.resize((img_size[0], img_size[1]), Image.ANTIALIAS)
#             pix_array = np.array(img)
#             test_image.append(pix_array)
#         test_image = np.array(test_image)
#         self.test_image = test_image
#         self.test_label = test_label
#         self.transform = transform
#
#     def __getitem__(self, index):
#         img1,  target1 = self.test_image[index],  self.test_label[index]
#         img1 = self.transform(img1)
#         return img1, target1
#
#     def __len__(self):
#         return len(self.test_image)
def load_data(input_data_path):
    with open(input_data_path) as f:
        data_file_list = open(input_data_path, 'rt').read().splitlines()
        # Get full list of image and labels
        file_image = [s.split(' ')[0] for s in data_file_list]
        file_label = [int(s.split(' ')[1]) for s in data_file_list]

    return file_image, file_label


# class RGBNT201Data(data.Dataset):
#     def __init__(self, data_dir, rgbnt_mode, transform_rgb=None, transform_ni=None, transform_ti=None, rgbIndex=None, niIndex=None, tiIndex=None):
#         # Load training images (path) and labels
#         data_dir = '../datasets/RGBNT201/RGBNT201'
#
#         train_rgb_path = data_dir + '/train_141/RGB/'
#         train_ni_path = data_dir + '/train_141/NI/'
#         trian_ti_path = data_dir + '/train_141/TI/'
#
#         train_rgb_list = os.listdir(train_rgb_path)
#         train_ni_list = os.listdir(train_ni_path)
#         train_ti_list = os.listdir(trian_ti_path)
#
#         train_rgb_label = [s.split('_')[0] for s in train_rgb_list]
#         train_ni_label = [s.split('_')[0] for s in train_ni_list]
#         train_ti_label = [s.split('_')[0] for s in train_ti_list]
#
#         pid2label = {pid: label for label, pid in enumerate(set(train_rgb_label))}
#         train_rgb_label = [pid2label[v] for v in train_rgb_label]
#         train_ni_label = [pid2label[v] for v in train_ni_label]
#         train_ti_label = [pid2label[v] for v in train_ti_label]
#         # print(train_rgb_label)
#
#         train_rgb_image = []
#         for i in range(len(train_rgb_list)):
#             img = Image.open(train_rgb_path + train_rgb_list[i])
#             img = img.resize((144, 288), Image.ANTIALIAS)
#             pix_array = np.array(img)
#             train_rgb_image.append(pix_array)
#         train_rgb_image = np.array(train_rgb_image)
#
#         train_ni_image = []
#         for i in range(len(train_ni_list)):
#             img = Image.open(train_ni_path + train_ni_list[i])
#             img = img.resize((144, 288), Image.ANTIALIAS)
#             pix_array = np.array(img)
#             train_ni_image.append(pix_array)
#         train_ni_image = np.array(train_ni_image)
#
#         train_ti_image = []
#         for i in range(len(train_ti_list)):
#             img = Image.open(trian_ti_path + train_ti_list[i])
#             img = img.resize((144, 288), Image.ANTIALIAS)
#             pix_array = np.array(img)
#             train_ti_image.append(pix_array)
#         train_ti_image = np.array(train_ti_image)
#
#
#         # BGR to RGB
#         self.train_rgb_image = train_rgb_image
#         self.train_rgb_label = train_rgb_label
#
#         self.train_ni_image = train_ni_image
#         self.train_ni_label = train_ni_label
#
#         self.train_ti_image = train_ti_image
#         self.train_ti_label = train_ti_label
#
#         self.transform_rgb = transform_rgb
#         self.transform_ni = transform_ni
#         self.transform_ti = transform_ti
#
#         self.rgbIndex = rgbIndex
#         self.niIndex = niIndex
#         self.tiIndex = tiIndex
#
#         self.rgbnt_mode = rgbnt_mode
#
#     def __getitem__(self, index):
#
#         if self.rgbnt_mode == 'rgb_ni' or self.rgbnt_mode == 'ni_rgb':
#             rgb_idx = index[0]
#             ni_idx = index[1]
#             img1, target1 = self.train_rgb_image[rgb_idx], self.train_rgb_label[rgb_idx]
#             img2, target2 = self.train_ni_image[ni_idx], self.train_ni_label[ni_idx]
#             target1 = int(target1)
#             target2 = int(target2)
#             img1 = self.transform_rgb(img1)
#             img2 = self.transform_ni(img2)
#             return img1, img2, target1, target2
#         elif self.rgbnt_mode == 'rgb_ti' or self.rgbnt_mode == 'ti_rgb':
#             rgb_idx = index[0]
#             ti_idx = index[1]
#             img1, target1 = self.train_rgb_image[rgb_idx], self.train_rgb_label[rgb_idx]
#             img2, target2 = self.train_ti_image[ti_idx], self.train_ti_label[ti_idx]
#             target1 = int(target1)
#             target2 = int(target2)
#             img1 = self.transform_rgb(img1)
#             img2 = self.transform_ti(img2)
#             return img1, img2, target1, target2
#         elif self.rgbnt_mode == 'ni_ti' or self.rgbnt_mode == 'ti_ni':
#             ni_idx = index[0]
#             ti_idx = index[1]
#             img1, target1 = self.train_ni_image[ni_idx], self.train_ni_label[ni_idx]
#             img2, target2 = self.train_ti_image[ti_idx], self.train_ti_label[ti_idx]
#             target1 = int(target1)
#             target2 = int(target2)
#             img1 = self.transform_ni(img1)
#             img2 = self.transform_ti(img2)
#             return img1, img2, target1, target2
#
#
#     def __len__(self):
#         # print(len(self.train_color_label))
#         return len(self.train_rgb_label)


# class RGBNT100Data(data.Dataset):
#     def __init__(self, data_dir, rgbnt_mode, transform_rgb=None, transform_ni=None, transform_ti=None, rgbIndex=None, niIndex=None, tiIndex=None):
#         # Load training images (path) and labels
#         # data_dir = '../datasets/RGBNT201/RGBNT201'
#         data_dir = '../datasets/AHU_NIRTIR/RGBNT100_CM'
#
#         train_rgb_path = data_dir + '/train_141/RGB/'
#         train_ni_path = data_dir + '/train_141/NI/'
#         trian_ti_path = data_dir + '/train_141/TI/'
#
#         train_rgb_list = os.listdir(train_rgb_path)
#         train_ni_list = os.listdir(train_ni_path)
#         train_ti_list = os.listdir(trian_ti_path)
#
#         train_rgb_label = [s.split('_')[0] for s in train_rgb_list]
#         train_ni_label = [s.split('_')[0] for s in train_ni_list]
#         train_ti_label = [s.split('_')[0] for s in train_ti_list]
#
#         pid2label = {pid: label for label, pid in enumerate(set(train_rgb_label))}
#         train_rgb_label = [pid2label[v] for v in train_rgb_label]
#         train_ni_label = [pid2label[v] for v in train_ni_label]
#         train_ti_label = [pid2label[v] for v in train_ti_label]
#         # print(train_rgb_label)
#
#         train_rgb_image = []
#         for i in range(len(train_rgb_list)):
#             img = Image.open(train_rgb_path + train_rgb_list[i])
#             img = img.resize((144, 288), Image.ANTIALIAS)
#             pix_array = np.array(img)
#             train_rgb_image.append(pix_array)
#         train_rgb_image = np.array(train_rgb_image)
#
#         train_ni_image = []
#         for i in range(len(train_ni_list)):
#             img = Image.open(train_ni_path + train_ni_list[i])
#             img = img.resize((144, 288), Image.ANTIALIAS)
#             pix_array = np.array(img)
#             train_ni_image.append(pix_array)
#         train_ni_image = np.array(train_ni_image)
#
#         train_ti_image = []
#         for i in range(len(train_ti_list)):
#             img = Image.open(trian_ti_path + train_ti_list[i])
#             img = img.resize((144, 288), Image.ANTIALIAS)
#             pix_array = np.array(img)
#             train_ti_image.append(pix_array)
#         train_ti_image = np.array(train_ti_image)
#
#
#         # BGR to RGB
#         self.train_rgb_image = train_rgb_image
#         self.train_rgb_label = train_rgb_label
#
#         self.train_ni_image = train_ni_image
#         self.train_ni_label = train_ni_label
#
#         self.train_ti_image = train_ti_image
#         self.train_ti_label = train_ti_label
#
#         self.transform_rgb = transform_rgb
#         self.transform_ni = transform_ni
#         self.transform_ti = transform_ti
#
#         self.rgbIndex = rgbIndex
#         self.niIndex = niIndex
#         self.tiIndex = tiIndex
#
#         self.rgbnt_mode = rgbnt_mode
#
#     def __getitem__(self, index):
#
#         if self.rgbnt_mode == 'rgb_ni' or self.rgbnt_mode == 'ni_rgb':
#             rgb_idx = index[0]
#             ni_idx = index[1]
#             img1, target1 = self.train_rgb_image[rgb_idx], self.train_rgb_label[rgb_idx]
#             img2, target2 = self.train_ni_image[ni_idx], self.train_ni_label[ni_idx]
#             target1 = int(target1)
#             target2 = int(target2)
#             img1 = self.transform_rgb(img1)
#             img2 = self.transform_ni(img2)
#             return img1, img2, target1, target2
#         elif self.rgbnt_mode == 'rgb_ti' or self.rgbnt_mode == 'ti_rgb':
#             rgb_idx = index[0]
#             ti_idx = index[1]
#             img1, target1 = self.train_rgb_image[rgb_idx], self.train_rgb_label[rgb_idx]
#             img2, target2 = self.train_ti_image[ti_idx], self.train_ti_label[ti_idx]
#             target1 = int(target1)
#             target2 = int(target2)
#             img1 = self.transform_rgb(img1)
#             img2 = self.transform_ti(img2)
#             return img1, img2, target1, target2
#         elif self.rgbnt_mode == 'ni_ti' or self.rgbnt_mode == 'ti_ni':
#             ni_idx = index[0]
#             ti_idx = index[1]
#             img1, target1 = self.train_ni_image[ni_idx], self.train_ni_label[ni_idx]
#             img2, target2 = self.train_ti_image[ti_idx], self.train_ti_label[ti_idx]
#             target1 = int(target1)
#             target2 = int(target2)
#             img1 = self.transform_ni(img1)
#             img2 = self.transform_ti(img2)
#             return img1, img2, target1, target2
#
#
#     def __len__(self):
#         # print(len(self.train_color_label))
#         return len(self.train_rgb_label)

class RGBN300Data(data.Dataset):
    def __init__(self, data_dir, transform_rgb=None, transform_gray=None, transform_nir=None, rgbIndex=None,
                 niIndex=None, imgh=256, imgw=256):


        data_dir = '/media/jqzhu/941A7DD31A7DB33A/ZQQ/datasets/RGBN300_autolabel'
        # data_dir = '/media/jqzhu/e/qqzhao/datasets/RGBN300_autolabel'
        # data_dir = '/media/l/b/qqzhao/datasets/AHU_NIRTIR/RGBN300'

        train_rgb_path = data_dir + '/R/'
        train_ni_path = data_dir + '/N/'
        # train_gray_path = data_dir + '/Gray/'
        print(imgh, imgw)

        train_rgb_list_all = os.listdir(train_rgb_path)  # all id in the rgb part of the whole dataset
        # train_gray_list_all = os.listdir(train_gray_path)  # all id in the rgb part of the whole dataset
        train_nir_list_all = os.listdir(train_ni_path)  # all id in the nir part of the whole dataset

        #############################select odd to construct training set#################
        # print(train_rgb_list, train_ni_list)
        train_rgb_list = []  # select odd
        for i in range(len(train_rgb_list_all)):
            cur_id = train_rgb_list_all[i]
            cur_id = int(cur_id)  # str2num
            if cur_id % 2 != 0:
                # print(cur_id)
                train_rgb_list.append(train_rgb_list_all[i])

        # train_gray_list = []  # select odd
        # for i in range(len(train_gray_list_all)):
        #     cur_id = train_gray_list_all[i]
        #     cur_id = int(cur_id)  # str2num
        #     if cur_id % 2 != 0:
        #         # print(cur_id)
        #         train_gray_list.append(train_gray_list_all[i])

        # print(len(train_rgb_list))
        train_nir_list = []  # select odd
        for i in range(len(train_nir_list_all)):
            cur_id = train_nir_list_all[i]
            cur_id = int(cur_id)  # str2num
            if cur_id % 2 != 0:
                # print(cur_id)
                train_nir_list.append(train_nir_list_all[i])

        train_rgb_name = []
        train_rgb_image = []
        for i in range(len(train_rgb_list)):
            cur_id_path = train_rgb_path + train_rgb_list[i]
            cur_id_list = os.listdir(cur_id_path)
            # print(cur_id_list)
            for j in range(len(cur_id_list)):
                imgname = cur_id_path + '/' + cur_id_list[j]
                if os.path.splitext(imgname)[1] == '.jpg':
                    img = Image.open(imgname)
                    img = img.resize((imgh, imgw), Image.ANTIALIAS)
                    pix_array = np.array(img)
                    train_rgb_image.append(pix_array)
                    train_rgb_name.append(cur_id_list[j])
        train_rgb_image = np.array(train_rgb_image)

        # train_gray_name = []
        # train_gray_image = []
        # for i in range(len(train_gray_list)):
        #     cur_id_path = train_gray_path + train_gray_list[i]
        #     cur_id_list = os.listdir(cur_id_path)
        #     # print(cur_id_list)
        #     for j in range(len(cur_id_list)):
        #         imgname = cur_id_path + '/' + cur_id_list[j]
        #         if os.path.splitext(imgname)[1] == '.jpg':
        #             img = Image.open(imgname)
        #             img = img.resize((imgh, imgw), Image.ANTIALIAS)
        #             pix_array = np.array(img)
        #             train_gray_image.append(pix_array)
        #             train_gray_name.append(cur_id_list[j])
        # train_gray_image = np.array(train_gray_image)

        train_nir_name = []
        train_nir_image = []
        for i in range(len(train_nir_list)):
            cur_id_path = train_ni_path + train_nir_list[i]
            cur_id_list = os.listdir(cur_id_path)
            for j in range(len(cur_id_list)):
                imgname = cur_id_path + '/' + cur_id_list[j]
                # print(imgname)
                if os.path.splitext(imgname)[1] == '.jpg':
                    img = Image.open(imgname)
                    img = img.resize((imgh, imgw), Image.ANTIALIAS)
                    pix_array = np.array(img)
                    train_nir_image.append(pix_array)
                    train_nir_name.append(cur_id_list[j])
        train_nir_image = np.array(train_nir_image)

        ################################process person id: tranform 0599 to 149.0##########################
        train_rgb_id = [s.split('_')[0] for s in train_rgb_name]
        # train_gray_id = [s.split('_')[0] for s in train_gray_name]
        train_nir_id = [s.split('_')[0] for s in train_nir_name]
        train_multi_modal_id = train_rgb_id + train_nir_id
        unique_id = np.unique(train_multi_modal_id)
        train_rgb_label = np.ones(len(train_rgb_id)) * -1
        # train_gray_label = np.ones(len(train_gray_id)) * -1
        train_nir_label = np.ones(len(train_nir_id)) * -1

        for i in range(len(unique_id)):
            tmp_rgb = [k for k, v in enumerate(train_rgb_id) if v == unique_id[i]]
            train_rgb_label[tmp_rgb] = i
            #
            # tmp_gray = [k for k, v in enumerate(train_gray_id) if v == unique_id[i]]
            # train_gray_label[tmp_gray] = i

            tmp_nir = [k for k, v in enumerate(train_nir_id) if v == unique_id[i]]
            train_nir_label[tmp_nir] = i

        ################################process camera id: tranform c0001 to 0.0 ##########################
        train_rgb_cid = [s.split('_')[1][1:] for s in train_rgb_name]
        # train_gray_cid = [s.split('_')[1][1:] for s in train_gray_name]
        train_nir_cid = [s.split('_')[1][1:] for s in train_nir_name]
        train_multi_modal_cid = train_rgb_cid + train_nir_cid
        unique_cid = np.unique(train_multi_modal_cid)
        train_rgb_clabel = np.ones(len(train_rgb_cid)) * -2
        # train_gray_clabel = np.ones(len(train_gray_cid)) * -2
        train_nir_clabel = np.ones(len(train_nir_cid)) * -2
        for i in range(len(unique_cid)):
            tmp_rgb = [k for k, v in enumerate(train_rgb_cid) if v == unique_cid[i]]
            train_rgb_clabel[tmp_rgb] = i

            # tmp_gray = [k for k, v in enumerate(train_gray_cid) if v == unique_cid[i]]
            # train_gray_clabel[tmp_gray] = i

            tmp_nir = [k for k, v in enumerate(train_nir_cid) if v == unique_cid[i]]
            train_nir_clabel[tmp_nir] = i

        self.train_rgb_image = train_rgb_image
        self.train_rgb_label = train_rgb_label
        self.train_rgb_clabel = train_rgb_clabel
        self.train_rgb_name = train_rgb_name

        # self.train_gray_image = train_gray_image
        # self.train_gray_label = train_gray_label
        # self.train_gray_clabel = train_gray_clabel
        # self.train_gray_name = train_gray_name

        self.train_nir_image = train_nir_image
        self.train_nir_label = train_nir_label
        self.train_nir_clabel = train_nir_clabel
        self.train_nir_name = train_nir_name

        self.transform_rgb = transform_rgb
        self.transform_gray = transform_gray
        self.transform_nir = transform_nir

    def __getitem__(self, index):

        rgb_idx = index[0]
        # gray_idx = index[1]
        nir_idx = index[1]

        img1, target1 = self.train_rgb_image[rgb_idx], self.train_rgb_label[rgb_idx]

        # img2, target2 = self.train_gray_image[rgb_idx], self.train_gray_label[rgb_idx]

        img3, target3 = self.train_nir_image[nir_idx], self.train_nir_label[nir_idx]

        target1 = int(target1)
        # target2 = int(target2)
        target3 = int(target3)

        img1 = self.transform_rgb(img1)  # color
        # img2 = self.transform_nir(img2)  # gray
        img3 = self.transform_nir(img3)  # nir

        name1 = self.train_rgb_name[rgb_idx]
        # name2 = self.train_gray_name[gray_idx]
        name3 = self.train_nir_name[nir_idx]

        return img1, img3, target1, target3, name1, name3

    def __len__(self):
        # print(len(self.train_color_label))
        return len(self.train_rgb_label)
