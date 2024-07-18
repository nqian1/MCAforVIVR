import os
import numpy as np
from torch.utils.data.sampler import Sampler
import sys
import os.path as osp
import torch
from collections import defaultdict
import copy
import random
import math




def load_data(input_data_path ):
    with open(input_data_path) as f:
        data_file_list = open(input_data_path, 'rt').read().splitlines()
        # Get full list of color image and labels
        file_image = [s.split(' ')[0] for s in data_file_list]
        file_label = [int(s.split(' ')[1]) for s in data_file_list]
        
    return file_image, file_label
    

def imgname(dir_path):
    vid_container = set()
    for vid in os.listdir(dir_path):
        vid_container.add(int(vid))
    # vid2label = {vid: label for label, vid in enumerate(vid_container)}

    # dataset = []
    imgname = []
    for vid in os.listdir(dir_path):
        vid_path = osp.join(dir_path, vid)
        r_data = os.listdir(osp.join(vid_path, 'vis'))
        for img in r_data:
            imgname.append(img)

    return imgname




def GenIdx( train_color_label, train_thermal_label):
    color_pos = []
    unique_label_color = np.unique(train_color_label)
    for i in range(len(unique_label_color)):
        tmp_pos = [k for k, v in enumerate(train_color_label) if v == unique_label_color[i]]
        color_pos.append(tmp_pos)

    thermal_pos = []
    unique_label_thermal = np.unique(train_thermal_label)
    for i in range(len(unique_label_thermal)):
        tmp_pos = [k for k, v in enumerate(train_thermal_label) if v == unique_label_thermal[i]]
        thermal_pos.append(tmp_pos)
    return color_pos, thermal_pos
    
def GenCamIdx(gall_img, gall_label, mode):
    if mode =='indoor':
        camIdx = [1,2]
    else:
        camIdx = [1,2,4,5]
    gall_cam = []
    for i in range(len(gall_img)):
        gall_cam.append(int(gall_img[i][-10]))
    
    sample_pos = []
    unique_label = np.unique(gall_label)
    for i in range(len(unique_label)):
        for j in range(len(camIdx)):
            id_pos = [k for k,v in enumerate(gall_label) if v==unique_label[i] and gall_cam[k]==camIdx[j]]
            if id_pos:
                sample_pos.append(id_pos)
    return sample_pos
    
def ExtractCam(gall_img):
    gall_cam = []
    for i in range(len(gall_img)):
        cam_id = int(gall_img[i][-10])
        # if cam_id ==3:
            # cam_id = 2
        gall_cam.append(cam_id)
    
    return np.array(gall_cam)
    
class IdentitySampler(Sampler):
    """Sample person identities evenly in each batch.
        Args:
            train_color_label, train_thermal_label: labels of two modalities
            color_pos, thermal_pos: positions of each identity
            batchSize: batch size
    """

    def __init__(self, train_color_label, train_thermal_label, color_pos, thermal_pos, num_pos, batchSize, epoch):        
        uni_label = np.unique(train_color_label)

        print(len(train_color_label),len(color_pos))
        print(len(train_thermal_label),len(thermal_pos))
        # print(uni_label)
        np.random.shuffle(uni_label)
        # print(uni_label)
        self.n_classes = len(uni_label)



        for i in range(len(color_pos)):
            t = color_pos[i]
            # print(t)
            np.random.shuffle(t)
            # print(t)
            color_pos[i] = t

        for i in range(len(thermal_pos)):
            t = thermal_pos[i]
            np.random.shuffle(t)
            thermal_pos[i] = t



        N = np.maximum(len(train_color_label), len(train_thermal_label))
        # N = np.minimum(len(train_color_label), len(train_thermal_label))
        for j in range(int(N/(batchSize*num_pos))+1):

            batch_idx = np.random.choice(uni_label, batchSize, replace=False)
            # if (j+1)*batchSize>len(uni_label):
            #     batch_idx = np.random.choice(uni_label, batchSize, replace=False)
            # else:
            #     batch_idx = uni_label[j*batchSize:(j+1)*batchSize]




            for i in range(batchSize):
                sample_color  = np.random.choice(color_pos[batch_idx[i]], num_pos,replace=True)
                sample_thermal = np.random.choice(thermal_pos[batch_idx[i]], num_pos,replace=True)

                if j ==0 and i==0:
                    index1= sample_color
                    index2= sample_thermal
                else:
                    index1 = np.hstack((index1, sample_color))
                    index2 = np.hstack((index2, sample_thermal))



        self.index1 = index1
        self.index2 = index2
        self.N  = N
        
    def __iter__(self):

        return iter(np.arange(len(self.index1)))

    def __len__(self):
        return self.N


class RandomErasing(object):
    """ Randomly selects a rectangle region in an image and erases its pixels.
        'Random Erasing Data Augmentation' by Zhong et al.
        See https://arxiv.org/pdf/1708.04896.pdf
    Args:
         probability: The probability that the Random Erasing operation will be performed.
         sl: Minimum proportion of erased area against input image.
         sh: Maximum proportion of erased area against input image.
         r1: Minimum aspect ratio of erased area.
         mean: Erasing value.
    """

    def __init__(self, probability=0.5, sl=0.02, sh=0.4, r1=0.3, mean=[0.4914, 0.4822, 0.4465]):
        self.probability = probability
        self.mean = mean
        self.sl = sl
        self.sh = sh
        self.r1 = r1

    def __call__(self, img):

        if random.uniform(0, 1) > self.probability:
            return img

        for attempt in range(100):
            area = img.size()[1] * img.size()[2]

            target_area = random.uniform(self.sl, self.sh) * area
            aspect_ratio = random.uniform(self.r1, 1 / self.r1)

            h = int(round(math.sqrt(target_area * aspect_ratio)))
            w = int(round(math.sqrt(target_area / aspect_ratio)))

            if w < img.size()[2] and h < img.size()[1]:
                x1 = random.randint(0, img.size()[1] - h)
                y1 = random.randint(0, img.size()[2] - w)
                if img.size()[0] == 3:
                    img[0, x1:x1 + h, y1:y1 + w] = self.mean[0]
                    img[1, x1:x1 + h, y1:y1 + w] = self.mean[1]
                    img[2, x1:x1 + h, y1:y1 + w] = self.mean[2]
                else:
                    img[0, x1:x1 + h, y1:y1 + w] = self.mean[0]
                return img

        return img



class RandomIdentitySampler_DM(Sampler):
    """
    Randomly sample N identities, then for each identity,
    randomly sample K instances, therefore batch size is N*K.
    Args:
    - data_source (list): list of (img_path, pid, camid).
    - num_instances (int): number of instances per identity in a batch.
    - batch_size (int): number of examples in a batch.
    """

    def __init__(self, train_color_label, train_thermal_label, color_pos, thermal_pos, num_pos, batchSize, epoch):
        # print('...........................')
        self.batch_size = batchSize*num_pos
        # self.num_instances = num_pos*2
        self.num_instances_one_modal= num_pos
        self.num_pids_per_batch = batchSize
        # print(self.batch_size,self.num_pids_per_batch,self.num_instances_one_modal)



        self.index_dic_vis = defaultdict(list) #dict with list value
        #{783: [0, 5, 116, 876, 1554, 2041],...,}
        for i in range(len(color_pos)):
            t=np.array(color_pos[i])
            # print(t.shape)
            self.index_dic_vis[i].append(t)
        # print(self.index_dic_vis)
        self.pids_vis = list(self.index_dic_vis.keys())
        # print('*******',len(self.pids_vis))



        self.index_dic_the = defaultdict(list)
        for i in range(len(thermal_pos)):
            t = np.array(thermal_pos[i])
            self.index_dic_the[i].append(t)
        self.pids_the = list(self.index_dic_the.keys())





        # estimate number of examples in an epoch
        self.length_vis = 0
        for pid in self.pids_vis:
            idxs = self.index_dic_vis[pid]
            num = len(idxs)
            if num < self.num_instances_one_modal:
                num = self.num_instances_one_modal
            self.length_vis += num - num % self.num_instances_one_modal



        self.length_the = 0
        for pid in self.pids_the:
            idxs = self.index_dic_the[pid]
            num = len(idxs)
            if num < self.num_instances_one_modal:
                num = self.num_instances_one_modal
                # print('.....',pid)
            self.length_the += num - num % self.num_instances_one_modal


        self.length = min(self.length_the,self.length_vis)
        # print('........................................',self.length_the,self.length_vis)
        self.pids = list(set(self.pids_vis).intersection(set(self.pids_the)))
        # print(len(self.pids))






    def __iter__(self):

        batch_idxs_dict_vis = defaultdict(list)
        batch_idxs_dict_the = defaultdict(list)

        for pid in self.pids:

            idxs_vis = np.array(copy.deepcopy(self.index_dic_vis[pid])).squeeze()
            # print(np.array(idxs_vis).squeeze())
            # print(idxs_vis[])
            idxs_the = np.array(copy.deepcopy(self.index_dic_the[pid])).squeeze()

            if len(idxs_vis) < self.num_instances_one_modal:
                idxs_vis = np.random.choice(idxs_vis, size=self.num_instances_one_modal, replace=True)

            if len(idxs_the) < self.num_instances_one_modal:
                idxs_the = np.random.choice(idxs_the, size=self.num_instances_one_modal, replace=True)

            random.shuffle(idxs_vis)
            random.shuffle(idxs_the)
            batch_idxs_vis = []
            for idx in idxs_vis:
                batch_idxs_vis.append(idx)
                if len(batch_idxs_vis) == self.num_instances_one_modal:
                    batch_idxs_dict_vis[pid].append(batch_idxs_vis)
                    batch_idxs_vis = []

            batch_idxs_the = []
            for idx in idxs_the:
                batch_idxs_the.append(idx)
                if len(batch_idxs_the) == self.num_instances_one_modal:
                    batch_idxs_dict_the[pid].append(batch_idxs_the)
                    batch_idxs_the = []

        # avai_pids = copy.deepcopy(self.pids)
        # # final_idxs = []
        # vis_final_idxs = []
        # the_final_idxs = []
        # while len(avai_pids) >= self.num_pids_per_batch:
        #     selected_pids = random.sample(avai_pids, self.num_pids_per_batch)
        #     for pid in selected_pids:
        #         batch_idxs_vis = np.array(batch_idxs_dict_vis[pid].pop(0)).reshape(1, -1)
        #         batch_idxs_the = np.array(batch_idxs_dict_the[pid].pop(0)).reshape(1, -1)
        #         # print(batch_idxs_vis)
        #         vis_final_idxs.extend(batch_idxs_vis)
        #         the_final_idxs.extend(batch_idxs_the)
        #         if len(batch_idxs_dict_vis[pid]) == 0 or len(batch_idxs_dict_the[pid]) == 0:
        #             avai_pids.remove(pid)
        #
        # self.index1 = np.array(vis_final_idxs)
        # # print(len(self.index1))
        # self.index2 = the_final_idxs
        #
        # return iter(np.arange(len(self.index1)))

        avai_pids = copy.deepcopy(self.pids)
        final_idxs = []
        while len(avai_pids) >= self.num_pids_per_batch:
            selected_pids = random.sample(avai_pids, self.num_pids_per_batch)
            for pid in selected_pids:
                batch_idxs_vis = np.array(batch_idxs_dict_vis[pid].pop(0)).reshape(1, -1)
                batch_idxs_the = np.array(batch_idxs_dict_the[pid].pop(0)).reshape(1, -1)
                # print(batch_idxs_vis)
                X = np.concatenate((batch_idxs_vis, batch_idxs_the), axis=0).transpose()
                # print(X)
                final_idxs.extend(X)
                # final_idxs.extend([batch_idxs_vis,batch_idxs_the])
                if len(batch_idxs_dict_vis[pid]) == 0 or len(batch_idxs_dict_the[pid]) == 0:
                    avai_pids.remove(pid)
        # print('xxxxxxxx',len(final_idxs))
        # print(final_idxs)
        return iter(final_idxs)

    def __len__(self):
        return self.length



class RandomIdentitySampler_TM(Sampler):
    """
    Randomly sample N identities, then for each identity,
    randomly sample K instances, therefore batch size is N*K.
    Args:
    - data_source (list): list of (img_path, pid, camid).
    - num_instances (int): number of instances per identity in a batch.
    - batch_size (int): number of examples in a batch.
    """

    def __init__(self, train_color_label,train_gray_label, train_thermal_label, color_pos, gray_pos, thermal_pos, num_pos, batchSize, epoch):
        # print('...........................')
        self.batch_size = batchSize*num_pos
        self.num_instances_one_modal= num_pos
        self.num_pids_per_batch = batchSize



        self.index_dic_vis = defaultdict(list) #dict with list value
        for i in range(len(color_pos)):
            t=np.array(color_pos[i])
            self.index_dic_vis[i].append(t)
        self.pids_vis = list(self.index_dic_vis.keys())




        self.index_dic_gray = defaultdict(list) #dict with list value
        for i in range(len(gray_pos)):
            t=np.array(gray_pos[i])
            self.index_dic_gray[i].append(t)
        self.pids_gray = list(self.index_dic_gray.keys())




        self.index_dic_the = defaultdict(list)
        for i in range(len(thermal_pos)):
            t = np.array(thermal_pos[i])
            self.index_dic_the[i].append(t)
        self.pids_the = list(self.index_dic_the.keys())



        # estimate number of examples in an epoch
        self.length_vis = 0
        for pid in self.pids_vis:
            idxs = self.index_dic_vis[pid]
            num = len(idxs)
            if num < self.num_instances_one_modal:
                num = self.num_instances_one_modal
            self.length_vis += num - num % self.num_instances_one_modal

        # estimate number of examples in an epoch
        self.length_gray = 0
        for pid in self.pids_gray:
            idxs = self.index_dic_gray[pid]
            num = len(idxs)
            if num < self.num_instances_one_modal:
                num = self.num_instances_one_modal
            self.length_gray += num - num % self.num_instances_one_modal

        # estimate number of examples in an epoch
        self.length_the = 0
        for pid in self.pids_the:
            idxs = self.index_dic_the[pid]
            num = len(idxs)
            if num < self.num_instances_one_modal:
                num = self.num_instances_one_modal
            self.length_the += num - num % self.num_instances_one_modal


        self.length = min(self.length_the,self.length_vis, self.length_gray)
        self.pids = list(set(self.pids_vis).intersection(set(self.pids_the).intersection(set(self.pids_gray))))












    def __iter__(self):

        batch_idxs_dict_vis = defaultdict(list)
        batch_idxs_dict_gray = defaultdict(list)
        batch_idxs_dict_the = defaultdict(list)

        for pid in self.pids:

            idxs_vis = np.array(copy.deepcopy(self.index_dic_vis[pid])).squeeze()
            idxs_gray = np.array(copy.deepcopy(self.index_dic_gray[pid])).squeeze()
            idxs_the = np.array(copy.deepcopy(self.index_dic_the[pid])).squeeze()

            if len(idxs_vis) < self.num_instances_one_modal:
                idxs_vis = np.random.choice(idxs_vis, size=self.num_instances_one_modal, replace=True)


            if len(idxs_gray) < self.num_instances_one_modal:
                idxs_gray = np.random.choice(idxs_gray, size=self.num_instances_one_modal, replace=True)


            if len(idxs_the) < self.num_instances_one_modal:
                idxs_the = np.random.choice(idxs_the, size=self.num_instances_one_modal, replace=True)

            random.shuffle(idxs_vis)
            random.shuffle(idxs_gray)
            random.shuffle(idxs_the)
            batch_idxs_vis = []
            for idx in idxs_vis:
                batch_idxs_vis.append(idx)
                if len(batch_idxs_vis) == self.num_instances_one_modal:
                    batch_idxs_dict_vis[pid].append(batch_idxs_vis)
                    batch_idxs_vis = []

            batch_idxs_gray = []
            for idx in idxs_gray:
                batch_idxs_gray.append(idx)
                if len(batch_idxs_gray) == self.num_instances_one_modal:
                    batch_idxs_dict_gray[pid].append(batch_idxs_gray)
                    batch_idxs_gray = []



            batch_idxs_the = []
            for idx in idxs_the:
                batch_idxs_the.append(idx)
                if len(batch_idxs_the) == self.num_instances_one_modal:
                    batch_idxs_dict_the[pid].append(batch_idxs_the)
                    batch_idxs_the = []



        avai_pids = copy.deepcopy(self.pids)
        final_idxs = []
        while len(avai_pids) >= self.num_pids_per_batch:
            selected_pids = random.sample(avai_pids, self.num_pids_per_batch)
            for pid in selected_pids:
                batch_idxs_vis = np.array(batch_idxs_dict_vis[pid].pop(0)).reshape(1, -1)
                batch_idxs_gray = np.array(batch_idxs_dict_gray[pid].pop(0)).reshape(1, -1)
                batch_idxs_the = np.array(batch_idxs_dict_the[pid].pop(0)).reshape(1, -1)

                # print(batch_idxs_vis)
                X = np.concatenate((batch_idxs_vis, batch_idxs_gray, batch_idxs_the), axis=0).transpose()
                final_idxs.extend(X)
                if len(batch_idxs_dict_vis[pid]) == 0 or len(batch_idxs_dict_gray[pid]) == 0 or len(batch_idxs_dict_the[pid]) == 0:
                    avai_pids.remove(pid)


        return iter(final_idxs)

    def __len__(self):
        return self.length


class AverageMeter(object):
    """Computes and stores the average and current value""" 
    def __init__(self):
        self.reset()
                   
    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0 

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

import errno
def mkdir_if_missing(directory):
    if not osp.exists(directory):
        try:
            os.makedirs(directory)
        except OSError as e:
            if e.errno != errno.EEXIST:
                raise   
class Logger(object):
    """
    Write console output to external text file.
    Code imported from https://github.com/Cysu/open-reid/blob/master/reid/utils/logging.py.
    """  
    def __init__(self, fpath=None):
        self.console = sys.stdout
        self.file = None
        if fpath is not None:
            mkdir_if_missing(osp.dirname(fpath))
            self.file = open(fpath, 'a')

    def __del__(self):
        self.close()

    def __enter__(self):
        pass

    def __exit__(self, *args):
        self.close()

    def write(self, msg):
        self.console.write(msg)
        if self.file is not None:
            self.file.write(msg)

    def flush(self):
        self.console.flush()
        if self.file is not None:
            self.file.flush()
            os.fsync(self.file.fileno())

    def close(self):
        self.console.close()
        if self.file is not None:
            self.file.close()
            
def set_seed(seed, cuda=True):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if cuda:
        torch.cuda.manual_seed(seed)

def set_requires_grad(nets, requires_grad=False):
            """Set requies_grad=Fasle for all the networks to avoid unnecessary computations
            Parameters:
                nets (network list)   -- a list of networks
                requires_grad (bool)  -- whether the networks require gradients or not
            """
            if not isinstance(nets, list):
                nets = [nets]
            for net in nets:
                if net is not None:
                    for param in net.parameters():
                        param.requires_grad = requires_grad


import os
import torch
import numpy as np
from PIL import Image

import torch.nn.functional as F


def normalize(x):
    return x.mul_(2).add_(-1)


def same_padding(images, ksizes, strides, rates):
    assert len(images.size()) == 4
    batch_size, channel, rows, cols = images.size()
    out_rows = (rows + strides[0] - 1) // strides[0]
    out_cols = (cols + strides[1] - 1) // strides[1]
    effective_k_row = (ksizes[0] - 1) * rates[0] + 1
    effective_k_col = (ksizes[1] - 1) * rates[1] + 1
    padding_rows = max(0, (out_rows - 1) * strides[0] + effective_k_row - rows)
    padding_cols = max(0, (out_cols - 1) * strides[1] + effective_k_col - cols)
    # Pad the input
    padding_top = int(padding_rows / 2.)
    padding_left = int(padding_cols / 2.)
    padding_bottom = padding_rows - padding_top
    padding_right = padding_cols - padding_left
    paddings = (padding_left, padding_right, padding_top, padding_bottom)
    images = torch.nn.ZeroPad2d(paddings)(images)
    return images


def extract_image_patches(images, ksizes, strides, rates, padding='same'):
    """
    Extract patches from images and put them in the C output dimension.
    :param padding:
    :param images: [batch, channels, in_rows, in_cols]. A 4-D Tensor with shape
    :param ksizes: [ksize_rows, ksize_cols]. The size of the sliding window for
     each dimension of images
    :param strides: [stride_rows, stride_cols]
    :param rates: [dilation_rows, dilation_cols]
    :return: A Tensor
    """
    assert len(images.size()) == 4
    assert padding in ['same', 'valid']
    batch_size, channel, height, width = images.size()

    if padding == 'same':
        images = same_padding(images, ksizes, strides, rates)
    elif padding == 'valid':
        pass
    else:
        raise NotImplementedError('Unsupported padding type: {}.\
                Only "same" or "valid" are supported.'.format(padding))

    unfold = torch.nn.Unfold(kernel_size=ksizes,
                             dilation=rates,
                             padding=0,
                             stride=strides)
    patches = unfold(images)
    return patches  # [N, C*k*k, L], L is the total number of such blocks


def reduce_mean(x, axis=None, keepdim=False):
    if not axis:
        axis = range(len(x.shape))
    for i in sorted(axis, reverse=True):
        x = torch.mean(x, dim=i, keepdim=keepdim)
    return x


def reduce_std(x, axis=None, keepdim=False):
    if not axis:
        axis = range(len(x.shape))
    for i in sorted(axis, reverse=True):
        x = torch.std(x, dim=i, keepdim=keepdim)
    return x


def reduce_sum(x, axis=None, keepdim=False):
    if not axis:
        axis = range(len(x.shape))
    for i in sorted(axis, reverse=True):
        x = torch.sum(x, dim=i, keepdim=keepdim)
    return x