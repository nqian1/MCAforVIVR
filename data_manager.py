from __future__ import print_function, absolute_import
import os
import numpy as np
import random
import numpy as np
from PIL import Image


def process_eval_rgbnt(data_path, eval_mode='v2t', trial = 0, relabel=False,imgh=128,imgw=128):
    random.seed(trial)

    data_dir = '/media/jqzhu/941A7DD31A7DB33A/ZQQ/datasets/RGBN300_autolabel'
    # data_dir = '/media/jqzhu/e/qqzhao/datasets/RGBN300_autolabel'
    # data_dir = '/media/l/b/qqzhao/datasets/AHU_NIRTIR/RGBN300'
    print(imgh,imgw)

    test_rgb_path = data_dir + '/R/'
    test_ni_path = data_dir + '/N/'

    test_rgb_list_all = os.listdir(test_rgb_path)  # all id in the rgb part of the whole dataset
    test_nir_list_all = os.listdir(test_ni_path)  # all id in the nir part of the whole dataset

    #############################select odd to construct testing set#################
    # print(test_rgb_list, test_nir_list)
    test_rgb_list = []  # select odd
    for i in range(len(test_rgb_list_all)):
        cur_id = test_rgb_list_all[i]
        cur_id = int(cur_id)  # str2num
        if cur_id % 2 == 0:
            # print(cur_id)
            test_rgb_list.append(test_rgb_list_all[i])
    # print(len(test_rgb_list))
    test_nir_list = []  # select odd
    for i in range(len(test_nir_list_all)):
        cur_id = test_nir_list_all[i]
        cur_id = int(cur_id)  # str2num
        if cur_id % 2 == 0:
            # print(cur_id)
            test_nir_list.append(test_nir_list_all[i])

    test_rgb_name = []
    test_rgb_image = []
    for i in range(len(test_rgb_list)):
        cur_id_path = test_rgb_path + test_rgb_list[i]
        cur_id_list = os.listdir(cur_id_path)
        # print(cur_id_list)
        for j in range(len(cur_id_list)):
            imgname = cur_id_path + '/' + cur_id_list[j]
            if os.path.splitext(imgname)[1] == '.jpg':
                img = Image.open(imgname)
                img = img.resize((imgh, imgw), Image.ANTIALIAS)
                pix_array = np.array(img)
                test_rgb_image.append(pix_array)
                test_rgb_name.append(cur_id_list[j])

    test_rgb_image = np.array(test_rgb_image)
    # print(test_rgb_image.shape)

    test_nir_name = []
    test_nir_image = []
    for i in range(len(test_nir_list)):
        cur_id_path = test_ni_path + test_nir_list[i]
        cur_id_list = os.listdir(cur_id_path)
        for j in range(len(cur_id_list)):
            imgname = cur_id_path + '/' + cur_id_list[j]
            # print(imgname)
            if os.path.splitext(imgname)[1] == '.jpg':
                img = Image.open(imgname)
                img = img.resize((imgh, imgw), Image.ANTIALIAS)
                pix_array = np.array(img)
                test_nir_image.append(pix_array)
                test_nir_name.append(cur_id_list[j])

    test_nir_image = np.array(test_nir_image)

    ################################process person id: tranform 0599 to 149.0##########################
    test_rgb_id = [s.split('_')[0] for s in test_rgb_name]
    test_nir_id = [s.split('_')[0] for s in test_nir_name]
    test_twomodal_id = test_rgb_id + test_nir_id
    unique_id = np.unique(test_twomodal_id)
    test_rgb_label = np.ones(len(test_rgb_id)) * -1
    test_nir_label = np.ones(len(test_nir_id)) * -1
    for i in range(len(unique_id)):
        tmp_rgb = [k for k, v in enumerate(test_rgb_id) if v == unique_id[i]]
        test_rgb_label[tmp_rgb] = i

        tmp_nir = [k for k, v in enumerate(test_nir_id) if v == unique_id[i]]
        test_nir_label[tmp_nir] = i

    ################################process camera id: tranform c0001 to 0.0 ##########################
    test_rgb_cid = [s.split('_')[1][1:] for s in test_rgb_name]
    test_nir_cid = [s.split('_')[1][1:] for s in test_nir_name]
    test_twomodal_cid = test_rgb_cid + test_nir_cid
    unique_cid = np.unique(test_twomodal_cid)
    test_rgb_clabel = np.ones(len(test_rgb_cid)) * -2
    test_nir_clabel = np.ones(len(test_nir_cid)) * -2
    for i in range(len(unique_cid)):
        tmp_rgb = [k for k, v in enumerate(test_rgb_cid) if v == unique_cid[i]]
        test_rgb_clabel[tmp_rgb] = i
        tmp_nir = [k for k, v in enumerate(test_nir_cid) if v == unique_cid[i]]
        test_nir_clabel[tmp_nir] = i



    if eval_mode=='v2t':
        #v--probe/query t---gallery
        print('.....................v2t...................')
        gallery_img = test_nir_image
        gallery_label = test_nir_label
        gallery_clabel = test_nir_clabel
        gallery_name = test_nir_name
        # print(gallery_name[0],  gallery_label[0],  gallery_clabel[0])
        # print(gallery_name[10], gallery_label[10], gallery_clabel[10])
        # print(gallery_name[20], gallery_label[20], gallery_clabel[20])
        # print(gallery_name[30], gallery_label[30], gallery_clabel[30])

        #all vimage applied as probe/query
        query_img = test_rgb_image
        query_label = test_rgb_label
        query_clabel = test_rgb_clabel
        query_name = test_rgb_name


        print('gallery visible', len(gallery_label))

        print('query visible', len(query_label))
        # #shuffle
        # index = [i for i in range(len(test_rgb_name))]
        # np.random.shuffle(index)
        # query_img = query_img[index,:,:,:]
        # query_label = query_label[index]
        # query_clabel = query_clabel[index]
        # query_name_=[]
        # for i in range(len(test_rgb_name)):
        #     query_name_.append(query_name[index[i]])
        # print('shufule',query_img.shape,query_label.shape)
        #
        # #sample
        # sampleidx= [i for i in range(0, len(test_rgb_name), 10)]
        # query_img = query_img[sampleidx,:,:,:]
        # query_label = query_label[sampleidx]
        # query_clabel = query_clabel[sampleidx]
        # query_name__ = []
        # for i in range(len(sampleidx)):
        #     query_name__.append(query_name_[sampleidx[i]])
        #
        #
        # print('sample',query_img.shape,query_label.shape)
        #
        # print(query_name__[0],query_label[0],query_clabel[0])
        # print(query_name__[10], query_label[10], query_clabel[10])
        # print(query_name__[20], query_label[20], query_clabel[20])
        # print(query_name__[30], query_label[30], query_clabel[30])

        return query_img, query_label, query_clabel, gallery_img, gallery_label,gallery_clabel

    elif eval_mode=='t2v': #t2v
        # t--probe/query v---gallery
        print('.....................t2v...................')
        gallery_img = test_rgb_image
        gallery_label = test_rgb_label
        gallery_clabel = test_rgb_clabel
        gallery_name = test_rgb_name

        # all nir applied as probe/query
        query_img = test_nir_image
        query_label = test_nir_label
        query_clabel = test_nir_clabel
        query_name = test_nir_name


        print('gallery visible', len(gallery_label))

        print('query visible', len(query_label))
        # # shuffle
        # index = [i for i in range(len(test_nir_name))]
        # np.random.shuffle(index)
        # query_img = query_img[index, :, :, :]
        # query_label = query_label[index]
        # query_clabel = query_clabel[index]
        # query_name_=[]
        # for i in range(len(test_nir_name)):
        #     query_name_.append(query_name[index[i]])
        # print(query_img.shape, query_label.shape)
        #
        # # sample
        # sampleidx = [i for i in range(0, len(test_nir_name), 10)]
        # query_img = query_img[sampleidx, :, :, :]
        # query_label = query_label[sampleidx]
        # query_clabel = query_clabel[sampleidx]
        # query_name__ = []
        # for i in range(len(sampleidx)):
        #     query_name__.append(query_name_[index[i]])
        #
        # print(query_img.shape, query_label.shape)
        #
        #

        return query_img, query_label, query_clabel, gallery_img, gallery_label, gallery_clabel

    elif eval_mode=='mix': #t2v
        # t--probe/query v---gallery
        print('.....................mix..................')

        unilabel = np.unique(np.concatenate((test_rgb_label,test_nir_label),axis=0))

        gallery_idx_part1 = []
        gallery_idx_part2 = []
        query_idx_part1 = []
        query_idx_part2 = []
        for i in range(0,len(unilabel)):
            curlabel =  unilabel[i]
            rgbfullidx = np.arange(len(test_rgb_label))
            rgbidx = rgbfullidx[test_rgb_label==curlabel]
            for j in range(len(rgbidx)):
                if j % 2 == 0:
                    gallery_idx_part1.append(rgbidx[j])
                else:
                    query_idx_part1.append(rgbidx[j])

            nirfullidx = np.arange(len(test_nir_label))
            niridx = nirfullidx[test_nir_label==curlabel]
            for j in range(len(niridx)):
                if j % 2 == 0:
                    gallery_idx_part2.append(niridx[j])
                else:
                    query_idx_part2.append(niridx[j])


        gallery_idx_part1 = np.array(gallery_idx_part1)
        gallery_idx_part2 = np.array(gallery_idx_part2)
        query_idx_part1 = np.array(query_idx_part1)
        query_idx_part2 = np.array(query_idx_part2)


        gallery_img = np.concatenate((test_rgb_image[gallery_idx_part1,:,:,:],test_nir_image[gallery_idx_part2,:,:,:]),axis=0)
        gallery_label = np.concatenate((test_rgb_label[gallery_idx_part1],test_nir_label[gallery_idx_part2]),axis=0)#
        gallery_clabel = np.concatenate((test_rgb_clabel[gallery_idx_part1],test_nir_clabel[gallery_idx_part2]),axis=0)#

        gallery_name = []
        for i,v in enumerate(gallery_idx_part1):
            gallery_name.append(test_rgb_name[v])
        for i,v in enumerate(gallery_idx_part2):
            gallery_name.append(test_nir_name[v])


        # print(gallery_img.shape,gallery_label.shape,gallery_clabel.shape,len(gallery_name))

        # print(gallery_idx_part1(0),gallery_idx_part2.size(0))
        # for i in range(gallery_idx_part1.size(0)):
        #     gallery_name.append(test_rgb_name[gallery_idx_part1[i]])
        # for i in range(gallery_idx_part2.size(0)):
        #     gallery_name.append(test_nir_name[gallery_idx_part2[i]])


        query_img = np.concatenate((test_rgb_image[query_idx_part1,:,:,:],test_nir_image[query_idx_part2,:,:,:]),axis=0)
        query_label = np.concatenate((test_rgb_label[query_idx_part1],test_nir_label[query_idx_part2]),axis=0)#
        query_clabel = np.concatenate((test_rgb_clabel[query_idx_part1],test_nir_clabel[query_idx_part2]),axis=0)#


        query_name = []
        for i, v in enumerate(query_idx_part1):
            query_name.append(test_rgb_name[v])
        for i, v in enumerate(query_idx_part2):
            query_name.append(test_nir_name[v])

        # print(len(query_name))

        return query_img, query_label, query_clabel, gallery_img, gallery_label, gallery_clabel




    # 
    # if rgbnt_mode == 'rgb_ni' or rgbnt_mode == 'rgb_ti':
    #     query_camera = 'RGB'
    # elif rgbnt_mode == 'ni_ti' or rgbnt_mode == 'ni_rgb':
    #     query_camera = 'NI'
    # elif rgbnt_mode == 'ti_ni' or rgbnt_mode == 'ti_rgb':
    #     query_camera = 'TI'
    # 
    # test_query_path = data_path + 'test/' + query_camera + '/'
    # files_query_all = os.listdir(test_query_path)
    # files_query_all_id = [s.split('_')[0] for s in files_query_all]
    # files_query_all_img = [test_query_path + i for i in files_query_all]
    # 
    # files_query = []
    # for new_id in list(set(files_query_all_id)):
    #     new_id_file_list = [s for s in files_query_all if s.split('_')[0] == new_id]
    #     id_random_query = random.sample(new_id_file_list, 10)
    #     id_random_query = random.sample(new_id_file_list, 10)
    #     for i in id_random_query:
    #         files_query.append(i)
    # 
    # files_query_id = []
    # files_query_img =[]
    # for s in files_query:
    #     files_query_id.append(s.split('_')[0])
    #     files_query_img.append(test_query_path + s)
    #     # print(files_query_id)
    #     # print(files_query_img)
    #     # exit()
    # 
    # 
    # return files_query_img, np.array(files_query_id)

