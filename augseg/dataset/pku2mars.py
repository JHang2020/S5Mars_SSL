import json
import os
from collections import namedtuple
import random
from scipy.cluster.vq import *
import torch
import torch.utils.data as data
from skimage.feature import canny,hog
from torch.nn import functional as F
from PIL import Image, ImageFilter
import numpy as np
import cv2
from pycocotools import mask as mask_utils
import sys,os

import torch.distributed as dist
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
import copy
from . import augs_TIBA as img_trsform
from .augs_ALIA import object_augment
from .base import BaseDataset

# https://pytorch.org/docs/stable/notes/randomness.html
def seed_worker(worker_id):
    cur_seed = np.random.get_state()[1][0]
    cur_seed += worker_id
    np.random.seed(cur_seed)
    random.seed(cur_seed)


class PKU2MARS(data.Dataset):
    """Cityscapes <http://www.cityscapes-dataset.com/> Dataset.
    
    **Parameters:**
        - **root** (string): Root directory of dataset where directory 'leftImg8bit' and 'gtFine' or 'gtCoarse' are located.
        - **split** (string, optional): The image split to use, 'train', 'test' or 'val' if mode="gtFine" otherwise 'train', 'train_extra' or 'val'
        - **mode** (string, optional): The quality mode to use, 'gtFine' or 'gtCoarse' or 'color'. Can also be a list to output a tuple with all specified target types.
        - **transform** (callable, optional): A function/transform that takes in a PIL image and returns a transformed version. E.g, ``transforms.RandomCrop``
        - **target_transform** (callable, optional): A function/transform that takes in the target and transforms it.
    """

    # Based on https://github.com/mcordts/cityscapesScripts
    Ai4MarsClass = namedtuple('Ai4MarsClass', ['name', 'id', 'train_id', 'category', 'category_id',
                                                     'has_instances', 'ignore_in_eval', 'color'])
    classes = [
        Ai4MarsClass('NULL',                0, 255, 'void', 0, False, True, (0, 0, 0)),
        Ai4MarsClass('sky',                 1, 0, 'nature', 1, True, False, (107, 142, 35)),
        Ai4MarsClass('ridge',               2, 1, 'nature', 1, True, False, (128, 64, 128)),
        Ai4MarsClass('soil',                3, 2, 'nature', 1, True, False, (220, 20, 60)),
        Ai4MarsClass('sand',                4, 3, 'nature', 1, True, False, (152, 251, 152)),
        Ai4MarsClass('bedrock',             5, 4, 'nature', 1, True, False, (119, 11, 32)),
        Ai4MarsClass('rock',                6, 5, 'nature', 1, True, False, (20, 60, 60)),
        Ai4MarsClass('rover',               7, 6, 'nature', 1, True, False, (15, 51, 252)),
        Ai4MarsClass('trace',               8, 7, 'nature', 1, True, False, (22, 220, 60)),
        Ai4MarsClass('hole',                9, 8, 'nature', 1, True, False, (152, 21, 12)),
    ]

    train_id_to_color = [c.color for c in classes if (c.train_id != -1 and c.train_id != 255)]
    train_id_to_color.append([0, 0, 0])
    train_id_to_color = np.array(train_id_to_color)
    id_to_train_id = np.array([c.train_id for c in classes])

    def __init__(self, split='train',trs_form=None,trs_form_strong=None,flag_semi=False,idx_list=None):
        if 'train' in split:
            split = 'new_train'
        else:
            split = 'new_test'

        self.images_dir = "/mnt/netdisk/wangwenjing/Datasets/MarsData/Final/images"
        self.targets_dir = "/mnt/netdisk/wangwenjing/Datasets/MarsData/Final/labels"
        self.transform_weak = trs_form
        self.transform_strong = trs_form_strong
        self.mask_dir = '/mnt/netdisk/Datasets/093-SAM_MarsMask' #SAM output mask
        self.trf_normalize = self._get_to_tensor_and_normalize(mean=[0.625, 0.519, 0.363,], std= [0.01803789, 0.01469946, 0.00991374])
        self.flag_semi = flag_semi
        self.crop_size = [512,512]
        self.mode = split
        self.images = self._load_json(f"/mnt/netdisk/zhangjh/Code/DeepLabV3Plus-Pytorch/datasets/data/pku2mars/{split}_random.json")
        
        if idx_list==None:
            self.idx_list = list(range(0,len(self.images)))
            self.label_list = list(range(0,len(self.images)))
        elif flag_semi:
            self.idx_list = list(range(0,len(self.images)))
            with open(idx_list,'r') as f:
                self.label_list = f.readlines()
            for i in range(len(self.label_list)):
                self.label_list[i] = int(self.label_list[i].strip()) 
        else:
            with open(idx_list,'r') as f:
                self.idx_list = f.readlines()
            for i in range(len(self.idx_list)):
                self.idx_list[i] = int(self.idx_list[i].strip()) 
            print("=====================")
            print("data sample num:", len(self.idx_list), self.flag_semi)
            print("=====================")
            if len(self.idx_list)<len(self.images):
                repeat_num = (len(self.images) // len(self.idx_list))
                self.idx_list = self.idx_list * repeat_num
        self.num = len(self.idx_list)
        self.two_crop = True
        self.return_coord = True
    @classmethod
    def encode_target(cls, target):
        return cls.id_to_train_id[np.array(target)]
    @staticmethod
    def _get_to_tensor_and_normalize(mean, std):
        return img_trsform.ToTensorAndNormalize(mean, std)
    @classmethod
    def decode_target(cls, target):
        target[target == 255] = 9
        #target = target.astype('uint8') + 1
        return cls.train_id_to_color[target]
    @staticmethod
    def decode_from_rle(path):
        def load_json(path):
            with open(path, 'r') as file:
                data = json.load(file)
            return data
        anno = load_json(path)
        #有许多mask
        mask_list = []
        quality = []
        #TODO filter the mask according to the confidence
        num = len(anno)
        for i in range(num):
            m = mask_utils.decode(anno[i]["segmentation"])
            
            h,w = m.shape

            m = Image.fromarray((m*255).astype(np.uint8))
            m = m.filter(ImageFilter.GaussianBlur(radius=5.0))
            m = np.array(m)
            m[m>128] = 255
            m[m<129] = 0
            m = (m==255).astype(np.uint8)

            if np.array(m).sum() > 512*512:
                continue
            if np.array(m).sum() < 1700 or anno[i]['predicted_iou'] < 0.97 or i>10:
                break
            mask_list.append(np.array(m.copy()))
            quality.append(np.exp(anno[i]['predicted_iou']/0.07))
        if len(mask_list)!=0:
            mask = random.choices(mask_list,k=1)[0]
        else:
            mask = np.zeros((h,w))
            #print("!")
        return Image.fromarray(mask.astype(np.uint8))#H,W

    @staticmethod
    def decode_from_rle_filter_k(path,label,k):
        def load_json(path):
            with open(path, 'r') as file:
                data = json.load(file)
            return data
        anno = load_json(path)
        #有许多mask
        mask_list = []
        quality = []
        #TODO filter the mask according to the confidence
        num = len(anno)
        for i in range(num):
            m = mask_utils.decode(anno[i]["segmentation"])
            
            h,w = m.shape

            m = Image.fromarray((m*255).astype(np.uint8))
            m = m.filter(ImageFilter.GaussianBlur(radius=5.0))
            m = np.array(m)
            m[m>128] = 255
            m[m<129] = 0
            m = (m==255).astype(np.uint8)

            if np.array(m).sum() > 512*512:
                continue
            if (np.array(label)*np.array(m)).sum()>0:
                continue
            if np.array(m).sum() < 1300 or anno[i]['predicted_iou'] < 0.95 or i>20:
                break
            mask_list.append(np.array(m.copy()))
        
        if len(mask_list)>=3:
            mask = random.choices(mask_list,k=k)
        else:
            mask = mask_list
            while(len(mask)<3):
                mask.append(np.zeros((h,w)))
            #print("!")
        mask = np.stack(mask, axis=-1)#h,w,k
        return Image.fromarray(mask.astype(np.uint8))#H,W,k

    def __getitem__(self, index):
        """
        for object mix
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is a tuple of all target types if target_type is a list with more
            than one item. Otherwise target is a json object if target_type="polygon", else the image segmentation.
        """
        index = self.idx_list[index]
        image = Image.open(os.path.join(self.images_dir, self.images[index] + ".jpg")).convert('RGB')
        #image = Image.open('/mnt/netdisk/zhangjh/Code/DeepLabV3Plus-Pytorch/0_image.png').convert('RGB')
        label = Image.open(os.path.join(self.targets_dir, self.images[index] + ".png")).convert('L')
        
        if self.transform_strong is None:
            mask = self.decode_from_rle(os.path.join(self.mask_dir, self.images[index] + ".json"))
            image, label, mask = self.transform_weak(image, label, mask)
            # print(image.shape, label.shape)
            obj, object_label, mask = object_augment(image, label, mask) 
            
            obj, object_label, mask = self.trf_normalize(obj, object_label, mask)
            image, label, mask = self.trf_normalize(image, label, mask)
            
            label = self.encode_target(label)
            object_label = self.encode_target(object_label)
            if not self.flag_semi:
                return index, image, label, obj, object_label, mask
            else:
                return index, image, image.clone(), label
        else:
            # apply augmentation
            image_weak, label = self.transform_weak(image, label)
            image_strong = self.transform_strong(image_weak)

            image_weak, label = self.trf_normalize(image_weak, label)
            image_strong, _ = self.trf_normalize(image_strong, label)
            label = self.encode_target(label)
            if index not in self.label_list:
                label = np.full(label.shape,255)
            return index, image_weak, image_strong, label
    def __len__(self):
        return len(self.idx_list)

    def _load_json(self, path):
        with open(path, 'r') as file:
            data = json.load(file)
            #data.sort()
        return data

    def _get_target_suffix(self, mode, target_type):
        if target_type == 'instance':
            return '{}_instanceIds.png'.format(mode)
        elif target_type == 'semantic':
            return '{}_labelIds.png'.format(mode)
        elif target_type == 'color':
            return '{}_color.png'.format(mode)
        elif target_type == 'polygon':
            return '{}_polygons.json'.format(mode)
        elif target_type == 'depth':
            return '{}_disparity.png'.format(mode)

def build_additional_strong_transform(cfg):
    assert cfg.get("strong_aug", False) != False
    strong_aug_nums = cfg["strong_aug"].get("num_augs", 2)
    flag_use_rand_num = cfg["strong_aug"].get("flag_use_random_num_sampling", True)
    strong_img_aug = img_trsform.strong_img_aug(strong_aug_nums,
            flag_using_random_num=flag_use_rand_num)
    return strong_img_aug


def build_basic_transfrom(cfg, split="val", mean=[0.625, 0.519, 0.363,]):
    ignore_label = cfg["ignore_label"]
    trs_form = []
    if split != "val":
        if cfg.get("rand_resize", False):
            trs_form.append(img_trsform.Resize(cfg.get("resize_base_size", [1024, 1024]), cfg["rand_resize"]))
        
        if cfg.get("flip", False):
            trs_form.append(img_trsform.RandomFlip(prob=0.5, flag_hflip=True))
    
        # crop also sometime for validating
        if cfg.get("crop", False):
            crop_size, crop_type = cfg["crop"]["size"], cfg["crop"]["type"]
            trs_form.append(img_trsform.Crop(crop_size, crop_type=crop_type, mean=mean, ignore_value=0))
            #ignore_value is 0, because the label has not been converted to the label_id
    else:
        trs_form.append(img_trsform.Crop([1024,1024], crop_type='center', mean=mean, ignore_value=ignore_label))
    return img_trsform.Compose(trs_form)


def build_marsloader(split, all_cfg, seed=0):
    # extract augs config from "train"/"val" into the higher level.
    cfg_dset = all_cfg["dataset"]
    cfg = copy.deepcopy(cfg_dset)
    cfg.update(cfg.get(split, {}))

    # set up workers and batchsize
    workers = cfg.get("workers", 2)
    batch_size = cfg.get("batch_size", 1)
    n_sup = cfg.get("n_sup", 2975)

    # build transform
    mean, std = cfg["mean"], cfg["std"]
    trs_form = build_basic_transfrom(cfg, split=split, mean=mean)

    # create dataset
    dset = PKU2MARS(split, trs_form, None, False)

    # build sampler
    sample = DistributedSampler(dset)
    loader = DataLoader(
        dset,
        batch_size=batch_size,
        num_workers=workers,
        sampler=sample,
        shuffle=False,
        pin_memory=False,
        worker_init_fn=seed_worker,
    )
    return loader


def build_mars_semi_loader(split, all_cfg, seed=0):
    split = "train"
    # extract augs config from "train" into the higher level.
    cfg_dset = all_cfg["dataset"]
    cfg = copy.deepcopy(cfg_dset)
    cfg.update(cfg.get(split, {}))

    # set up workers and batchsize
    workers = cfg.get("workers", 2) 
    batch_size = cfg.get("batch_size", 2)
    n_sup = 2975 - cfg.get("n_sup", 2975)

    # build transform
    mean, std = cfg["mean"], cfg["std"]
    trs_form_weak = build_basic_transfrom(cfg, split=split, mean=mean)
    if cfg.get("strong_aug", False):
        trs_form_strong = build_additional_strong_transform(cfg)
    else:
        trs_form_strong = None
    
    dset = PKU2MARS(split, trs_form_weak, None,idx_list=cfg_dset['semi'])

    sample_sup = DistributedSampler(dset)

    data_list_unsup = cfg["data_list"].replace("labeled.txt", "unlabeled.txt")
    dset_unsup = PKU2MARS(split, trs_form_weak, trs_form_strong,
                            flag_semi=True, idx_list=cfg_dset['semi'])
    sample_unsup = DistributedSampler(dset_unsup)

    # create dataloader
    loader_sup = DataLoader(
        dset,
        batch_size=batch_size,
        num_workers=workers,
        sampler=sample_sup,
        shuffle=False,
        pin_memory=True,
        drop_last=True,
        worker_init_fn=seed_worker,
    )
    loader_unsup = DataLoader(
        dset_unsup,
        batch_size=batch_size,
        num_workers=workers,
        sampler=sample_unsup,
        shuffle=False,
        pin_memory=True,
        drop_last=True,
        worker_init_fn=seed_worker,
    )
    return loader_sup, loader_unsup

def patch_mask_gen(size,patch_size=8,ratio=0.2):
    h,w = size
    assert h==w
    mask = np.ones((h,w))
    patch_num = h//patch_size
    mask = mask.reshape(patch_num, patch_size, patch_num, patch_size)
    mask = np.transpose(mask,(0,2,1,3)).reshape(patch_num*patch_num,patch_size,patch_size)
    mask_idx = random.sample(list(range(patch_num**2)), int(patch_num*patch_num*ratio))
    mask[np.array(mask_idx)] = mask[np.array(mask_idx)] - 1
    mask = mask.reshape(patch_num, patch_num, patch_size, patch_size)
    mask = mask.transpose(0,2,1,3)
    mask = mask.reshape(h,w)
    return mask
