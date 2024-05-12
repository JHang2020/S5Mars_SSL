import numpy as np
import scipy.stats as stats
from PIL import Image, ImageOps, ImageFilter, ImageEnhance
import random
import collections
import cv2
import torch
from skimage.exposure import match_histograms
from torchvision import transforms
from .randaug import RandAugmentMC

# # # # # # # # # # # # # # # # # # # # # # # # 
# # # 1. Augmentation for image and labels (with masks)
# # # # # # # # # # # # # # # # # # # # # # # # 
class Compose(object):
    def __init__(self, segtransforms):
        self.segtransforms = segtransforms

    def __call__(self, image, label, mask=None):
        for idx, t in enumerate(self.segtransforms):
            if isinstance(t, strong_img_aug):
                image = t(image)
            else:
                if mask!=None:
                    image, label, mask = t(image, label,mask)
                else:
                    image, label = t(image, label)
        if mask!=None:
            return image, label, mask
        else:
            return image, label


class ToTensorAndNormalize(object):
    def __init__(self, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
        assert len(mean) == len(std)
        assert len(mean) == 3
        self.normalize = transforms.Normalize(mean, std)
        self.to_tensor = transforms.ToTensor()

    def __call__(self, in_image, in_label, in_mask=None):
        in_image = Image.fromarray(np.uint8(in_image))
        image = self.normalize(self.to_tensor(in_image))
        label = torch.from_numpy(np.array(in_label, dtype=np.int32)).long()
        if in_mask!=None:
            mask = torch.from_numpy(np.array(in_mask, dtype=np.int32)).long()
            return image, label, mask
        return image, label


class Resize(object):
    def __init__(self, base_size, ratio_range, scale=True, bigger_side_to_base_size=True):
        assert isinstance(ratio_range, collections.Iterable) and len(ratio_range) == 2
        self.base_size = base_size
        self.ratio_range = ratio_range
        self.scale = scale
        self.bigger_side_to_base_size = bigger_side_to_base_size

    def __call__(self, in_image, in_label,in_mask=None):
        w, h = in_image.size
        
        if isinstance(self.base_size, int):
            # obtain long_side
            if self.scale:
                long_side = random.randint(int(self.base_size * self.ratio_range[0]), 
                                        int(self.base_size * self.ratio_range[1]))
            else:
                long_side = self.base_size
                
            # obtain new oh, ow
            if self.bigger_side_to_base_size:
                if h > w:
                    oh = long_side
                    ow = int(1.0 * long_side * w / h + 0.5)
                else:
                    oh = int(1.0 * long_side * h / w + 0.5)
                    ow = long_side
            else:
                oh, ow = (long_side, int(1.0 * long_side * w / h + 0.5)) if h < w else (
                        int(1.0 * long_side * h / w + 0.5), long_side)
            
            if in_mask==None:
                image = in_image.resize((ow, oh), Image.BILINEAR)
                label = in_label.resize((ow, oh), Image.NEAREST)
                return image, label
            else:
                image = in_image.resize((ow, oh), Image.BILINEAR)
                label = in_label.resize((ow, oh), Image.NEAREST)
                mask = in_mask.resize((ow, oh), Image.NEAREST)
                return image, label, mask
        elif (isinstance(self.base_size, list) or isinstance(self.base_size, tuple)) and len(self.base_size) == 2:
            if self.scale:
                # scale = random.random() * 1.5 + 0.5  # Scaling between [0.5, 2]
                scale = self.ratio_range[0] + random.random() * (self.ratio_range[1] - self.ratio_range[0])
                # print("="*100, h, self.base_size[0])
                # print("="*100, w, self.base_size[1])
                oh, ow = int(self.base_size[0] * scale), int(self.base_size[1] * scale)
            else:
                oh, ow = self.base_size
            if in_mask==None:
                image = in_image.resize((ow, oh), Image.BILINEAR)
                label = in_label.resize((ow, oh), Image.NEAREST)
                return image, label
            else:
                image = in_image.resize((ow, oh), Image.BILINEAR)
                label = in_label.resize((ow, oh), Image.NEAREST)
                mask = in_mask.resize((ow, oh), Image.NEAREST)
                return image, label, mask

        else:
            raise ValueError


class Crop(object):
    def __init__(self, crop_size, crop_type="rand", mean=[0.485, 0.456, 0.406], ignore_value=255):
        if (isinstance(crop_size, list) or isinstance(crop_size, tuple)) and len(crop_size) == 2:
            self.crop_h, self.crop_w = crop_size
        elif isinstance(crop_size, int):
            self.crop_h, self.crop_w = crop_size, crop_size
        else:
            raise ValueError
        
        self.crop_type = crop_type
        self.image_padding = (np.array(mean) * 255.).tolist()
        self.ignore_value = ignore_value

    def __call__(self, in_image, in_label, in_mask=None):
        # Padding to return the correct crop size
        w, h = in_image.size
        pad_h = max(self.crop_h - h, 0)
        pad_w = max(self.crop_w - w, 0)
        
        #print(w,h,self.crop_w,self.crop_h)
        pad_kwargs = {
            "top": 0,
            "bottom": pad_h,
            "left": 0,
            "right": pad_w,
            "borderType": cv2.BORDER_CONSTANT, 
        }
        if pad_h > 0 or pad_w > 0:
            image = cv2.copyMakeBorder(np.asarray(in_image, dtype=np.float32), 
                                       value=self.image_padding, **pad_kwargs)
            label = cv2.copyMakeBorder(np.asarray(in_label, dtype=np.int32), 
                                       value=self.ignore_value, **pad_kwargs)
            if in_mask!=None:
                mask = cv2.copyMakeBorder(np.asarray(in_mask, dtype=np.int32), 
                                          value=0, **pad_kwargs)
                
            image = Image.fromarray(np.uint8(image))
            label = Image.fromarray(np.uint8(label))
            if in_mask!=None:
                mask = Image.fromarray(np.uint8(mask))
        else:
            image = in_image
            label = in_label
            if in_mask!=None:
                mask = in_mask
        
        # cropping
        w, h = image.size
        if self.crop_type == "rand":
            x = random.randint(0, w - self.crop_w)
            y = random.randint(0, h - self.crop_h)
        else:
            x = (w - self.crop_w) // 2
            y = (h - self.crop_h) // 2
        image = image.crop((x, y, x + self.crop_w, y + self.crop_h))
        label = label.crop((x, y, x + self.crop_w, y + self.crop_h))
        if in_mask!=None:
            mask = mask.crop((x, y, x + self.crop_w, y + self.crop_h))
            return image, label, mask
        return image, label


class RandomFlip(object):
    def __init__(self, prob=0.5, flag_hflip=True,):
        self.prob = prob
        if flag_hflip:
            self.type_flip = Image.FLIP_LEFT_RIGHT
        else:
            self.type_flip = Image.FLIP_TOP_BOTTOM
            
    def __call__(self, in_image, in_label, in_mask=None):
        if in_mask!=None:
            if random.random() < self.prob:
                in_image = in_image.transpose(self.type_flip)
                in_label = in_label.transpose(self.type_flip)
                in_mask = in_mask.transpose(self.type_flip)
            return in_image, in_label, in_mask
        else:
            if random.random() < self.prob:
                in_image = in_image.transpose(self.type_flip)
                in_label = in_label.transpose(self.type_flip)
            return in_image, in_label


# # # # # # # # # # # # # # # # # # # # # # # # 
# # # 2. Strong Augmentation for image only
# # # # # # # # # # # # # # # # # # # # # # # # 

def img_aug_identity(img, scale=None):
    return img


def img_aug_autocontrast(img, scale=None):
    return ImageOps.autocontrast(img)


def img_aug_equalize(img, scale=None):
    return ImageOps.equalize(img)


def img_aug_invert(img, scale=None):
    return ImageOps.invert(img)


def img_aug_blur(img, scale=[0.1, 2.0]):
    assert scale[0] < scale[1]
    sigma = np.random.uniform(scale[0], scale[1])
    # print(f"sigma:{sigma}")
    return img.filter(ImageFilter.GaussianBlur(radius=sigma))


def img_aug_contrast(img, scale=[0.05, 0.95]):
    min_v, max_v = min(scale), max(scale)
    v = float(max_v - min_v)*random.random()
    v = max_v - v
    # # print(f"final:{v}")
    # v = np.random.uniform(scale[0], scale[1])
    return ImageEnhance.Contrast(img).enhance(v)


def img_aug_brightness(img, scale=[0.05, 0.95]):
    min_v, max_v = min(scale), max(scale)
    v = float(max_v - min_v)*random.random()
    v = max_v - v
    #v=0.3
    # print(f"final:{v}")
    return ImageEnhance.Brightness(img).enhance(v)


def img_aug_color(img, scale=[0.05, 0.95]):
    min_v, max_v = min(scale), max(scale)
    v = float(max_v - min_v)*random.random()
    v = max_v - v
    # print(f"final:{v}")
    return ImageEnhance.Color(img).enhance(v)


def img_aug_sharpness(img, scale=[0.05, 0.95]):
    min_v, max_v = min(scale), max(scale)
    v = float(max_v - min_v)*random.random()
    v = max_v - v
    # print(f"final:{v}")
    return ImageEnhance.Sharpness(img).enhance(v)


def img_aug_hue(img, scale=[0, 0.5]):
    min_v, max_v = min(scale), max(scale)
    v = float(max_v - min_v)*random.random()
    v += min_v
    if np.random.random() < 0.5:
        hue_factor = -v
    else:
        hue_factor = v
    # print(f"Final-V:{hue_factor}")
    input_mode = img.mode
    if input_mode in {"L", "1", "I", "F"}:
        return img
    h, s, v = img.convert("HSV").split()
    np_h = np.array(h, dtype=np.uint8)
    # uint8 addition take cares of rotation across boundaries
    with np.errstate(over="ignore"):
        np_h += np.uint8(hue_factor * 255)
    h = Image.fromarray(np_h, "L")
    img = Image.merge("HSV", (h, s, v)).convert(input_mode)
    return img

def img_aug_adain(img, type=2):
    '''
        img: tensor BCHW
        exchange the mean and std
    '''

    u_rand_index = torch.randperm(img.size()[0])[:img.size()[0]]
    new_img = img.clone()
    mea = img.mean(dim=[2,3])
    std = (img**2).mean(dim=[2,3]) - mea**2
    std = torch.sqrt(std).unsqueeze(-1).unsqueeze(-1)
    mea = mea.unsqueeze(-1).unsqueeze(-1)
    if type==0: # mean
        new_img = ((new_img[:]-mea[:])) + mea[u_rand_index]
    elif type==1: # std
        new_img = ((new_img[:]-mea[:])/std)*std[u_rand_index] + mea[:]
    elif type==2: #both
        new_img = ((new_img[:]-mea[:])/std)*std[u_rand_index] + mea[u_rand_index]
    # print(f"final:{v}")
    return new_img

#EFDM
def exact_feature_distribution_matching(content_feat):
    u_rand_index = torch.randperm(content_feat.size()[0])[:content_feat.size()[0]]
    new_img = content_feat.clone()
    style_feat = content_feat[u_rand_index]

    assert (content_feat.size() == style_feat.size())
    B, C, W, H = content_feat.size(0), content_feat.size(1), content_feat.size(2), content_feat.size(3)
    value_content, index_content = torch.sort(content_feat.view(B,C,-1))  # sort conduct a deep copy here.
    value_style, _ = torch.sort(style_feat.view(B,C,-1))  # sort conduct a deep copy here.
    inverse_index = index_content.argsort(-1)
    new_content = content_feat.view(B,C,-1) + (value_style.gather(-1, inverse_index) - content_feat.view(B,C,-1).detach())

    return new_content.view(B, C, W, H)

## HM
def histogram_matching(content_feat):
    u_rand_index = torch.randperm(content_feat.size()[0])[:content_feat.size()[0]]
    new_img = content_feat.clone()
    style_feat = content_feat[u_rand_index]
    assert (content_feat.size() == style_feat.size())
    B, C, W, H = content_feat.size(0), content_feat.size(1), content_feat.size(2), content_feat.size(3)
    x_view = content_feat.view(-1, W,H)
    image1_temp = match_histograms(np.array(x_view.detach().clone().cpu().float().transpose(0, 2)),
                                   np.array(style_feat.view(-1, W, H).detach().clone().cpu().float().transpose(0, 2)),
                                   multichannel=True)
    image1_temp = torch.from_numpy(image1_temp).float().to(content_feat.device).transpose(0, 2).view(B, C, W, H)
    return content_feat + (image1_temp - content_feat).detach()


def whiten_and_color(cF,sF):
    cFSize = cF.size()
    c_mean = torch.mean(cF,1) # c x (h x w)
    c_mean = c_mean.unsqueeze(1).expand_as(cF)
    cF = cF - c_mean

    contentConv = torch.mm(cF,cF.t()).div(cFSize[1]-1) + torch.eye(cFSize[0]).double().cuda()
    c_u,c_e,c_v = torch.svd(contentConv,some=False)

    k_c = cFSize[0]
    for i in range(cFSize[0]):
        if c_e[i] < 0.00001:
            k_c = i
            break

    sFSize = sF.size()
    s_mean = torch.mean(sF,1)
    sF = sF - s_mean.unsqueeze(1).expand_as(sF)
    styleConv = torch.mm(sF,sF.t()).div(sFSize[1]-1)
    s_u,s_e,s_v = torch.svd(styleConv,some=False)

    k_s = sFSize[0]
    for i in range(sFSize[0]):
        if s_e[i] < 0.00001:
            k_s = i
            break

    c_d = (c_e[0:k_c]).pow(-0.5)
    step1 = torch.mm(c_v[:,0:k_c],torch.diag(c_d))
    step2 = torch.mm(step1,(c_v[:,0:k_c].t()))
    whiten_cF = torch.mm(step2,cF)

    s_d = (s_e[0:k_s]).pow(0.5)
    targetFeature = torch.mm(torch.mm(torch.mm(s_v[:,0:k_s],torch.diag(s_d)),(s_v[:,0:k_s].t())),whiten_cF)
    targetFeature = targetFeature + s_mean.unsqueeze(1).expand_as(targetFeature)
    return targetFeature

def transform(cF,alpha=0.6):
    cF = cF.double()
    u_rand_index = torch.randperm(cF.size()[0])[:cF.size()[0]]
    new_img = cF.clone()
    sF = new_img[u_rand_index]
    C,W,H = cF.size(0),cF.size(1),cF.size(2)
    _,W1,H1 = sF.size(0),sF.size(1),sF.size(2)
    cFView = cF.view(C,-1)
    sFView = sF.view(C,-1)

    targetFeature = whiten_and_color(cFView,sFView)
    targetFeature = targetFeature.view_as(cF)
    ccsF = alpha * targetFeature + (1.0 - alpha) * cF
    ccsF = ccsF.float()
    return ccsF

def img_aug_posterize(img, scale=[4, 8]):
    min_v, max_v = min(scale), max(scale)
    v = float(max_v - min_v)*random.random()
    # print(min_v, max_v, v)
    v = int(np.ceil(v))
    v = max(1, v)
    v = max_v - v
    # print(f"final:{v}")
    return ImageOps.posterize(img, v)


def img_aug_solarize(img, scale=[1, 256]):
    min_v, max_v = min(scale), max(scale)
    v = float(max_v - min_v)*random.random()
    # print(min_v, max_v, v)
    v = int(np.ceil(v))
    v = max(1, v)
    v = max_v - v
    # print(f"final:{v}")
    return ImageOps.solarize(img, v)

def get_augment_list(flag_using_wide=False):  
    if flag_using_wide:
        l = [
        (img_aug_identity, None),
        (img_aug_autocontrast, None),
        (img_aug_equalize, None),
        (img_aug_blur, [0.1, 2.0]),
        (img_aug_contrast, [0.1, 1.8]),
        (img_aug_brightness, [0.1, 1.8]),
        (img_aug_color, [0.1, 1.8]),
        (img_aug_sharpness, [0.1, 1.8]),
        (img_aug_posterize, [2, 8]),
        (img_aug_solarize, [1, 256]),
        (img_aug_hue, [0, 0.5])
        ]
    else:
        l = [
            (img_aug_identity, None), #不变
            (img_aug_autocontrast, None), #可以考虑，但是对比度也会改变颜色分布，包括下面的均衡化
            (img_aug_equalize, None),
            (img_aug_blur, [0.1, 2.0]), #可以考虑
            (img_aug_contrast, [0.05, 0.95]),
            (img_aug_brightness, [0.05, 0.95]),
            (img_aug_color, [0.05, 0.95]),
            (img_aug_sharpness, [0.05, 0.95]),
            (img_aug_posterize, [4, 8]),
            (img_aug_solarize, [1, 256]),
            (img_aug_hue, [0, 0.5])
        ]
        l = [
            (img_aug_identity, None), #不变
        ]

    return l


class strong_img_aug:
    def __init__(self, num_augs, flag_using_random_num=False):
        assert 1<= num_augs <= 11
        self.n = num_augs
        self.augment_list = get_augment_list(flag_using_wide=False)
        self.flag_using_random_num = flag_using_random_num
    
    def __call__(self, img):
        if self.flag_using_random_num:
            max_num = np.random.randint(1, high=self.n + 1)
        else:
            max_num =self.n
        ops = random.choices(self.augment_list, k=max_num)
        for op, scales in ops:
            # print("="*20, str(op))
            img = op(img, scales)
        return img
