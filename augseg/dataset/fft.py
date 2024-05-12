# -*- coding: utf-8 -*-
import cv2 
import numpy as np
from matplotlib import pyplot as plt
from PIL import Image
 
def filter_high_f(fshift, radius_ratio):
    """
    过滤掉除了中心区域外的高频信息
    """
    # 1, 生成圆形过滤器, 圆内值1, 其他部分为0的过滤器, 过滤
    template = np.zeros(fshift.shape, np.uint8)
    crow, ccol = int(fshift.shape[0] / 2), int(fshift.shape[1] / 2)  # 圆心
    radius = int(radius_ratio * img.shape[0] / 2)
    if len(img.shape) == 3:
        cv2.circle(template, (crow, ccol), radius, (1, 1, 1), -1)
    else:
        cv2.circle(template, (crow, ccol), radius, 1, -1)
    # 2, 过滤掉除了中心区域外的高频信息
    return template * fshift
 
 
def filter_low_f(fshift, radius_ratio):
    """
    去除中心区域低频信息
    """
    # 1 生成圆形过滤器, 圆内值0, 其他部分为1的过滤器, 过滤
    filter_img = np.ones(fshift.shape, np.uint8)
    crow, col = int(fshift.shape[0] / 2), int(fshift.shape[1] / 2)
    radius = int(radius_ratio * img.shape[0] / 2)
    if len(img.shape) == 3:
        cv2.circle(filter_img, (crow, col), radius, (0, 0, 0), -1)
    else:
        cv2.circle(filter_img, (crow, col), radius, 0, -1)
    # 2 过滤中心低频部分的信息
    return filter_img * fshift
 
 
def ifft(fshift):
    """
    傅里叶逆变换
    """
    ishift = np.fft.ifftshift(fshift)  # 把低频部分sift回左上角
    iimg = np.fft.ifftn(ishift)  # 出来的是复数，无法显示
    iimg = np.abs(iimg)  # 返回复数的模
    return iimg
 
 
def get_low_high_f(img, radius_ratio):
    """
    获取低频和高频部分图像
    """
    # 傅里叶变换
    # np.fft.fftn
    f = np.fft.fftn(img)  # Compute the N-dimensional discrete Fourier Transform. 零频率分量位于频谱图像的左上角
    fshift = np.fft.fftshift(f)  # 零频率分量会被移到频域图像的中心位置，即低频
 
    # 获取低频和高频部分
    hight_parts_fshift = filter_low_f(fshift.copy(), radius_ratio=radius_ratio)  # 过滤掉中心低频
    low_parts_fshift = filter_high_f(fshift.copy(), radius_ratio=radius_ratio)
 
    low_parts_img = ifft(low_parts_fshift)  # 先sift回来，再反傅里叶变换
    high_parts_img = ifft(hight_parts_fshift)
 
    # 显示原始图像和高通滤波处理图像
    img_new_low = (low_parts_img - np.amin(low_parts_img)) / (np.amax(low_parts_img) - np.amin(low_parts_img) + 0.00001)
    img_new_high = (high_parts_img - np.amin(high_parts_img) + 0.00001) / (np.amax(high_parts_img) - np.amin(high_parts_img) + 0.00001)
 
    # uint8
    img_new_low = np.array(img_new_low*255, np.uint8)
    img_new_high = np.array(img_new_high * 255, np.uint8)
    return img_new_low, img_new_high
 
 
if __name__ == '__main__':
    radius_ratio = 0.9  # 圆形过滤器的半径：ratio * w/2
    img = Image.open('/mnt/netdisk/zhangjh/Code/DeepLabV3Plus-Pytorch/example_img/1_image.png').convert('RGB')
    img.save('ori.png')
    img = np.array(img)

    low_freq_part_img0, high_freq_part_img0 = get_low_high_f(img[:,:,0], radius_ratio=radius_ratio)  # multi channel or single
    low_freq_part_img1, high_freq_part_img1 = get_low_high_f(img[:,:,1], radius_ratio=radius_ratio)  # multi channel or single
    low_freq_part_img2, high_freq_part_img2 = get_low_high_f(img[:,:,2], radius_ratio=radius_ratio)  # multi channel or single
    
    low_freq_part_img = np.stack([low_freq_part_img0,low_freq_part_img1,low_freq_part_img2],axis=2)
    high_freq_part_img = np.stack([high_freq_part_img0,high_freq_part_img1,high_freq_part_img2],axis=2)
    
    print(low_freq_part_img.mean(0).mean(0), img.mean(0).mean(0))
    low_freq_part_img = Image.fromarray(low_freq_part_img)
    low_freq_part_img.save('low.png')
    high_freq_part_img = Image.fromarray(high_freq_part_img)
    high_freq_part_img.save('high.png')