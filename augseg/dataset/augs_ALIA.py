import numpy as np
import random
import torch
import scipy.stats as stats
import cv2
from PIL import Image 
# # # # # # # # # # # # # # # # # # # # # 
# # 0 random box
# # # # # # # # # # # # # # # # # # # # # 
def rand_bbox(size, lam=None):
    # past implementation
    if len(size) == 4:
        W = size[2]
        H = size[3]
    elif len(size) == 3:
        W = size[1]
        H = size[2]
    else:
        raise Exception
    B = size[0]
    
    cut_rat = np.sqrt(1. - lam)
    cut_w = int(W * cut_rat)
    cut_h = int(H * cut_rat)

    cx = np.random.randint(size=[B, ], low=int(W/8), high=W)
    cy = np.random.randint(size=[B, ], low=int(H/8), high=H)
    
    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)

    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)


    return bbx1, bby1, bbx2, bby2


# # # # # # # # # # # # # # # # # # # # # 
# # 1 cutmix label-adaptive 
# # # # # # # # # # # # # # # # # # # # # 
def cut_mix_label_adaptive(unlabeled_image, unlabeled_mask, unlabeled_logits, 
        labeled_image, labeled_mask, lst_confidences):
    assert len(lst_confidences) == len(unlabeled_image), "Ensure the confidence is properly obtained"
    assert labeled_image.shape == unlabeled_image.shape, "Ensure shape match between lb and unlb"
    mix_unlabeled_image = unlabeled_image.clone()
    mix_unlabeled_target = unlabeled_mask.clone()
    mix_unlabeled_logits = unlabeled_logits.clone()
    labeled_logits = torch.ones_like(labeled_mask)

    # 1) get the random mixing objects
    u_rand_index = torch.randperm(unlabeled_image.size()[0])[:unlabeled_image.size()[0]]
    
    # 2) get box
    ##更容易取到大的值 被cut的部分size更小
    l_bbx1, l_bby1, l_bbx2, l_bby2 = rand_bbox(unlabeled_image.size(), lam=np.random.beta(8, 2))
    u_bbx1, u_bby1, u_bbx2, u_bby2 = rand_bbox(unlabeled_image.size(), lam=np.random.beta(4, 4))
    
    # 3) labeled adaptive
    for i in range(0, mix_unlabeled_image.shape[0]):
        if np.random.random() > lst_confidences[i]:
            mix_unlabeled_image[i, :, l_bbx1[i]:l_bbx2[i], l_bby1[i]:l_bby2[i]] = \
            labeled_image[u_rand_index[i], :, l_bbx1[i]:l_bbx2[i], l_bby1[i]:l_bby2[i]]
        
            mix_unlabeled_target[i, l_bbx1[i]:l_bbx2[i], l_bby1[i]:l_bby2[i]] = \
                labeled_mask[u_rand_index[i], l_bbx1[i]:l_bbx2[i], l_bby1[i]:l_bby2[i]]
            
            mix_unlabeled_logits[i, l_bbx1[i]:l_bbx2[i], l_bby1[i]:l_bby2[i]] = \
                labeled_logits[u_rand_index[i], l_bbx1[i]:l_bbx2[i], l_bby1[i]:l_bby2[i]]
    
    # 4) copy and paste
    for i in range(0, unlabeled_image.shape[0]):
            unlabeled_image[i, :, u_bbx1[i]:u_bbx2[i], u_bby1[i]:u_bby2[i]] = \
                mix_unlabeled_image[u_rand_index[i], :, u_bbx1[i]:u_bbx2[i], u_bby1[i]:u_bby2[i]]

            unlabeled_mask[i, u_bbx1[i]:u_bbx2[i], u_bby1[i]:u_bby2[i]] = \
                mix_unlabeled_target[u_rand_index[i], u_bbx1[i]:u_bbx2[i], u_bby1[i]:u_bby2[i]]
                
            unlabeled_logits[i, u_bbx1[i]:u_bbx2[i], u_bby1[i]:u_bby2[i]] = \
                mix_unlabeled_logits[u_rand_index[i], u_bbx1[i]:u_bbx2[i], u_bby1[i]:u_bby2[i]]
    
    del mix_unlabeled_image, mix_unlabeled_target, mix_unlabeled_logits, labeled_logits
    
    return unlabeled_image, unlabeled_mask, unlabeled_logits 

def obj_rotate(image, angle, mask, label):
        # grab the dimensions of the image and then determine the
        # center
        assert 0<=angle<45
        (h, w, c) = image.shape
        (cX, cY) = (w // 2, h // 2)
        
        # grab the rotation matrix (applying the negative of the
        # angle to rotate clockwise), then grab the sine and cosine
        # (i.e., the rotation components of the matrix)
        M = cv2.getRotationMatrix2D((cX, cY), -angle, 1.0)
        cos = np.abs(M[0, 0])
        sin = np.abs(M[0, 1])
        # compute the new bounding dimensions of the image
        nW = int((h * sin) + (w * cos))
        nH = int((h * cos) + (w * sin))
        # adjust the rotation matrix to take into account translation
        M[0, 2] += (nW / 2) - cX
        M[1, 2] += (nH / 2) - cY
    
        # perform the actual rotation and return the image
        image = cv2.warpAffine(image, M, (nW, nH))
        mask = cv2.warpAffine(mask, M, (nW, nH))
        label = cv2.warpAffine(label, M, (nW, nH))
        
        image = cv2.resize(image, (w, h),interpolation=cv2.INTER_LINEAR)
        mask = cv2.resize(mask, (w, h),interpolation=cv2.INTER_NEAREST)
        label = cv2.resize(label, (w, h),interpolation=cv2.INTER_NEAREST)

        return image, mask, label

def ClassMix(img, label, max_logits=None, pred_logits=None):
    '''
    ClassMix Implementation: https://github.com/WilhelmT/ClassMix/blob/master/utils/transformsgpu.py#L54
    img: B,C,H,W
    label: B,H,W
    '''
    def generate_class_mask(pred, classes):
        '''
        pred: H,W
        classes: a 1-d tensor contains the class-idx appearing in the image 
        '''
        pred, classes = torch.broadcast_tensors(pred.unsqueeze(0), classes.unsqueeze(1).unsqueeze(2))
        N = pred.eq(classes).sum(0)
        return N
    b,c,h,w = img.shape
    for i in range(b):
        classes = torch.unique(label[i])
        classes = classes[classes != 255]
        nclasses = classes.shape[0]
        classes = (classes[torch.Tensor(np.random.choice(nclasses, int((nclasses - nclasses % 2) / 2), replace=False)).long()]).cuda()
        if i == 0:
            MixMask = generate_class_mask(label[i], classes).unsqueeze(0).cuda()
        else:
            MixMask = torch.cat((MixMask, generate_class_mask(label[i], classes).unsqueeze(0).cuda()))
        #MixMask N, H, W
    
    mask = MixMask
    data = img
    target = label
    if mask.shape[0] == data.shape[0]:
        data = torch.cat([(mask[i] * data[i] + (1 - mask[i]) * data[(i + 1) % data.shape[0]]).unsqueeze(0) for i in range(data.shape[0])])
        target = torch.cat([(mask[i] * target[i] + (1 - mask[i]) * target[(i + 1) % target.shape[0]]).unsqueeze(0) for i in range(target.shape[0])])
        if max_logits!=None:
            max_logits = torch.cat([(mask[i] * max_logits[i] + (1 - mask[i]) * max_logits[(i + 1) % max_logits.shape[0]]).unsqueeze(0) for i in range(max_logits.shape[0])])
        if pred_logits!=None:
            pred_logits = torch.cat([(mask[i].unsqueeze(0) * pred_logits[i] + (1 - mask[i].unsqueeze(0)) * pred_logits[(i + 1) % pred_logits.shape[0]]).unsqueeze(0) for i in range(pred_logits.shape[0])])
    
    return data, target, max_logits, pred_logits

def DAMix(img, label, img2, label2, max_logits=None, pred_logits=None, max_logits2=None, pred_logits2=None):
    '''
    DACS: Domain Adaptation via Cross-domain Mixed Sampling
    ClassMix Implementation: https://github.com/WilhelmT/ClassMix/blob/master/utils/transformsgpu.py#L54
    根据ground-truth来混合
    img: B,C,H,W
    label: B,H,W
    '''
    def generate_class_mask(pred, classes):
        '''
        pred: H,W
        classes: a 1-d tensor contains the class-idx appearing in the image 
        '''
        pred, classes = torch.broadcast_tensors(pred.unsqueeze(0), classes.unsqueeze(1).unsqueeze(2))
        N = pred.eq(classes).sum(0)
        return N
    b,c,h,w = img.shape
    for i in range(b):
        classes = torch.unique(label2[i])
        classes = classes[classes != 255]
        nclasses = classes.shape[0]
        classes = (classes[torch.Tensor(np.random.choice(nclasses, int((nclasses - nclasses % 2) / 2), replace=False)).long()]).cuda()
        if i == 0:
            MixMask = generate_class_mask(label2[i], classes).unsqueeze(0).cuda()
        else:
            MixMask = torch.cat((MixMask, generate_class_mask(label2[i], classes).unsqueeze(0).cuda()))
        #MixMask N, H, W
    
    mask = MixMask
    data = img
    target = label
    if mask.shape[0] == data.shape[0]:
        data = torch.cat([(mask[i] * img2[i] + (1 - mask[i]) * data[(i + 1) % data.shape[0]]).unsqueeze(0) for i in range(data.shape[0])])
        target = torch.cat([(mask[i] * label2[i] + (1 - mask[i]) * target[(i + 1) % target.shape[0]]).unsqueeze(0) for i in range(target.shape[0])])
        if max_logits!=None:
            max_logits = torch.cat([(mask[i] * max_logits2[i] + (1 - mask[i]) * max_logits[(i + 1) % max_logits.shape[0]]).unsqueeze(0) for i in range(max_logits.shape[0])])
        if pred_logits!=None:
            pred_logits2 = pred_logits2.permute(0,3,1,2)
            pred_logits = torch.cat([(mask[i].unsqueeze(0) * pred_logits2[i] + (1 - mask[i].unsqueeze(0)) * pred_logits[(i + 1) % pred_logits.shape[0]]).unsqueeze(0) for i in range(pred_logits.shape[0])])
    
    return data, target, max_logits, pred_logits

def object_augment(img, label, mask, ignore_value=0):
    '''
        img: img (H,W,C)
        label: img (H,W)
        mask: (H,W) for img's object 0 or 1
    '''
    object = np.array(img)#(B,C,H,W)
    label = np.array(label)
    mask = np.array(mask)
    label[mask==0] = ignore_value
    object_label = label.copy()

    #Augmentation for object
    rotation_prob = 0.0
    flip_prob = 0.5
    rescale_prob = 0.0

    #rotation
    if np.random.uniform() < rotation_prob:
        ang = random.randint(0,30)
        object, mask, object_label = obj_rotate(object, ang, mask, object_label)

    #flip_prob
    if np.random.uniform() < flip_prob:
        object = np.array(cv2.flip(object, 1))
        object_label = np.array(cv2.flip(object_label, 1))
        mask = np.array(cv2.flip(mask, 1))

    #rescale
    #if np.random() < rescale_prob:
    
    return Image.fromarray(np.array(object)), Image.fromarray(np.array(object_label)), Image.fromarray(np.array(mask))

def Copy_Paste(object, img, object_label,img_label, object_mask, img_logits, object_logits, pred_img=None, pred_obj=None):
    '''
        object: tensor (B,C,H,W)
        img: tensor (B,C,H,W)
        object_label: tensor (B,1,H,W)
        img_label: tensor (B,1,H,W)
        object_mask: (B,1,H,W) for img1's object 0 or 1
    '''
    mix_image = img.clone()
    mix_label = img_label.clone()
    mix_logits = img_logits.clone()
    if pred_obj!=None:
        mix_pred = pred_img.clone()
    u_rand_index = torch.randperm(img.size()[0])[:img.size()[0]]
    u_bbx1, u_bby1, u_bbx2, u_bby2 = rand_bbox(img.size(), lam=np.random.beta(4, 4))
    for i in range(img.size()[0]):
        idx = u_rand_index[i]
        if object_mask[idx].sum()<100:
            mix_image[i, :, u_bbx1[i]:u_bbx2[i], u_bby1[i]:u_bby2[i]] = \
                img[u_rand_index[i], :, u_bbx1[i]:u_bbx2[i], u_bby1[i]:u_bby2[i]]

            mix_label[i, u_bbx1[i]:u_bbx2[i], u_bby1[i]:u_bby2[i]] = \
                img_label[u_rand_index[i], u_bbx1[i]:u_bbx2[i], u_bby1[i]:u_bby2[i]]
                
            mix_logits[i, u_bbx1[i]:u_bbx2[i], u_bby1[i]:u_bby2[i]] = \
                img_logits[u_rand_index[i], u_bbx1[i]:u_bbx2[i], u_bby1[i]:u_bby2[i]]
            if pred_img!=None:
                mix_pred[i, :, u_bbx1[i]:u_bbx2[i], u_bby1[i]:u_bby2[i]] = \
                    pred_img[u_rand_index[i], :, u_bbx1[i]:u_bbx2[i], u_bby1[i]:u_bby2[i]]
            
        else:
            mix_image[i] = object[idx]*object_mask[idx] + img[i]*(1.-object_mask[idx])
            mix_label[i] = object_label[idx]*object_mask[idx] + img_label[i]*(1.-object_mask[idx])
            mix_logits[i] = object_logits[idx]*object_mask[idx] + img_logits[i]*(1.-object_mask[idx])
            if pred_img!=None:
                mix_pred[i] = pred_obj[idx]*object_mask[idx].unsqueeze(0) + pred_img[i]*(1.-object_mask[idx]).unsqueeze(0)

            mix_image[i, :, u_bbx1[i]:u_bbx2[i], u_bby1[i]:u_bby2[i]] = \
                img[u_rand_index[i], :, u_bbx1[i]:u_bbx2[i], u_bby1[i]:u_bby2[i]]

            mix_label[i, u_bbx1[i]:u_bbx2[i], u_bby1[i]:u_bby2[i]] = \
                img_label[u_rand_index[i], u_bbx1[i]:u_bbx2[i], u_bby1[i]:u_bby2[i]]
                
            mix_logits[i, u_bbx1[i]:u_bbx2[i], u_bby1[i]:u_bby2[i]] = \
                img_logits[u_rand_index[i], u_bbx1[i]:u_bbx2[i], u_bby1[i]:u_bby2[i]]
            if pred_img!=None:
                mix_pred[i, :, u_bbx1[i]:u_bbx2[i], u_bby1[i]:u_bby2[i]] = \
                    pred_img[u_rand_index[i], :, u_bbx1[i]:u_bbx2[i], u_bby1[i]:u_bby2[i]]
    if pred_img!=None:
        return mix_image, mix_label.squeeze(dim=1), mix_logits, mix_pred
    return mix_image, mix_label.squeeze(dim=1), mix_logits
    









    
