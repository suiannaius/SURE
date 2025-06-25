import torch
import torch.nn as nn
import torch.nn.functional as F
import random
import numpy as np
import json
import argparse
import os
import torchvision.transforms as transforms
from scipy.ndimage import gaussian_filter
from skimage import segmentation as skimage_seg
from scipy.ndimage import rotate
from scipy.ndimage import distance_transform_edt as distance
from preprocessing.resampling import resample_data_or_seg
from torchvision.transforms import transforms
from utilities.color import apply_color_map, apply_heatmap
from model.probabilistic_unet2D import model_PU, Uentropy
import ttach as tta
    

class WeightedSum(nn.Module):
    def __init__(self, num_modalities):
        super(WeightedSum, self).__init__()
        self.weights = nn.Parameter(torch.randn(num_modalities))
        self.num_modalities = num_modalities

    def forward(self, outputs):
        normalized_weights = F.softmax(self.weights, dim=0)
        weighted_sum = sum(w * o for w, o in zip(normalized_weights, outputs))
        
        return weighted_sum
    

class MeanSum(nn.Module):
    def __init__(self, num_modalities):
        super(MeanSum, self).__init__()
        self.num_modalities = num_modalities

    def forward(self, outputs):
        # 计算平均权重
        average_weight = 1.0 / self.num_modalities
        mean_sum = sum(average_weight * o for o in outputs)
        
        return mean_sum
    
    
class MaxMinNorm(object):
    def __call__(self, sample):
        image = sample['image']
        label = sample['label']
        Max = np.max(image)
        Min = np.min(image)
        image = (image - Min) / (Max - Min)

        return {'image': image, 'label': label}
    

class ZScoreNorm(object):
    def __call__(self, sample):
        image = sample['image']
        label = sample['label']
        mean = image.mean()
        std = image.std()
        image = (image - mean) / (max(std, 1e-8))
        return {'image': image, 'label': label}


class RandomFlip(object):
    def __call__(self, sample):
        image = sample['image']
        label = sample['label']
        if random.random() < 0.5:
            image = np.flip(image, 1)
            label = np.flip(label, 0)
        if random.random() < 0.5:
            image = np.flip(image, 2)
            label = np.flip(label, 1)
        if random.random() < 0.5:
            image = np.flip(image, 3)
            label = np.flip(label, 2)

        return {'image': image, 'label': label}


class RandomCrop(object):
    def __init__(self, output_size=None):
        self.output_size = output_size
        
    def __call__(self, sample):
        image = sample['image']
        label = sample['label']
        if self.output_size is not None:
            c, h, w, d = image.shape
            assert c == 4, f'Input images should be channel-first, i.e., c = 4, but got c = {c}'
            new_h, new_w, new_d = self.output_size

            top = np.random.randint(0, h - new_h)
            left = np.random.randint(0, w - new_w)
            front = np.random.randint(0, d - new_d)

            image = image[:, top: top + new_h, left: left + new_w, front: front + new_d]
            label = label[top: top + new_h, left: left + new_w, front: front + new_d]

        return {'image': image, 'label': label}


class RandomIntensityShift(object):
    def __call__(self, sample, factor=0.1):
        image = sample['image']
        label = sample['label']

        scale_factor = np.random.uniform(1.0-factor, 1.0+factor, size=[image.shape[0], 1, image.shape[2], 1])
        shift_factor = np.random.uniform(-factor, factor, size=[image.shape[0], 1, image.shape[2], 1])

        image = image*scale_factor+shift_factor

        return {'image': image, 'label': label}


class RandomRotate(object):
    def __call__(self, sample):
        image = sample['image']
        label = sample['label']

        angle = round(np.random.uniform(-10, 10), 2)
        image = rotate(image, angle, axes=(1, 2), reshape=False)
        label = rotate(label, angle, axes=(1, 2), reshape=False)

        return {'image': image, 'label': label}


class Pad(object):
    def __call__(self, sample):
        image = sample['image']
        label = sample['label']

        image = np.pad(image, ((0, 0), (0, 0), (0, 0), (0, 5)), mode='constant')
        label = np.pad(label, ((0, 0), (0, 0), (0, 5)), mode='constant')
        
        return {'image': image, 'label': label}
    
    
class Resample(object):
    def __init__(self, new_shape):
        self.image_shape = new_shape
        
    def __call__(self, sample):
        image = sample['image']
        label = sample['label']

        image = resample_data_or_seg(image, new_shape=self.image_shape)
        label = resample_data_or_seg(np.expand_dims(label, axis=0), new_shape=self.image_shape, is_seg=True)
        label = label.squeeze(0)
        
        return {'image': image, 'label': label}
    

class ToTensor_test(object):
    def __call__(self, sample):
        image = sample['image']
        image = np.ascontiguousarray(image)
        label = sample['label']
        label = np.ascontiguousarray(label)
        spacing = sample['spacing']
        spacing = np.ascontiguousarray(spacing)

        image = torch.from_numpy(image).float()
        label = torch.from_numpy(label).long()
        spacing = torch.from_numpy(spacing).float()

        return {'image': image, 'label': label, 'spacing': spacing}
    
    
class ToTensor(object):
    def __call__(self, sample):
        image = sample['image']
        image = np.ascontiguousarray(image)
        label = sample['label']
        label = np.ascontiguousarray(label)

        image = torch.from_numpy(image).float()
        label = torch.from_numpy(label).long()

        return {'image': image, 'label': label}
    

import random
import torch
import cv2
import numbers
import numpy as np
from PIL import Image, ImageOps
from scipy.ndimage.filters import gaussian_filter
from scipy.ndimage.interpolation import map_coordinates
from scipy import ndimage


class RandomCenterCrop(object):
    def __init__(self, size):
        if isinstance(size, numbers.Number):
            self.size = (int(size), int(size))
        else:
            self.size = size # h, w
        self.padding = self.size[0]//2

    def __call__(self, sample):
        img, mask = sample['image'], sample['label']

        img = ImageOps.expand(img, border=self.padding, fill=0)
        mask = ImageOps.expand(mask, border=self.padding, fill=255)

        assert img.width == mask.width
        assert img.height == mask.height
        w, h = img.size
        th, tw = self.size # target size
        if w == tw and h == th:
            return {'image': img,
                    'label': mask,
                    'img_name': sample['img_name']}

        lab_array = np.array(mask)
        _mask = np.zeros([lab_array.shape[0], lab_array.shape[1]])
        _mask[lab_array > 200] = 255
        _mask[(lab_array > 50) & (lab_array < 201)] = 128
        index = np.where(_mask !=255)

        hight_center, width_center = (np.min(index[0])+np.max(index[0]))//2, (np.min(index[1])+np.max(index[1]))//2
        x1 = random.randint(0, 50)
        y1 = random.randint(0, 50)
        img = img.crop((width_center - tw//2 + x1, hight_center - th//2 + y1, width_center + tw//2 + x1, hight_center + th//2 + y1))
        mask = mask.crop((width_center - tw//2 + x1, hight_center - th//2 + y1, width_center + tw//2 + x1, hight_center + th//2 + y1))
        return {'image': img,
                'label': mask,
                'img_name': sample['img_name']}

class RandomScaleCenterCrop(object):
    def __init__(self, size):
        self.size = size
        self.crop = RandomCenterCrop(self.size)

    def __call__(self, sample):
        img = sample['image']
        mask = sample['label']
        name = sample['img_name']
        # print(img.size)
        assert img.width == mask.width
        assert img.height == mask.height

        seed = random.random()
        if seed > 0.5:
            w = int(random.uniform(0.8, 1.2) * img.size[0])
            h = int(random.uniform(0.8, 1.2) * img.size[1])

            img, mask = img.resize((w, h), Image.BILINEAR), mask.resize((w, h), Image.NEAREST)
            sample = {'image': img, 'label': mask, 'img_name': name}

        return self.crop(sample)
    

class Scale(object):
    def __init__(self, size):
        if isinstance(size, numbers.Number):
            self.size = (int(size), int(size))
        else:
            self.size = size # h, w

    def __call__(self, sample):
        img = sample['image']
        mask = sample['label']
        name = sample['img_name']
        # print(img.size)
        assert img.width == mask.width
        assert img.height == mask.height

        w = self.size[1]
        h = self.size[0]

        img, mask = img.resize((w, h), Image.BILINEAR), mask.resize((w, h), Image.NEAREST)
        sample = {'image': img, 'label': mask, 'img_name': name}

        return sample


class RandomRotate(object):
    def __init__(self, size=512):
        self.degree = random.randint(1, 4) * 90
        self.size = size

    def __call__(self, sample):
        img = sample['image']
        mask = sample['label']

        seed = random.random()
        if seed > 0.5:
            rotate_degree = self.degree
            img = img.rotate(rotate_degree, Image.BILINEAR, expand=0)
            mask = mask.rotate(rotate_degree, Image.NEAREST, expand=255)

            sample = {'image': img, 'label': mask, 'img_name': sample['img_name']}
        return sample

class RandomFlip(object):
    def __call__(self, sample):
        img = sample['image']
        mask = sample['label']
        name = sample['img_name']
        if random.random() < 0.5:
            img = img.transpose(Image.FLIP_LEFT_RIGHT)
            mask = mask.transpose(Image.FLIP_LEFT_RIGHT)
        if random.random() < 0.5:
            img = img.transpose(Image.FLIP_TOP_BOTTOM)
            mask = mask.transpose(Image.FLIP_TOP_BOTTOM)

        return {'image': img,
                'label': mask,
                'img_name': name
                }


class elastic_transform():
    """Elastic deformation of images as described in [Simard2003]_.
        .. [Simard2003] Simard, Steinkraus and Platt, "Best Practices for
           Convolutional Neural Networks applied to Visual Document Analysis", in
           Proc. of the International Conference on Document Analysis and
           Recognition, 2003.
        """

    # def __init__(self):

    def __call__(self, sample):
        image, label = sample['image'], sample['label']
        alpha = image.size[1] * 2
        sigma = image.size[1] * 0.08
        random_state = None
        seed = random.random()
        if seed > 0.5:
            # print(image.size)
            assert len(image.size) == 2

            if random_state is None:
                random_state = np.random.RandomState(None)

            shape = image.size[0:2]
            dx = gaussian_filter((random_state.rand(*shape) * 2 - 1), sigma, mode="constant", cval=0) * alpha
            dy = gaussian_filter((random_state.rand(*shape) * 2 - 1), sigma, mode="constant", cval=0) * alpha

            x, y = np.meshgrid(np.arange(shape[0]), np.arange(shape[1]), indexing='ij')
            indices = np.reshape(x + dx, (-1, 1)), np.reshape(y + dy, (-1, 1))

            transformed_image = np.zeros([image.size[0], image.size[1], 3])
            transformed_label = np.zeros([image.size[0], image.size[1]])

            for i in range(3):
                # print(i)
                transformed_image[:, :, i] = map_coordinates(np.array(image)[:, :, i], indices, order=1).reshape(shape)
                # break
            if label is not None:
                transformed_label[:, :] = map_coordinates(np.array(label)[:, :], indices, order=1, mode='nearest').reshape(shape)
            else:
                transformed_label = None
            transformed_image = transformed_image.astype(np.uint8)

            if label is not None:
                transformed_label = transformed_label.astype(np.uint8)

            return {'image': transformed_image,
                    'label': transformed_label,
                    'img_name': sample['img_name']}
        else:
            return {'image': np.array(sample['image']),
                    'label': np.array(sample['label']),
                    'img_name': sample['img_name']}


class add_salt_pepper_noise():
    def __call__(self, sample):

        image = np.array(sample['image']).astype(np.uint8)
        X_imgs_copy = image.copy()
        # row = image.shape[0]
        # col = image.shape[1]
        salt_vs_pepper = 0.2
        amount = 0.004

        num_salt = np.ceil(amount * X_imgs_copy.size * salt_vs_pepper)
        num_pepper = np.ceil(amount * X_imgs_copy.size * (1.0 - salt_vs_pepper))

        seed = random.random()
        if seed > 0.75:
            # Add Salt noise
            coords = [np.random.randint(0, i - 1, int(num_salt)) for i in X_imgs_copy.shape]
            X_imgs_copy[coords[0], coords[1], :] = 1
        elif seed > 0.5:
            # Add Pepper noise
            coords = [np.random.randint(0, i - 1, int(num_pepper)) for i in X_imgs_copy.shape]
            X_imgs_copy[coords[0], coords[1], :] = 0

        return {'image': X_imgs_copy,
                'label': sample['label'],
                'img_name': sample['img_name']}

class adjust_light():
    def __call__(self, sample):
        image = sample['image']
        seed = random.random()
        if seed > 0.5:
            gamma = random.random() * 3 + 0.5
            invGamma = 1.0 / gamma
            table = np.array([((i / 255.0) ** invGamma) * 255 for i in np.arange(0, 256)]).astype(np.uint8)
            image = cv2.LUT(np.array(image).astype(np.uint8), table).astype(np.uint8)
            return {'image': image,
                    'label': sample['label'],
                    'img_name': sample['img_name']}
        else:
            return sample


class eraser():
    def __call__(self, sample, s_l=0.02, s_h=0.06, r_1=0.3, r_2=0.6, v_l=0, v_h=255, pixel_level=False):
        image = sample['image']
        img_h, img_w, img_c = image.shape


        if random.random() > 0.5:
            return sample

        while True:
            s = np.random.uniform(s_l, s_h) * img_h * img_w
            r = np.random.uniform(r_1, r_2)
            w = int(np.sqrt(s / r))
            h = int(np.sqrt(s * r))
            left = np.random.randint(0, img_w)
            top = np.random.randint(0, img_h)

            if left + w <= img_w and top + h <= img_h:
                break

        if pixel_level:
            c = np.random.uniform(v_l, v_h, (h, w, img_c))
        else:
            c = np.random.uniform(v_l, v_h)

        image[top:top + h, left:left + w, :] = c

        return {'image': image,
                'label': sample['label'],
                'img_name': sample['img_name']}


class GetBoundary(object):
    def __init__(self, width = 5):
        self.width = width
    def __call__(self, mask):
        cup = mask[:, :, 0]
        disc = mask[:, :, 1]
        dila_cup = ndimage.binary_dilation(cup, iterations=self.width).astype(cup.dtype)
        eros_cup = ndimage.binary_erosion(cup, iterations=self.width).astype(cup.dtype)
        dila_disc= ndimage.binary_dilation(disc, iterations=self.width).astype(disc.dtype)
        eros_disc= ndimage.binary_erosion(disc, iterations=self.width).astype(disc.dtype)
        cup = dila_cup + eros_cup
        disc = dila_disc + eros_disc
        cup[cup==2]=0
        disc[disc==2]=0
        boundary = (cup + disc) > 0
        return boundary.astype(np.uint8)


def to_multilabel(pre_mask, classes = 2):
    mask = np.zeros((pre_mask.shape[0], pre_mask.shape[1], classes))
    mask[pre_mask == 1] = [0, 1]
    mask[pre_mask == 2] = [1, 1]
    return mask


class Normalize_tf(object):
    """Normalize a tensor image with mean and standard deviation.
    Args:
        mean (tuple): means for each channel.
        std (tuple): standard deviations for each channel.
    """
    def __init__(self, mean=(0., 0., 0.), std=(1., 1., 1.)):
        self.mean = mean
        self.std = std
        self.get_boundary = GetBoundary()

    def __call__(self, sample):
        img = np.array(sample['image']).astype(np.float32)
        __mask = np.array(sample['label']).astype(np.uint8)
        name = sample['img_name']
        img /= 127.5
        img -= 1.0
        _mask = np.zeros([__mask.shape[0], __mask.shape[1]])
        _mask[__mask > 200] = 255
        _mask[(__mask > 50) & (__mask < 201)] = 128

        __mask[_mask == 0] = 2
        __mask[_mask == 128] = 1
        __mask[_mask == 255] = 0


        # mask = to_multilabel(__mask)
        # boundary = self.get_boundary(mask) * 255
        # boundary = ndimage.gaussian_filter(boundary, sigma=3) / 255.0
        # boundary = np.expand_dims(boundary, -1)

        return {'image': img,
                'map': __mask,
                #'boundary': boundary,
                'img_name': name
               }


def transform_train(sample):
    transform = transforms.Compose([
        # Pad(),
        # ZScoreNorm(),
        # Random_rotate(),  # time-consuming
        # RandomCrop((64, 64, 64)),
        # RandomIntensityShift(),
        # Resample(new_shape=(96, 96, 96)),
        ToTensor()
    ])

    return transform(sample)


def transform_valid(sample):
    transform = transforms.Compose([
        # Pad(),
        # ZScoreNorm(),
        # MaxMinNorm(),
        # Resample(new_shape=(96, 96, 96)),
        # Resample(new_shape=(128, 128, 128)),
        # RandomCrop((64, 64, 64)),
        ToTensor()
    ])

    return transform(sample)


def transform_test(sample):
    transform = transforms.Compose([
        # RandomCrop((64, 64, 64)),
        ToTensor_test()
    ])

    return transform(sample)


def summarize_tensor(tensor, name="Tensor"):
    """
    Compute and print the statistical information of a tensor with 4 decimal places.
    
    Args:
    - tensor (torch.Tensor): The input tensor.
    - name (str): The name of the tensor, default is "Tensor".

    Output:
    Prints the mean, standard deviation, minimum, and maximum values.
    """
    mean = tensor.mean().item()
    std = tensor.std().item()
    min_val = tensor.min().item()
    max_val = tensor.max().item()
    
    print(f"Summary of {name}:")
    print(f"Mean: {mean:.4f}")
    print(f"Std: {std:.4f}")
    print(f"Min: {min_val:.4f}")
    print(f"Max: {max_val:.4f}")


def compute_gradient(image):
    if image.shape[1] == 3:
        to_grayscale = transforms.Grayscale(num_output_channels=1)
        image = to_grayscale(image)
    image = image.detach().cpu().numpy()
    if image.ndim == 5: 
        gradient = np.gradient(image, axis=(-3, -2, -1))
        value = np.sqrt(gradient[0]**2 + gradient[1]**2 + gradient[2]**2) + 1e-8 # [N,1,H,W,D]
        value = np.transpose(value, (0, 2, 3, 4, 1)).reshape(-1, 1) # [NHWD,1]
        return  torch.from_numpy(value)
    elif image.ndim == 4:
        gradient = np.gradient(image, axis=(-2, -1))
        value = np.sqrt(gradient[0]**2 + gradient[1]**2) + 1e-8  # [N,1,H,W]
        value = np.transpose(value, (0, 2, 3, 1)).reshape(-1, 1)  # [NHW,1]
        return torch.from_numpy(value)


def compute_distance_map(labels):
    """
    Input: labels [N, C, H, W, D]
    Output: distance2boundary [NHWD, 1]
    """
    labels = labels.cpu().detach().numpy()
    if labels.ndim == 5:
        N, C, H, W, D = labels.shape
        distance2boundary_batch = np.zeros((N, H, W, D))
        
        for n in range(N):
            boundary = np.zeros((H, W, D))
            for c in range(1, C):
                img_gt = labels[n, c, :, :, :]
                posmask = img_gt > 0
                boundary += skimage_seg.find_boundaries(posmask, connectivity=img_gt.ndim, mode='thick').astype(np.uint16)
            boundary_bool = boundary > 0
            distance2boundary_batch[n] = distance(~boundary_bool) # [N, H, W, D]
            distance2boundary = distance2boundary_batch.reshape(-1, 1) # [NHWD, 1]
    elif labels.ndim == 4:
        N, C, H, W = labels.shape
        distance2boundary_batch = np.zeros((N, H, W))
        
        for n in range(N):
            boundary = np.zeros((H, W))
            for c in range(1, C):
                img_gt = labels[n, c, :, :]
                posmask = img_gt > 0
                boundary += skimage_seg.find_boundaries(posmask, connectivity=img_gt.ndim, mode='thick').astype(np.uint16)
            boundary_bool = boundary > 0
            distance2boundary_batch[n] = distance(~boundary_bool) # [N, H, W]
            distance2boundary = distance2boundary_batch.reshape(-1, 1) # [NHW, 1]
    
    return distance2boundary
     

def save_args(args, filepath='config.json'):
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, 'w') as f:
        json.dump(vars(args), f, indent=4)
        

def load_args(filepath='config.json'):
    with open(filepath, 'r') as f:
        config = json.load(f)
    parser = argparse.ArgumentParser()
    for key, value in config.items():
        parser.add_argument(f'--{key}', default=value, type=type(value))
    args = parser.parse_args()
    return args


def generate_noisy_images(images, device, mu=0.5, sigma=0.3, seed=None):
    if seed is not None:
        torch.manual_seed(seed)
    images = images.to(device)
    _, _, H, W = images.shape
    noise = torch.normal(mean=mu, std=sigma, size=(H, W)).to(device)
    noisy_images = images + noise
    return noisy_images


def generate_blurred_images(images, device, sigma_blur=0.5):
    if sigma_blur==0.0:
        return images
    else:
        images = images.to(device)
        blurred_images = torch.zeros_like(images)
        for i in range(images.shape[0]):
            blurred_images[i] = torch.from_numpy(gaussian_filter(images[i].cpu().numpy(), sigma=sigma_blur))
        return blurred_images


def run_forward_to_get_u(model, images, num_classes, method='base', dataset='ACDC'):
    if method == 'base' or method == 'devis':
        if method == 'base':
            pred = model(images).permute(0, 2, 3, 1).contiguous().view(-1, num_classes) # [NHW,C]
            evidence = F.softplus(pred)
        elif method == 'devis':
            evidence = model(images).permute(0, 2, 3, 1).contiguous().view(-1, num_classes)
        alpha = evidence + 1
        S = torch.sum(alpha, dim=1, keepdim=True) # [NHW,1]
        u = (num_classes / S) # [NHW,1]
    elif method == 'pu' or method == 'flow' or method == 'glow' or method == 'udrop':
        if method == 'pu' or method == 'flow' or method == 'glow':
            if 'Refuge-no' in dataset:
                resized_images = F.interpolate(images, size=(256, 256), mode='bilinear', align_corners=False)
            else:
                resized_images = images
            logits = model_PU(resized_images, model)
            del resized_images
            if 'Refuge-no' in dataset:
                logits = F.interpolate(logits, size=(512, 512), mode='nearest')
            
        elif method == 'udrop':
            logits = model(images)
        
        if logits.ndim == 3:
            logits = logits.unsqueeze(0)
        u = Uentropy(logits, num_classes).view(-1, 1)
    else:
        if method == 'eu':
            for i in range(4):
                logits = model[i](images)
                if i == 0:
                    u = Uentropy(logits, num_classes).view(-1, 1)
                else:
                    u += Uentropy(logits, num_classes).view(-1, 1)
            u /= 4.
            
        elif method == 'tta':
            # defined 2 * 2 * 3 * 3 = 36 augmentations !
            transforms_img = tta.Compose(
                [
                    tta.HorizontalFlip(),
                    tta.Rotate90(angles=[0, 180]),
                    tta.Scale(scales=[1, 2, 4]),
                    tta.Multiply(factors=[0.9, 1, 1.1]),        
                ]
            )
            tta_model = tta.SegmentationTTAWrapper(model, transforms_img)
            logits = tta_model(images)
            u = Uentropy(logits, num_classes).view(-1, 1)
    return u


def sample_class_wise_noised_patch_images(images, distance_map, uncertainty_mask, label, device, k=3, mu_range=(0, 1.0), sigma=0.3, num_patch=4, threshold=4, seed=None):
    """
    为当前batch内的每个图像生成num_patch个随机正方形mask, 并在该区域内分别添加两种不同高斯噪声mu1, mu2, 其余区域不变
    
    参数：
    - images: 输入图像 (batch_size, C, H, W)
    - distance_map: 距离图, 用于均衡采样和输出mask中心点的距离,注意输入的是numpy (batch_size, H, W)
    - uncertainty_mask: 不确定性图, 用于选取整张图加噪后不符合假设的pixels, 作为hard samples (batch_size, H, W)
    - label: 独热编码的分割金标准label (batch_size, num_classes, H, W)
    - k: mask的边长
    - mu_range: 高斯噪声均值的取值范围(最小值, 最大值)
    - sigma: 高斯噪声的方差
    - num_patch: 每个类别生成的patch数量
    
    返回：
    - mu1: 第一个mask的高斯噪声均值
    - mu2: 第二个mask的高斯噪声均值
    - d: batch内每个图像mask中心点处的distance map的值
    - noisy_images_mu1: mu1, d1的高斯噪声图像
    - noisy_images_mu2: mu2, d2的高斯噪声图像
    - indexes: 每个类别分别添加的mask中心点索引
    """
    if seed is not None:
        random.seed(seed)
        torch.manual_seed(seed)
    distance_map_tensor = torch.from_numpy(distance_map).to(device)
    images = images.to(device)
    batch_size, C, H, W = images.shape
    num_classes = label.shape[1]

    # batch-wise加噪
    mu1 = random.uniform(*mu_range)
    noise1 = torch.normal(mean=mu1, std=sigma, size=(C, k, k)).to(device)
    mu2 = random.uniform(*mu_range)
    noise2 = torch.normal(mean=mu2, std=sigma, size=(C, k, k)).to(device)

    noised_images_mu1 = images.clone().detach()
    noised_images_mu2 = images.clone().detach()

    # 根据uncertainty_mask找到所有可能的中心点, slice-wise
    d = torch.zeros((batch_size, num_patch*num_classes)).to(device)
    indexes = []
    for i in range(batch_size):
        indexes_slice = []

        # 标记每张slice已加噪的区域
        noise_mask = torch.zeros_like(images, dtype=torch.bool)

        starts = []
        for c in range(num_classes):
            cls_starts = []
            # 根据每张图片的uncertainty_mask和label找到当前类别所有可能的中心点
            uncertainty_mask_img = uncertainty_mask[i].view(-1)
            label_img = label[i, c].view(-1)
            nonzero_indices = torch.nonzero(torch.logical_and(uncertainty_mask_img, label_img))

            if nonzero_indices.numel() == 0:
                possible_indices = torch.tensor([])  # 返回一个空的一维张量
            else:
                possible_indices = nonzero_indices.squeeze(1)  # 确保结果是一维张量
            
            if possible_indices.numel() == 0:
                # 没有可能的中心点，用None填充
                # print(f"Not enough patches satisfy the condition for image {i}, class {c}.")
                pass
                        
            else:
                # 随机打乱indices
                possible_indices = possible_indices[torch.randperm(len(possible_indices))]

            # 逐个遍历可能点
            for idx in possible_indices:
                y, x = divmod(int(idx), W)
                x_start, y_start = x - k // 2, y - k // 2
            
                # 检查是否超出边界
                if x_start < 0 or y_start < 0 or x_start + k > W or y_start + k > H:
                    continue

                # 检查背景加噪是否离边界太远
                if c==0 and distance_map_tensor[i, y_start + k // 2, x_start + k // 2] >= threshold:
                    continue

                # 检查是否已经加过噪声
                if not noise_mask[i, :, y_start:y_start + k, x_start:x_start + k].any():
                    cls_starts.append((x_start, y_start))
                    starts.append((x_start, y_start))
                    noise_mask[i, :, y_start:y_start + k, x_start:x_start + k] = True
                    
                # 若已找到满足条件的patch，则退出循环
                if len(cls_starts) >= num_patch:
                    break

        # # 检查是否采样足够数量
        while len(starts) < num_patch * num_classes:
            # 数量不够随机来凑
            x_start = random.randint(0, W - k)
            y_start = random.randint(0, H - k)

            # 检查是否超出边界
            if x_start < 0 or y_start < 0 or x_start + k > W or y_start + k > H:
                continue
            # 检查是否已经加过噪声
            if not noise_mask[i, :, y_start:y_start + k, x_start:x_start + k].any():
                starts.append((x_start, y_start))
                noise_mask[i, :, y_start:y_start + k, x_start:x_start + k] = True

        # 开始逐个patch加噪
        for j, (x_start, y_start) in enumerate(starts):
            # mu1, d
            noised_images_mu1[i, :, y_start:y_start + k, x_start:x_start + k] += noise1
            d[i, j] = distance_map_tensor[i, y_start + k // 2, x_start + k // 2]
            # mu2, d
            noised_images_mu2[i, :, y_start:y_start + k, x_start:x_start + k] += noise2

            indexes_slice.append((y_start + k // 2, x_start + k // 2))
            
        indexes.append(indexes_slice)

    return mu1, mu2, d, noised_images_mu1, noised_images_mu2, indexes
