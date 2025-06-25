import os
import glob
import torch
import random
import numpy as np
import SimpleITK as sitk
from PIL import Image, ImageOps
from preprocessing.resampling import resample_data_or_seg
from utilities.utils import RandomCenterCrop, RandomFlip, RandomRotate, RandomScaleCenterCrop, \
    eraser, elastic_transform, add_salt_pepper_noise, adjust_light, Normalize_tf, ToTensor, Scale
from torchvision import transforms
from torch.utils.data import Dataset
    

def obtain_filenames_from_id_list(datapath, id_list=None):
    Files = []
    if id_list is not None:
        for id in id_list:
            Files.extend(glob.glob(os.path.join(datapath, '*%03d' % id, '*_frame??.nii*')))
    else:
        Files.extend(glob.glob(os.path.join(datapath, '*', '*_frame??.nii*')))
    return Files


def Generate_ACDC_Train_Val_Test_List(datapath, val_ratio=0.2): 
    np.random.seed(1)
    Train_Files, Vali_Files, Test_Files = [], [], []
    NUM_FOLDS = 5
    PATIENTS_EACH_FOLD = 20
    if 'ACDC' in datapath:
        train_num_per_class = round(PATIENTS_EACH_FOLD * (1 - val_ratio))
        id_numpys = np.random.permutation(np.arange(1, 101).reshape(5, 20).transpose())
        for cv in range(NUM_FOLDS):
            vali_id_list_cv = id_numpys[4 * cv:4 * (cv + 1), :].reshape(-1).tolist()
            train_id_array_cv = np.random.permutation(
                np.delete(id_numpys, np.arange(4 * cv, 4 * (cv + 1)).tolist(), axis=0))
            train_id_list_cv = train_id_array_cv[0:train_num_per_class, :].reshape(-1).tolist()
            vali_filenames_list = obtain_filenames_from_id_list(datapath, vali_id_list_cv)
            train_filenames_list = obtain_filenames_from_id_list(datapath, train_id_list_cv)
            Train_Files.append(train_filenames_list)
            Vali_Files.append(vali_filenames_list)
            
        test_filenames_list = obtain_filenames_from_id_list(datapath.replace('train', 'test'))
        Test_Files.append(test_filenames_list)

    return Train_Files, Vali_Files, Test_Files


def Generate_Refuge_Train_Val_Test_List(dir_path, seed=1, shuffle=False):
    np.random.seed(seed)
    Train_imgs_list = glob.glob(dir_path + '/train/images/Glaucoma/*')
    Val_imgs_list = glob.glob(dir_path + '/validation/images/*')
    Test_imgs_list = glob.glob(dir_path + '/test/images/*')
    if shuffle:
        np.random.shuffle(Train_imgs_list)
    return Train_imgs_list, Val_imgs_list, Test_imgs_list


class ACDC2017_Dataset(Dataset):
    def __init__(self, ImgFiles, num_classes: int = 4):
        self.ImgFiles = ImgFiles
        self.ori_size = 128
        self.num_classes = num_classes
        imgs = np.zeros((1, 128, 128))
        labs = np.zeros((1, 128, 128))
        spacings = np.zeros((1, 2))
        
        sitk.ProcessObject_SetGlobalWarningDisplay(False)
        
        for imgfile in self.ImgFiles:
            itkimg = sitk.ReadImage(imgfile)
            
            npimg = sitk.GetArrayFromImage(itkimg)  # Z,Y,X,
            npimg = (npimg - npimg.min()) / (npimg.max() - npimg.min())
            npimg = (npimg - npimg.mean()) / npimg.std()
            npimg = np.pad(npimg, ((0, 0), (50, 50), (50, 50)), 'minimum')

            itklab = sitk.ReadImage(imgfile.replace('.nii', '_gt.nii'))
            nplab = sitk.GetArrayFromImage(itklab)  # (H, W, D)
            nplab = np.pad(nplab, ((0, 0), (50, 50), (50, 50)), 'minimum')

            index = np.where(nplab != 0)

            npimg = npimg[:,
                    (np.min(index[1]) + np.max(index[1])) // 2 - 64:(np.min(index[1]) + np.max(index[1])) // 2 + 64,
                    (np.min(index[2]) + np.max(index[2])) // 2 - 64:(np.min(index[2]) + np.max(index[2])) // 2 + 64]
            nplab = nplab[:,
                    (np.min(index[1]) + np.max(index[1])) // 2 - 64:(np.min(index[1]) + np.max(index[1])) // 2 + 64,
                    (np.min(index[2]) + np.max(index[2])) // 2 - 64:(np.min(index[2]) + np.max(index[2])) // 2 + 64]
            
            spacing = np.array(itkimg.GetSpacing()).reshape(1, 3)[:, :-1]
            spacing = np.repeat(spacing, npimg.shape[0], axis=0)
            
            imgs = np.concatenate((imgs, npimg), axis=0)
            labs = np.concatenate((labs, nplab), axis=0)
            spacings = np.concatenate((spacings, spacing), axis=0)

        self.imgs = imgs[1:, :, :]
        self.labs = labs[1:, :, :]
        self.spacings = spacings[1:, :]
        self.imgs = self.imgs.astype(np.float32)  # [366, 128, 128]
        self.labs = self.labs.astype(np.uint8)
        self.spacings = self.spacings.astype(np.float32)

    def __len__(self):
        return self.imgs.shape[0]

    def __getitem__(self, item):
        img = self.imgs[item]
        lab = self.labs[item]
        spacing = self.spacings[item]
        
        img = img.copy()  # (128, 128)
        img = torch.from_numpy(img).unsqueeze(0).type(dtype=torch.FloatTensor)  # (1, 128, 128)
        
        lab = lab.copy()  # (128, 128)
        one_hot_lab = np.eye(self.num_classes)[lab]  # (128, 128) → (128, 128, 4)
        one_hot_lab = np.transpose(one_hot_lab, (2, 0, 1))  # (4, 128, 128)
        lab = torch.from_numpy(one_hot_lab).type(dtype=torch.LongTensor)  # (4, 128, 128)
        
        spacing = spacing.copy() # (2,)
        spacing = torch.from_numpy(spacing).type(dtype=torch.FloatTensor)  # (1, 2)

        return img, lab, spacing
    

class Refuge_Dataset(Dataset):
    def __init__(self, ImgFiles, times, num_classes: int = 3, train: bool = True):
        self.ImgFiles = ImgFiles
        self.transforms = transforms.Compose([
            RandomScaleCenterCrop(512),
            # Scale(256),
            RandomRotate(),
            RandomFlip(),
            elastic_transform(),
            # add_salt_pepper_noise(),
            adjust_light(),
            # eraser(),
            Normalize_tf(),
            ToTensor()
            ])
        self.time = times
        self.num_classes = num_classes
        self.train = train


    def __len__(self):
        return len(self.ImgFiles) * self.time

    def __getitem__(self, item):
        item, _ = divmod(item, self.time)
        _img = Image.open(self.ImgFiles[item]).convert('RGB')
        _target = Image.open(self.ImgFiles[item].replace('images', 'labels').replace('jpg', 'bmp'))
        if _target.mode == 'RGB':
            _target = _target.convert('L')
        _img_name = self.ImgFiles[item].split('/')[-1]
        spacing = (1.0, 1.0)

        if self.train:
            anco_sample = {'image': _img, 'label': _target, 'img_name': _img_name}
            anco_sample = self.transforms(anco_sample)
        
            one_hot_lab = np.eye(self.num_classes)[anco_sample['map']]  # (128, 128) → (128, 128, 4)
            one_hot_lab = np.transpose(one_hot_lab, (2, 0, 1))
            lab = torch.from_numpy(one_hot_lab).type(dtype=torch.LongTensor)  # (4, 128, 128)
            return anco_sample['image'], lab, spacing  # [3, 512, 512], [4, 512, 512]
        
        else:
            tw, th = 512, 512
            img_pad = ImageOps.expand(_img, border=320, fill=0)
            mask_pad = ImageOps.expand(_target, border=320, fill=255)

            img_pad = np.array(img_pad).astype(np.float32)  # H * W * C
            mask_pad = np.array(mask_pad).astype(np.uint8)

            _mask = np.zeros([mask_pad.shape[0], mask_pad.shape[1]])
            _mask[mask_pad > 200] = 255
            _mask[(mask_pad > 50) & (mask_pad < 201)] = 128

            mask_pad[_mask == 0] = 2
            mask_pad[_mask == 255] = 0
            mask_pad[_mask == 128] = 1

            mask = mask_pad[320:-320, 320:-320]

            index = np.where(mask_pad != 0)
            hight_center, width_center = (np.min(index[0]) + np.max(index[0])) // 2, (np.min(index[1]) + np.max(index[1])) // 2

            img_crop = img_pad[hight_center - th // 2 : hight_center + th // 2 , width_center - tw // 2 : width_center + tw // 2 ,: ]
            mask_crop = mask_pad[hight_center - th // 2 : hight_center + th // 2 , width_center - tw // 2 : width_center + tw // 2]

            img_crop /= 127.5
            img_crop -= 1.0
            img_input = img_crop.transpose((2, 0, 1))
            img_input = torch.from_numpy(img_input).float()

            one_hot_lab = np.eye(self.num_classes)[mask_crop]  # (128, 128) → (128, 128, 4)
            one_hot_lab = np.transpose(one_hot_lab, (2, 0, 1))
            lab = torch.from_numpy(one_hot_lab).type(dtype=torch.LongTensor)

            return img_input, lab, spacing  # [3, 512, 512], [4, 512, 512]
