import collections

import json
import os

import numpy as np
from torch.utils.data import Dataset

from PIL import Image
from PIL import ImageFile
import torch
ImageFile.LOAD_TRUNCATED_IMAGES = True
Image.MAX_IMAGE_PIXELS = None
from torchvision import transforms

from dataset.utils import pre_caption


class re_train_dataset(Dataset):
    def __init__(self, ann_file, transform, image_root, max_words=60):
        self.ann = []

        self.data_name = image_root.split('/')[-2]

        self.img_fea = np.load(f'/precomp/{self.data_name}_precomp/train_image_embed_regu.npy')
        self.text_fea = np.load(f'/precomp/{self.data_name}_precomp/train_text_embed_regu.npy')

        if len(ann_file) > 1:
            ann_tmp = json.load(open(ann_file[0], 'r'))
            id2num = collections.defaultdict(int)
            for d in ann_tmp:
                id2num[d['image']] += 1
                if id2num[d['image']] > 5:
                    continue
                self.ann.append(d)

            del ann_tmp, id2num

            self.val_ann = json.load(open(ann_file[1], 'r'))
            val_st_id = 1000000001
            for val in self.val_ann:
                caps = val['caption']
                for c in caps[:5]:
                    self.ann.append({
                        'caption': c,
                        'image': val['image'],
                        'image_id': val_st_id + 1
                    })
                val_st_id += 1
        else:
            for f in ann_file:
                self.ann += json.load(open(f, 'r'))
        self.transform = transform
        self.image_root = image_root
        self.max_words = max_words
        self.img_ids = {}

        n = 0
        for ann in self.ann:
            img_id = ann['image_id']
            if img_id not in self.img_ids.keys():
                self.img_ids[img_id] = n
                n += 1
    
        self.trans2 = transforms.Compose([
            transforms.RandomResizedCrop(384, scale=(0.5, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def __len__(self):
        return len(self.ann)

    def __getitem__(self, index):

        ann = self.ann[index]

        single_text_embed = self.text_fea[index]
        single_image_embed = self.img_fea[index]

        image_path = os.path.join(self.image_root, ann['image'])
        image = Image.open(image_path).convert('RGB')
        image1 = self.transform(image)

        caption = pre_caption(ann['caption'], self.max_words)
        return image1, caption, self.img_ids[ann['image_id']], single_text_embed, single_image_embed
    

class re_eval_dataset(Dataset):
    def __init__(self, ann_file, transform, image_root, max_words=30, coco_1k=False):
        self.ann = json.load(open(ann_file, 'r'))
        self.data_name = image_root.split('/')[-2]

        self.transform = transform
        self.image_root = image_root
        self.max_words = max_words
        if coco_1k:
            self.ann = self.ann[:1000]

        self.text = []
        self.image = []
        self.txt2img = {}
        self.img2txt = {}

        txt_id = 0
        for img_id, ann in enumerate(self.ann):
            self.image.append(ann['image'])
            self.img2txt[img_id] = []
            for i, caption in enumerate(ann['caption']):
                self.text.append(pre_caption(caption, self.max_words))
                self.img2txt[img_id].append(txt_id)
                self.txt2img[txt_id] = img_id
                txt_id += 1
                if len(self.img2txt[img_id]) == 5:
                    break

    def __len__(self):
        return len(self.image)

    def __getitem__(self, index):
        image_path = os.path.join(self.image_root, self.ann[index]['image'])
        image = Image.open(image_path).convert('RGB')
        image = self.transform(image)
        return image, index


class re_eval_dataset_test(Dataset):
    def __init__(self, ann_file, transform, image_root, max_words=30, coco_1k=False):
        self.ann = json.load(open(ann_file, 'r'))
        self.data_name = image_root.split('/')[-2]

        self.transform = transform
        self.image_root = image_root
        self.max_words = max_words
        if coco_1k:
            self.ann = self.ann[:1000]

        self.text = []
        self.image = []
        self.txt2img = {}
        self.img2txt = {}
        txt_id = 0
        for img_id, ann in enumerate(self.ann):
            self.image.append(ann['image'])
            self.img2txt[img_id] = []
            for i, caption in enumerate(ann['caption']):
                self.text.append(pre_caption(caption, self.max_words))
                self.img2txt[img_id].append(txt_id)
                self.txt2img[txt_id] = img_id
                txt_id += 1
                if len(self.img2txt[img_id]) == 5:
                    break

    def __len__(self):
        return len(self.image)

    def __getitem__(self, index):
        image_path = os.path.join(self.image_root, self.ann[index]['image'])
        image = Image.open(image_path).convert('RGB')
        image = self.transform(image)
        return image, index, self.ann[index]['image']