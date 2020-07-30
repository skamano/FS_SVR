import os
import math
from pathlib import Path

import torch
import numpy as np
import itertools
# import pandas as pd

from PIL import Image 
from torch.utils.data import Dataset, IterableDataset, DataLoader
from torchvision import transforms, utils
from torchvision.datasets import ImageFolder


# class ShapeNetDataset(Dataset):
class ShapeNetDataset(IterableDataset):
    def __init__(self, root_dir, transform=None, split='train'):
        """
        Args:
            root_dir (string): root directory of ShapeNet renderings
        """
        super(ShapeNetDataset).__init__()
        self.root_dir = root_dir
        self.img_path = None
        self.transform = transform
        self.path_iterator = Path(self.root_dir).rglob('*.jpg')
        # http://shapenet.cs.stanford.edu/shapenet/obj-zip/ShapeNetCore.v2-old/shapenet/tex/TechnicalReport/main.pdf
        # see the link above for semantic labels and synset ids
        # FIXME: hardcoded semantic labels, should be able to specify which classes to train on
        self.base_synset_ids = {'telephone': '04401088', 'display': '03211117'}
        # iterators = [Path(os.path.join(self.root_dir, synset_id)).rglob('*.png') for synset_id in \
                    #self.base_synset_ids.items()]
        # self.base_classes_iterator = Path(self.root_dir).glob('{04401088,04530566}/*/*/*.png')
        it_telephone = list(Path(os.path.join(self.root_dir, '04401088')).glob('**/*.jpg'))
        it_watercraft = list(Path(os.path.join(self.root_dir, '03211117')).glob('**/*.jpg'))
        self.base_classes_paths = it_telephone + it_watercraft
        self.base_classes_iterator = iter(self.base_classes_paths)
        self.novel_synset_ids = {'watercraft': '04530566'}
        self.current_path = None
    
    # def __getitem__(self, idx):
    #     next_img = None
    #     next_img = self.transform(Image.open(self.base_classes_paths[idx]))

    #     return next_img 
    def __iter__(self):
        return self
        # worker_info = torch.utils.data.get_worker_info()
        # if worker_info is None:  # single-process dataloading
            # return self
        # else:
        #     dataset = worker_info.dataset
        #     overall_start = 0
        #     overall_end = len(self.base_classes_paths) - 1
        #     per_worker = int(math.ceil((overall_end - overall_start) / float(worker_info.num_workers)))
        #     worker_id = worker_info.id
        #     dataset.start = overall_start + worker_id * per_worker
        #     dataset.end = min(dataset.start + per_worker, overall_end)
        #     dataset.base_classes_iterator = iter(self.base_classes_paths[dataset.start:dataset.end+1])
        #     return dataset

    def __next__(self):
        # get image from base classes
        # worker_info = torch.utils.data.get_worker_info()
        # dataset = worker_info.dataset
        next_img = None
        while next_img is None or next_img.shape[0] != 3:  # makes sure each jpg file is an RGB image
            # filter data by synset ids 
            # if next_img is not None:
                # print("File skipped; invalid image format (not RGB)")
            try:
                self.current_path = next(self.base_classes_iterator)
            except StopIteration:
                raise StopIteration
            next_img = self.transform(Image.open(self.current_path))
        # while next_img is None or next_img.shape[0] != 3:  # makes sure each jpg file is an RGB image
        #     # filter data by synset ids 
        #     next_path = next(self.path_iterator)
        #     (synset_id, model_id) = next_path.parts[-4], next_path.parts[-3]
        #     if synset_id in self.base_synset_ids.items():
        #         next_img = self.transform(Image.open(next(self.path_iterator)))
        # print(self.current_path)
        return next_img

    def worker_init_fn(self, worker_id):
        worker_info = torch.utils.data.get_worker_info()
        dataset = worker_info.dataset
        overall_start = 0
        overall_end = len(self.base_classes_paths) - 1
        per_worker = int(math.ceil((overall_end - overall_start) / float(worker_info.num_workers)))
        worker_id = worker_info.id
        dataset.start = overall_start + worker_id * per_worker
        dataset.end = min(dataset.start + per_worker, overall_end)
        dataset.base_classes_iterator = iter(self.base_classes_paths[dataset.start:dataset.end+1])


    def get_views(self, synset_id, model_id):
        """
        Retrieves multiple RGB views for a single object.

        Arguments:
            synset_id (string): eight-digit zero-padded code corresponding to class label for an object
            model_id (string): longer name consisting of alphanumeric characters [0-9, a-f]

        Example:
        synset_id = '02691156' 
        model_id = '10155655850468db78d106ce0a280f87' 

        return:
            list of all view images for each object in torch.Tensor format
        """
        all_views = []
        toTensor = transforms.toTensor()
        for path in Path(os.path.join(self.root_dir, synset_id, model_id)).glob('**/*.jpg'):
            img = toTensor(Image.open(path))
            all_views.append(img)

        return all_views 


    def get_points(self, synset_id, model_id):
        return np.load(os.path.join(self.root_dir, synset_id, model_id, 'points.npz')) 

