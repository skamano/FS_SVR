import os
from pathlib import Path

import torch
import numpy as np
# import pandas as pd

from PIL import Image 
from torch.utils.data import IterableDataset
from torchvision import transforms, utils


class ShapeNetDataset(IterableDataset):

    def __init__(self, root_dir, transform=None):
        """
        Args:
            root_dir (string): root directory of ShapeNet renderings
        """
        self.root_dir = root_dir
        self.img_path = None
        self.transform = transform
        self.path_iterator = Path(self.root_dir).rglob('*.jpg')
        # http://shapenet.cs.stanford.edu/shapenet/obj-zip/ShapeNetCore.v2-old/shapenet/tex/TechnicalReport/main.pdf
        # see the link above for semantic labels and synset ids
        # FIXME: hardcoded semantic labels, should be able to specify which classes to train on
        self.synset_ids = {'telephone': 04401088, 'watercraft': 04530566, 'display': 03211117}


    def __iter__(self):
        return self

    def __next__(self):
        next_img = None
        while next_img is None or next_img.shape[0] != 3:  # makes sure each jpg file is an RGB image
            next_img = self.transform(Image.open(next(self.path_iterator)))

        return next_img 


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
        for path in Path(os.path.join(self.root_dir, synset_id, model_id)).rglob('*.jpg'):
            img = toTensor(Image.open(path))
            all_views.append(img)

        return all_views 


    def get_points(self, synset_id, model_id):
        return np.load(os.path.join(self.root_dir, synset_id, model_id, 'points.npz')) 

        