# Copyright (c) EEEM071, University of Surrey

import os.path as osp

from PIL import Image
from torch.utils.data import Dataset
from src.multimodal import Fuse_RGB_Gray_Sketch

def read_image(img_path):
    """Keep reading image until succeed.
    This can avoid IOError incurred by heavy IO process."""
    got_img = False
    if not osp.exists(img_path):
        raise OSError(f"{img_path} does not exist")
    while not got_img:
        try:
            img = Fuse_RGB_Gray_Sketch(Image.open(img_path))
            got_img = True
        except OSError:
            print(
                f'IOError incurred when reading "{img_path}". Will redo. Don\'t worry. Just chill.'
            )
            pass
    return img


class ImageDataset(Dataset):
    """Image Person ReID Dataset"""

    def __init__(self, dataset):
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        img_path, pid, camid = self.dataset[index]
        img = read_image(img_path)
        return img, pid, camid, img_path
