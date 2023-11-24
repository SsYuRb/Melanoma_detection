from torch.utils.data import Dataset
from glob import glob
import os
from PIL import Image
from torchvision.datasets.utils import download_and_extract_archive, download_url
import pandas as pd
import os.path


class PH2Dataset(Dataset):
    url = 'http://fmb.images.gan4x4.ru/msu/usufov/PH2Dataset.zip'

    def __init__(self, root, transforms=None, mask_transforms = None, download = True):
        if download:
            download_and_extract_archive(self.url, root)
            root += "/PH2Dataset/PH2 Dataset images"
        print(f"{root}/*/")
        self.dirs = sorted(glob(f"{root}/*/", recursive = False))
        self.transforms = transforms
        self.mask_transforms = mask_transforms

    def __len__(self):
        return len(self.dirs)

    def __getitem__(self, n):
        dir = self.dirs[n]
        id = dir.split(os.sep)[-2]
        img = Image.open(f"{dir}{id}_Dermoscopic_Image/{id}.bmp")
        mask = Image.open(f"{dir}{id}_lesion/{id}_lesion.bmp")
        if self.transforms is not None:
            img = self.transforms(img)
        if self.mask_transforms is not None:
            mask = self.mask_transforms(mask)
        return img, mask


class ISICDataset(Dataset):

    def __init__(self, root, train=False, download=True, transforms=None, mask_transforms=None):
        self.masks = None
        self.images = None
        self.labels = None
        self.parts = {
            2016: {
                'train': {
                    'labels': 'https://isic-challenge-data.s3.amazonaws.com/2016/ISBI2016_ISIC_Part3B_Training_GroundTruth.csv',
                    'images': 'https://isic-challenge-data.s3.amazonaws.com/2016/ISBI2016_ISIC_Part3_Training_Data.zip',
                    'masks': 'https://isic-challenge-data.s3.amazonaws.com/2016/ISBI2016_ISIC_Part1_Training_GroundTruth.zip'
                    },
                'test': {
                    'labels': 'https://isic-challenge-data.s3.amazonaws.com/2016/ISBI2016_ISIC_Part3_Test_GroundTruth.csv',
                    'images': 'https://isic-challenge-data.s3.amazonaws.com/2016/ISBI2016_ISIC_Part1_Test_Data.zip',
                    'masks': 'https://isic-challenge-data.s3.amazonaws.com/2016/ISBI2016_ISIC_Part1_Test_GroundTruth.zip'

                }
            }
        }
        self.download(train, root)

        self.validate()
        self.labels = self.labels.iloc[:, 1:].values
        self.transforms = transforms
        self.mask_transforms = mask_transforms

    def validate(self):
        assert len(self.images) == len(self.masks)
        assert len(self.images) == len(self.labels)
        for i, row in self.labels.iterrows():
            img_id = row[0]
            # if element not found index method raise an ValueError
            self.images[i].index(img_id)
            self.masks[i].index(img_id)

    def download(self, train, root):
        key = 'train' if train else 'test'
        urls = self.parts[2016][key]

        # Labels
        download_url(self.urls['labels'], root)
        self.labels = pd.read_csv(os.path.basename(urls['labels']), header=None)

        # Images
        download_and_extract_archive(self.urls['images'], root)
        folder = os.path.splitext(os.path.basename(urls['images']))[0]
        self.images = sorted(glob(f"{root}/{folder}/*.jpg", recursive=False))

        # Masks
        folder = os.path.splitext(os.path.basename(urls['masks']))[0]
        download_and_extract_archive(self.urls['masks'], root)
        self.masks = sorted(glob(f"{root}/{folder}/*.png", recursive=False))

    def __len__(self):
        return len(self.images)

    def __getitem__(self, n):
        img = Image.open(self.images[n])
        mask = Image.open(self.masks[n])
        label = self.get_label(n)
        if self.transforms is not None:
            img = self.transforms(img)
        if self.mask_transforms is not None:
            mask = self.mask_transforms(mask)
        return img, mask, label

    def get_label(self, n):
        label = self.labels[n]
        if label == 'benign':
            return 0
        if label == 'malignant':
            return 1
        return int(label)
