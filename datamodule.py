import os
import pytorch_lightning as pl

from torch.utils.data import DataLoader
import torchvision.transforms as transforms

from dataset import FacialKeypointsDataset
from transforms import Rescale, RandomCrop, Normalize, ToTensor

class FacialKeypointsDatamodule(pl.LightningDataModule):
    def __init__(self,
        train_csv_file = '/data/training_frames_keypoints.csv',
        test_csv_file = './data/test_frames_keypoints.csv',
        root_dir = './data/',
        batch_size = 32,
        num_workers = 0
    ):
        """Constructor

        Args:
            train_csv_file:     Path to the csv file with annotations for train data.
            test_csv_file:      Path to the csv file with annotations for test data.
            root_dir:           Directory with all the images.
            batch_size:         Batch size
        """
        super().__init__()

        self.train_csv_file = train_csv_file
        self.test_csv_file = test_csv_file
        self.root_dir = root_dir
        self.batch_size = batch_size
        self.num_workers = num_workers

    def train_dataloader(self):
        transformations = transforms.Compose([
            Rescale(250),
            RandomCrop(224),
            Normalize(),
            ToTensor()
        ])

        train_images_path = os.path.join(self.root_dir, 'training')

        train_dataset = FacialKeypointsDataset(
            self.train_csv_file,
            train_images_path,
            transform=transformations,
        )

        data_loader = DataLoader(train_dataset,
            batch_size = self.batch_size,
            shuffle = True,
            num_workers = self.num_workers)
        
        return data_loader

    def test_dataloader(self):
        transformations = transforms.Compose([
            Rescale(250),
            RandomCrop(224),
            Normalize(),
            ToTensor()
        ])

        train_images_path = os.path.join(self.root_dir, 'test')

        train_dataset = FacialKeypointsDataset(
            self.test_csv_file,
            train_images_path,
            transform=transformations
        )

        data_loader = DataLoader(train_dataset,
            batch_size = self.batch_size,
            shuffle = True,
            num_workers = self.num_workers)
        
        return data_loader
