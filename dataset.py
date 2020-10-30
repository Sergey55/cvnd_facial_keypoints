import os
import pandas as pd
from torch.utils.data import Dataset
from PIL import Image

class FacialKeypointsDataset(Dataset):
    """Face Landmarks dataset."""

    def __init__(self, csv_file, root_dir, transform):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform: Transform to be applied on a sample.
        """
        super().__init__()

        self.key_pts_frame = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform

    def __getitem__(self, idx):
        image_path = os.path.join(self.root_dir,
                                self.key_pts_frame.iloc[idx, 0])

        # Load image and convert to graysacel
        image = Image.open(image_path).convert('L')
        image = self.transform(image)
        
        key_points = self.key_pts_frame.iloc[idx, 1:].as_matrix()
        key_points = key_points.astype('float')

        return image, key_points

    def __len__(self):
        return len(self.key_pts_frame)        