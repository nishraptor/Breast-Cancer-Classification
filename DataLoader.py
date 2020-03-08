import torchvision.transforms as transform
import pandas as pd
from torch.utils import data

class Dataset(data.Dataset):
    """
    Args:
        csv_file (string): Path to the csv file with annotations.
        root_dir (string): Directory with all the images.
        transform (callable, optional): Optional transform to be applied
            on a sample.
    """
    def __init__(self, csv_file, transform = None):
        'Initialization'
        self.frame = pd.read_csv(csv_file)
        self.transform = transform

    def __len__(self):
        'Denotes the total number of samples'
        return len(self.frame)

    def __getitem__(self, index):
        'Generates one sample of data'
        # Select sample
        img_name = self.frame.iloc[index, 6]
        image = io.imread(img_name)

        label = self.frame.iloc[index, 3]
        # Load data and get label

        sample = {'image': image, 'label': label}

        if self.transform:
            sample = self.transform(sample)

        return sample
