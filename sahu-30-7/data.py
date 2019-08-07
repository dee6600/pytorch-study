from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
import pandas as pd
import numpy as np

class ImageDataset(Dataset):
    def __init__(self, path):
        super(ImageDataset, self).__init__()
        csv_paths_df = pd.read_csv(path, sep=" ", header=None,)
        csv_paths_df.columns = ["path", "label"]
        required_classes = [10, 1, 11, 13, 5, 14, 7, 0]
        d = dict(enumerate(required_classes))
        d = dict((v, k) for k, v in d.items())

        csv_paths_df = csv_paths_df.loc[csv_paths_df["label"].isin(required_classes)]
        csv_paths_df = csv_paths_df.replace(d)

        self.fnames = list(csv_paths_df.iloc[:, 0])
        self.labels = list(csv_paths_df.iloc[:, 1])

        # Applying Transforms to the Data
        fn = lambda x: transforms.Pad((500-x.size[0]//2, 500-x.size[1]//2, 500-x.size[0]//2, 500-x.size[1]//2))(x)   # lamda input: function of input   
        self.transform = transforms.Compose([
            fn,
            transforms.Resize((1000,1000)),
            transforms.RandomRotation(degrees=20),
            transforms.ToTensor(),
        ])

    def __len__(self):
        return len(self.fnames)
    
    def __getitem__(self, idx):
        img = self.transform(Image.open('D:\\imp_data\\images\\'+self.fnames[idx]).convert("RGB")) # L does the grey scale
        label = self.labels[idx]
        return img, label

if __name__ == '__main__':

    path = "D:\\imp_data\\labels\\train.txt"
    dataset = ImageDataset(path)
    x, y = dataset[165]
    print(x, x.shape, y)
    print(len(dataset)//12)