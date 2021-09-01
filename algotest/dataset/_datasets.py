from torch.utils.data.dataset import Dataset
import cv2
import numpy as np
from PIL import Image

class MyDataset(Dataset):
    def __init__(self, dataframe, labelmap, transform=None):
        self.df = dataframe
        if transform is not None:
            self.transform = transform
        self.labelmap = labelmap

    def __len__(self):
        return self.df.shape[0]

    def __getitem__(self, idx):
        imgPath = self.df["image"].iloc[idx]
        label = self.df["code"].iloc[idx]
        try:
            img = cv2.imdecode(np.fromfile(imgPath, dtype=np.uint8), -1)
            img =cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
            img= Image.fromarray(img.astype('uint8'))
        except:
            print(imgPath)
            raise ValueError("The image path is incorrect, please check the image path.")
        if self.transform:
            img = self.transform(img=img)

        return imgPath, img, self.labelmap[label]