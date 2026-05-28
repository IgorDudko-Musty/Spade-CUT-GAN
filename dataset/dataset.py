from pathlib import Path
from torch.utils.data import Dataset
import cv2
import random


class RGB2TIRData(Dataset):
    def __init__(self, path):
        super().__init__()
        train_RGB = Path(path) / "train_RGB"
        train_TIR = Path(path) / "train_TIR"

        self.train_RGB = [img for img in train_RGB.iterdir()]
        self.train_TIR = [img for img in train_TIR.iterdir()]
        random.shuffle(self.train_RGB)
        random.shuffle(self.train_TIR)

    def shuffle_ds(self):
        random.shuffle(self.train_RGB)
        random.shuffle(self.train_TIR)

    def __getitem__(self, index):
        rgb_path = self.train_RGB[index]
        tir_path = self.train_TIR[index]

        rgb = cv2.imread(rgb_path)
        rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)
        tir = cv2.imread(tir_path, cv2.IMREAD_GRAYSCALE)
        return rgb, tir

    def __len__(self):
        return len(self.train_RGB)
