import albumentations
import torch
import numpy as np

from PIL import Image
from PIL import ImageFile

import config

ImageFile.LOAD_TRUNCATED_IMAGES = True

class ClassificationDataset:
    def __init__(self, image_paths, targets, resize):
        # resize : (h, w)
        self.image_paths = image_paths
        self.targets = targets
        self.resize = resize
        self.aug = albumentations.Compose([
            albumentations.Normalize(always=True)
        ])

    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, item):
        image = Image.open(self.image_paths[item]).convert("RGB")
        targets = self.targets[item]

        if self.resize is not None:
            image = image.resize((self.resize[1], self.resize[0]), resample=Image.BILINEAR)

        image = np.array(image)
        augmented = self.aug(image=image)
        image = augmented["image"]
        image = np.transpose(image, (2, 0, 1)).astype(np.float32)
        return {
            "images": torch.tensor(image, dtype=torch.float32),
            "targets": torch.tensor(targets, dtype=torch.float32)
        }


if __name__ == "__main__":
    image_paths = ["captcha_images_v2/2b827.png", "captcha_images_v2/2bg48.png"]
    targets = [[1, 2, 3, 4, 5], [6, 7, 8, 9, 10]]
    dataset = ClassificationDataset(image_paths, targets, resize=(config.IMAGE_HEIGHT, config.IMAGE_WIDTH))
    for i in range(len(dataset)):
        sample = dataset[i]
        print(i, sample["images"].shape, sample["targets"].shape)