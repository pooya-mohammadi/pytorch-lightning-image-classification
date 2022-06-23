import torch
from torch.utils.data import Dataset
import cv2
from sklearn.model_selection import train_test_split
from deep_utils import crawl_directory_dataset


class ImageClassificationDataset(Dataset):
    def __init__(self, image_list, label_list, transform=None, class_to_id=None):
        self.images = image_list
        self.labels = label_list
        self.transform = transform
        self.class_to_id = class_to_id

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        image_path = self.images[idx]
        img = cv2.imread(image_path)[..., ::-1]  # bgr2rgb
        if self.transform:
            img = self.transform(image=img)["image"]
        label_name = self.labels[idx]
        label = torch.tensor(self.class_to_id[label_name]).type(torch.long)
        sample = (img, label)
        return sample

    @staticmethod
    def get_loaders(config):
        x, y = crawl_directory_dataset(config.dataset_dir)
        x_train, x_val, y_train, y_val = train_test_split(x, y,
                                                          test_size=config.validation_size,
                                                          stratify=y)
        class_to_id = {v: k for k, v in enumerate(set(y_train))}
        train_dataset = ImageClassificationDataset(x_train, y_train, transform=config.train_transform,
                                                   class_to_id=class_to_id)
        train_loader = torch.utils.data.DataLoader(train_dataset,
                                                   batch_size=config.batch_size,
                                                   shuffle=True,
                                                   num_workers=config.n_workers,
                                                   pin_memory=config.pin_memory
                                                   )

        val_dataset = ImageClassificationDataset(x_val, y_val, transform=config.val_transform, class_to_id=class_to_id)
        val_loader = torch.utils.data.DataLoader(val_dataset,
                                                 batch_size=config.batch_size,
                                                 shuffle=False,
                                                 num_workers=config.n_workers,
                                                 pin_memory=config.pin_memory
                                                 )

        return train_loader, val_loader
