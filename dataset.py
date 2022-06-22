from os.path import join
import torch
from torch.utils.data import Dataset
import cv2
from sklearn.model_selection import train_test_split
from deep_utils import crawl_directory_dataset, dump_pickle
from settings import Config


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
    def get_loaders(output_dir=None):
        x, y = crawl_directory_dataset(Config.dataset_dir)
        x_train, x_val, y_train, y_val = train_test_split(x, y,
                                                          test_size=Config.validation_size,
                                                          stratify=y)
        class_to_id = {v: k for k, v in enumerate(set(y_train))}
        id_to_class = {v: k for k, v in class_to_id.items()}
        if output_dir:
            dump_pickle(join(output_dir, 'labels_map.pkl'), id_to_class)
        train_dataset = ImageClassificationDataset(x_train, y_train, transform=Config.train_transform,
                                                   class_to_id=class_to_id)
        train_loader = torch.utils.data.DataLoader(train_dataset,
                                                   batch_size=Config.batch_size,
                                                   shuffle=True,
                                                   num_workers=Config.n_workers,
                                                   )

        val_dataset = ImageClassificationDataset(x_val, y_val, transform=Config.val_transform, class_to_id=class_to_id)
        val_loader = torch.utils.data.DataLoader(val_dataset,
                                                 batch_size=Config.batch_size,
                                                 shuffle=False,
                                                 num_workers=Config.n_workers,
                                                 )

        return train_loader, val_loader
