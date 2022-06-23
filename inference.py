import os
from time import time
from argparse import ArgumentParser
from pathlib import Path
from typing import Union
import numpy as np
import cv2
import torch
from model import TorchVisionModel


class Inference:
    def __init__(self, model_path, device="cpu"):
        save_params = torch.load(model_path, map_location=device)
        config = save_params['config']
        self.model = TorchVisionModel(model_name=config.model_name, num_classes=config.n_classes,
                                      last_layer_nodes=config.last_layer_nodes, use_pretrained=False,
                                      feature_extract=True)
        try:
            self.model.load_state_dict(save_params['state_dict'])
        except:
            self.model.load_state_dict({".".join(k.split(".")[1:]): v for k, v in save_params['state_dict'].items()})

        self.device = device
        self.label_map = {v: k for k, v in save_params['class_to_id'].items()}
        self.model.eval()
        self.transform = config.val_transform

    def infer(self, img: Union[str, Path, np.ndarray]):
        if isinstance(img, Path) or isinstance(img, str):
            img = cv2.imread(img)[..., ::-1]
        image = self.transform(image=img)["image"]
        image = image.view(1, *image.size())
        with torch.no_grad():
            image = image.to(self.device)
            logits = self.model(image).squeeze(0).numpy()
        prediction = np.argmax(logits)
        return self.label_map[prediction]


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("--model_path", required=True, help="path to saved model")
    parser.add_argument("--device", default="cpu", help="cuda or cpu")
    parser.add_argument("--img_path", required=True, help="path to image")
    args = parser.parse_args()
    model = Inference(args.model_path, device=args.device)
    if os.path.isdir(args.img_path):
        for image_name in sorted(os.listdir(args.img_path)):
            try:
                tic = time()
                image_path = os.path.join(args.img_path, image_name)
                prediction = model.infer(image_path)
                toc = time()
                print(f"predicted class for {image_path} is {prediction}\ninference time: {toc - tic}")
            except:
                continue
    else:
        tic = time()
        prediction = model.infer(args.img_path)
        toc = time()
        print(f"predicted class is for {args.img_path} is {prediction}\ninference time: {toc - tic}")
