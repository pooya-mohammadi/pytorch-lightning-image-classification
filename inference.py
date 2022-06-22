from time import time
from argparse import ArgumentParser
from pathlib import Path
from typing import Union
import numpy as np
import cv2
import torch
from settings import Config
from PIL import Image, ImageDraw
from train import LitModel
from deep_utils import load_pickle, show_destroy_cv2


class ModelPred:
    def __init__(self, model_path, labels_map_path, device="cpu"):
        self.device = device
        self.model = LitModel.load_from_checkpoint(model_path).to(device).eval()
        # escape cold start
        self.model(torch.randn((1, 3, 224, 224)).to(device))
        self.label_map = load_pickle(labels_map_path)
        self.transform = Config.val_transform

    def detect(self, img: Union[str, Path, np.ndarray]):
        if type(img) is not np.ndarray:
            img = cv2.imread(img)[..., ::-1]
        image = self.transform(image=img)["image"]
        image = image.view(1, *image.size()).to(self.device)
        with torch.no_grad():
            logits = self.model(image).cpu().squeeze(0).numpy()
        prediction = np.argmax(logits)
        return self.label_map[prediction]


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("--model_path", default="output/exp_1/best.ckpt")
    parser.add_argument("--device", default="cpu", help="cuda or cpu")
    parser.add_argument("--labels_map_path", default="output/exp_1/labels_map.pkl", help="path to class label")
    parser.add_argument("--img_path", default="sample_images/samand_01.jpg")
    args = parser.parse_args()
    model = ModelPred(args.model_path,
                      labels_map_path=args.labels_map_path,
                      device=args.device)
    img = Image.open(args.img_path)
    tic = time()
    prediction = model.detect(args.img_path)
    toc = time()
    draw = ImageDraw.Draw(img)
    draw.text((20, 20), prediction, (0, 255, 0))
    print(f"Vehicle Color is: {prediction}\n, inference time: {toc - tic}")
    show_destroy_cv2(np.array(img)[..., ::-1])
