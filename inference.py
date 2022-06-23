import os
from time import time
from argparse import ArgumentParser
from deep_utils import TorchVisionInference

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("--model_path", required=True, help="path to saved model")
    parser.add_argument("--device", default="cpu", help="cuda or cpu")
    parser.add_argument("--img_path", required=True, help="path to image")
    args = parser.parse_args()
    model = TorchVisionInference(args.model_path, device=args.device)
    if os.path.isdir(args.img_path):
        model.infer_directory(args.img_path)
    else:
        tic = time()
        prediction = model.infer(args.img_path)
        toc = time()
        print(f"predicted class is for {args.img_path} is {prediction}\ninference time: {toc - tic}")
