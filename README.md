# pytorch-lightning-classification-template

This is a template project for training image classification tasks.
To start the training modify the `settings.py` based on your requirements.

# Dataset format:

The format of dataset directory should be as follows:

```
├── dataset
│   ├── class-1
│   │  ├──image-name.jpg
│   │  ├──image-name.jpg
│   │  ├──...
│   ├── class-2
│   │  ├──image-name.jpg
│   │  ├──image-name.jpg
│   │  ├──...
...
```

Note:

1. image-name is arbitrary
2. class-1 and class-2 should be renamed to the real name of the input classes.

# Training:

```commandline
cd pytorch-lightning-image-classification-template
python train.py --model_name squeezenet --dataset_dir <path to datasaet> --output_dir <path to output>  
```

run `python train.py -h` for more configuration.

# Inference

```commandline
python inference.py --model_path <path-to-ckpt-file> --img_path <path-to-img-file-img-directory>
```

# References

https://github.com/pooya-mohammadi/deep_utils