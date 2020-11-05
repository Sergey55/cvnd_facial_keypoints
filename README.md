[//]: # (Image References)

[image1]: ./images/key_pts_example.png "Facial Keypoint Detection"
[image2]: ./images/landmarks_numbered.jpg "Landmarks numbered"
[image3]: ./images/average_face_keypoints.png "Result image"

# Facial Keypoint Detection

## Project Overview

In this project, I combined computer vision techniques and deep learning architectures to build a facial keypoint detection system. Facial keypoints include points around the eyes, nose, and mouth on a face and are used in many applications.

![Facial Keypoint Detection][image1]

Facial keypoints (also called facial landmarks) are the small magenta dots shown on each of the faces in the image above. In each training and test image, there is a single face and 68 keypoints, with coordinates (x, y), for that face. These keypoints mark important areas of the face: the eyes, corners of the mouth, the nose, etc. These keypoints are relevant for a variety of tasks, such as face filters, emotion recognition, pose recognition, and so on. Here they are, numbered, and you can see that specific ranges of points match different portions of the face.

![Landmarks numbered][image2]

## Environment

The project assumes using the Conda package management system. Each project uses its own environment with installed packages of a particular version. Environment can be created from YAML file and activated using next commands:

```
conda env create -f environment.yml
conda activate facial_keypoints
```

or based on exact package versions:

```
conda create --name NEW_ENV_NAME -- file pkgs.txt
conda activate NEW_ENV_NAME
```

afterward your prompt will look similar to the following:

```
(facial_keypoints) $
```

## Data

All of the data needed to train a neural network is in the repository, in subdirectory `data`. Thid drt of image data has been extracted deom the [YouTube Faces Dataset](https://www.cs.tau.ac.il/~wolf/ytfaces/), which includes videos of people in YouTube videos. These videos have been fed through some processing steps and turned into sets of image frames containing one face and the associated keypoints.

## Architecture

// TODO:

## Training

The model trained using `AI Platform Notebook` instance in Google Cloud Platform. This project requires `PyTorch Lightning` package to be pre-installed. Since the model implemented using PyTorch Lightning it can be trained using both GPU and TPU. Training process can be started using `train.sh` file. It automatically stops an instance at the end of the training.

## Evaluation

Repository contains weights for trained network. Since GitHub restricts max file size by 100 Mb model state file was compressed and split up into multiple files. Next command can be executed in the project's home dir to decompress model weights:

```
cat ./saved_models/lightning_logs.tar.* | tar -xzvf -
```

After that new `lightning_logs` subdirectory should appear. The model can be instantiated next way:

```
model = Net.load_from_checkpoint('./lightning_logs/version_13/checkpoints/epoch=199.ckpt')
```

Having model instantiated we can obtain keypoints by photo using the following code snippet:

```
model.eval()

transformations = Compose([
    Rescale(250),
    RandomCrop(224),
    Normalize(),
    ToTensor()
])

image = Image.open('./images/michelle_detected.png')
tensor = transformations(image).unsqueeze(0)

result_pts = model.sample(tensor)
```

Further details can be found in [Evaluate.ipynb](./Evaluate.ipynb)

## Files

Repository contains next files and folders:

* [data](./data) - Images sed for model training and testing.
* [images](./images) - Images used in this readme.
* [savem_models](./saved_models) - Archived PyTorch Lightning logs and model state.
* [datamodule.py](./datamodule.py) - Pytorch Lightning datamodule which can be used for obtaining dataloader for train/test dataset.
* [dataset.py](./dataset.py) - Custom dataset implementation.
* [model.py](./model.py) - Model implementation.
* [pkgs.txt](./pkgs.txt) - List of required packages.
* [README.md](./README.md) - This file.
* [train.py](./train.py) - Script for training network.
* [train.sh](./train.sh) - Shell script for starting trainig process from terminal.
* [transforms.py](./transorms.py) - Custom transformations applied to dataset items.

## Results

Finally, I got a model that performs good enough and can be applied to images from the internet without major changes.

![Result image][image3]