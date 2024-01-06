# MLOps Project Description

## Group 75
Mike Auer (s232050) \
Malgorzata Korona (s223322) \
Samy Haffoudhi (s222887) \
Mateusz Baldachowski (s230241)

## Overall goal of the project
The goal of the project is to use computer vision to classify dog breeds from a set of dog images.

## What framework are you going to use and do you intend to include the framework into your project?
For this computer vision problem, we plan to use the [Pytorch Image Models](https://github.com/huggingface/pytorch-image-models) package. We aim to use scripts and models provided by this widely used package. As a starting point, we will evaluate some of the  pre-trained models' performance on the task and then decide on an architecture and how to further improve these results. 

## What data are you going to run on? (initially, may change)
We are using the Kaggle dataset [Dog Breed Identification](https://www.kaggle.com/competitions/dog-breed-identification/data). It contains above 20000 dog images with 120 dog breeds split into train and test datasets in half. Since the test set does not contain target labels we are going to use only the train dataset which means that we will have 10222 dog images at our disposal. Dataset is clean and requires no further cleaning. We chose it because it is straightforward, requires minimal effort in terms of data processing and looks feasible to work on given limited time.

## What models do you expect to use?
We plan to use a pre-trained ResNet model to start with (https://arxiv.org/abs/1512.03385), probably ResNet 18, included in the Pytorch Image Models framework. We are also going to spend some time training it additionally (fine-tuning) on our dog breeds dataset. We might also experiment with different ResNet architectures, such as ResNet 34 or ResNet 50, or [NASNet](https://pprp.github.io/timm/models/nasnet/), nevertheless we want to mostly focus on one of them due to time constraints.
