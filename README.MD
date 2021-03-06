# Convolutional Neural Net of CIFAR 10 Dataset
This is a convolutional neural net designed to be trained with the CIFAR 10 dataset

A CNN trained with the CIFAR 10 dataset that I was able to achieve around 70% accuracy with after a couple epochs, currently has a couple
random test images found off DDG that you can use to test the CNN if you wish to test it. 

No weights saved but you can simply train and save some yourself.

## IMPORTANT
The code here is highly experimental and with how it's currently laid out, its not really suitable for actual projects. But feel free to use what I wrote as a learning experience for keras and if you want for your own personal projects. 

I mostly kept the code in one main py file just to keep it a lot easier to read and follow for beginners.

## Usage example

The five main functions you can simply comment or uncomment to use the CNN:

fit_model - trains the model with the CIFAR 10 dataset

save_model - serializes both model and stored weights currently being used

load_model - loads both model and stored weights from modelweights.json/h5 file

evaluate_model - evaluates the model

model_predict - pass in any image URI and will get the model to predict if any of the CIFAR 10 categories are within the image.

## Requirements

Python 2/3 (with Tensorflow/Kera's backend)

## Author
kenevil1
