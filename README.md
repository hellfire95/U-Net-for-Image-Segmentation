#  U-Net for Image Segmentation

## Overview
This notebook illustrates how to build a [U-Net](https://arxiv.org/abs/1505.04597) for semantic image segmentation. U-Net is a fully convolutional network that uses skip connections from the encoder to the decoder. This allows the model to capture both fine-grained details and high-level context. At the end of this lab, you will be able to use the U-Net to output segmentation masks that show which pixels of an input image belong to the background, foreground, and outline.

![Sample Output](https://github.com/hellfire95/U-Net-for-Image-Segmentation/blob/main/u-net_prediction.png?raw=true)

## Dataset
You will be training the model on the Oxford Pets - IIT dataset, which contains pet images and their associated segmentation masks. For this lab, you will only use the images and segmentation masks. The dataset is included in TensorFlow Datasets and can be easily downloaded.

## Dataset Preparation
The dataset is prepared by performing the following steps:
- Simple data augmentation by flipping the images.
- Normalizing the pixel values.
- Resizing the images.
- Adjusting the pixel values in the segmentation masks to map them to the classes {'pet', 'background', 'outline'}.

## Model Architecture
The U-Net model consists of an encoder, a bottleneck, and a decoder. Skip connections are used to concatenate encoder block outputs to the corresponding decoder stages. Here's an overview of the architecture:

![U-Net Architecture](https://github.com/hellfire95/U-Net-for-Image-Segmentation/blob/main/U-net-architecture.png?raw=true)

- **Encoder**: The encoder contains repeated blocks with Conv2D layers, ReLU activations, MaxPooling, and Dropout.
- **Bottleneck**: The bottleneck extracts more features without a pooling layer.
- **Decoder**: The decoder upsamples the features back to the original image size and uses skip connections.

## Model Output
The U-Net model outputs three channels, corresponding to the labels {'pet', 'background', 'outline'}. It predicts the class for each pixel, making it a multi-class segmentation task.

## Training
The model is trained using the `sparse_categorical_crossentropy` loss, which is suitable for multi-class prediction. It learns to assign each pixel a label based on the true segmentation mask.
![training curve](https://github.com/hellfire95/U-Net-for-Image-Segmentation/blob/main/training_curve-unet.png?raw=true)

## Making Predictions
The trained model can be used to make predictions on new images from the test dataset.

## Class-wise Metrics
You can compute class-wise metrics, including Intersection over Union (IOU) and Dice Score, to assess the model's performance for each class in the segmentation task.
