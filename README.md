# Segmentation Model for Medical Image Segmentation

This repository contains a deep learning model for medical image segmentation using U-Net. The model is trained on a dataset of medical images and corresponding masks.

<a href="https://colab.research.google.com/drive/1wv1kaT5SAU6j1UL3v6jlGeuSwmR-Tij0?usp=sharing"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open Notebook In Colab"></a>

## Dataset

The dataset used for training the model consists of **595 RGB images** of kidneys. These images are paired with corresponding **binary masks** for segmentation tasks. The dataset is split into **80% for training** and **20% for testing**.

- **Number of Images**: 595 RGB images
- **Image Dimensions**: 128x128 pixels (resized for model input)
- **Mask Dimensions**: 128x128 pixels (binary masks)
- **Split**: 80% training, 20% testing

## Model

The model used in this repository is a **U-Net**, a popular architecture for medical image segmentation. The U-Net is composed of an **encoder-decoder** structure with skip connections, which allows it to learn spatial features while maintaining context throughout the segmentation process.

### Key Details of the Model:
- **Encoder**: Uses a series of convolutional layers to extract features from input images.
- **Decoder**: Upsamples the feature maps and uses skip connections to refine the segmentation output.
- **Loss Function**: **Binary Cross-Entropy with Logits Loss** is used for training the model.
- **Optimizer**: **Adam optimizer** is used to minimize the loss.
- **Metrics**: The model is evaluated using **Dice coefficient** and **Intersection over Union (IoU)** to assess the segmentation performance.
- **Training**: The model is trained for **100 epochs** with a batch size of **128**. Training uses a **learning rate of 0.0001**.
- **Output**: The model predicts binary masks for kidney segmentation.

## Sample Input, Output and Mask

### Input and Mask Image:
![Input and Mask Image](images/sample_input_mask.png)

### Input, Output and Mask:
![Input, Output and Mask Image](images/sample_outputs.png)

## Loss Diagrams

### Training and Validation Loss:
![Loss Plot](images/loss_diagram.png)

### Training and Validation IoU:
![IoU Plot](images/IoU_diagram.png)

### Training and Validation Dice:
![Dice Plot](images/Dice_diagram.png)

## Requirements

You can install the necessary dependencies by running:

```bash
pip install -r requirements.txt
