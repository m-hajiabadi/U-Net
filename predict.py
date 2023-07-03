# USAGE
# python predict.py
# import the necessary packages
from utils import config
import matplotlib.pyplot as plt
import numpy as np
import torch
import cv2
import os

def prepare_plot(origImage, origMask, predMask, imagePath, index):
    plot_image(index, 1, origImage, "Image")
    plot_image(index, 2, origMask, "Original Mask")
    # write the image path as text below the plot
    plt.text(0.5, -0.5, imagePath, ha='center', va='center', transform=plt.gca().transAxes)
    plot_image(index, 3, predMask, "Predicted Mask")
    



def plot_image(index, arg1, arg2, arg3):
    # plot the original image, its mask, and the predicted mask
    plt.subplot(len(imagePaths), 3, index*3 + arg1)
    plt.imshow(arg2)
    plt.title(arg3)
 
def make_predictions(model, imagePath, index):
	# set model to evaluation mode
	model.eval()
	# turn off gradient tracking
	with torch.no_grad():
		# load the image from disk, swap its color channels, cast it
		# to float data type, and scale its pixel values
		image = cv2.imread(imagePath)
		image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
		image = image.astype("float32") / 255.0
		# resize the image and make a copy of it for visualization
		image = cv2.resize(image, (128, 128))
		orig = image.copy()
		# find the filename and generate the path to ground truth
		# mask
		filename = imagePath.split(os.path.sep)[-1]
		groundTruthPath = os.path.join(config.MASK_DATASET_PATH,
			filename)
		# load the ground-truth segmentation mask in grayscale mode
		# and resize it
		gtMask = cv2.imread(groundTruthPath, 0)
		gtMask = cv2.resize(gtMask, (config.INPUT_IMAGE_HEIGHT,
			config.INPUT_IMAGE_HEIGHT))
        # make the channel axis to be the leading one, add a batch
		# dimension, create a PyTorch tensor, and flash it to the
		# current device
		image = np.transpose(image, (2, 0, 1))
		image = np.expand_dims(image, 0)
		image = torch.from_numpy(image).to(config.DEVICE)
		# make the prediction, pass the results through the sigmoid
		# function, and convert the result to a NumPy array
		predMask = model(image).squeeze()
		predMask = torch.sigmoid(predMask)
		predMask = predMask.cpu().numpy()
		# filter out the weak predictions and convert them to integers
		predMask = (predMask > config.THRESHOLD) * 255
		predMask = predMask.astype(np.uint8)
		# prepare a plot for visualization
		prepare_plot(orig, gtMask, predMask, imagePath, index)
  
# load the image paths in our testing file and randomly select 10
# image paths
print("[INFO] loading up test image paths...")
imagePaths = open(config.TEST_PATHS).read().strip().split("\n")
imagePaths = np.random.choice(imagePaths, size=3)
# load our model from disk and flash it to the current device
print("[INFO] load up model...")
unet = torch.load(config.MODEL_PATH).to(config.DEVICE)

figure = plt.figure(figsize=(10, 10))

# iterate over the randomly selected test image paths
for index, path in enumerate(imagePaths):
    # make predictions and visualize the results
    make_predictions(unet, path, index)

# set the layout of the figure and display it
plt.tight_layout()
plt.savefig(config.PLOT_PATH_PREDICTION)