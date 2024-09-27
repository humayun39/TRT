import cv2
import os
from keras.models import load_model
import matplotlib.pyplot as plt

from codebase.models.segnet import resnet50_segnet
# Ensure other model imports are correct
# from codebase.models.pspnet import *
# from codebase.models.unet import *
# from codebase.models.fcn import *

import tensorflow as tf
# Use the updated TensorFlow 2.x method for GPU memory configuration
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)

# Load the SegNet model
model = resnet50_segnet(n_classes=7, input_height=576, input_width=768)

# Replace the custom train call with model.fit
train_images_dir = "trainingDataset/train_images/"
train_annotations_dir = "trainingDataset/train_annotations/"
val_images_dir = "trainingDataset/val_images/"
val_annotations_dir = "trainingDataset/val_annotations/"

# Ensure that your data generator and format align with model.fit requirements
# Assuming you have a data generator that yields image and annotation batches
# Here's a simple example using ImageDataGenerator or custom generators

# Train the model
model.fit(
    train_images_dir, 
    train_annotations_dir,
    validation_data=(val_images_dir, val_annotations_dir),
    epochs=15
)

# Print model summary
model.summary()

# Perform segmentation on test images
folder = "testingDataset/test_images/"
for filename in os.listdir(folder):
    out = model.predict_segmentation(
        inp=os.path.join(folder, filename),
        out_fname=os.path.join("testingDataset/segmentation_results/", filename)
    )

# Evaluate model segmentation
print(model.evaluate_segmentation(
    inp_images_dir="testingDataset/test_images/",
    annotations_dir="testingDataset/test_annotations/"
))
