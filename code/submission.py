import os
import tensorflow as tf
import numpy as np
from PIL import Image
import json

# Set the seed to replicate the experiments
SEED = 1234
tf.random.set_seed(SEED)
print("Start Submitting...\n\n\n")
# Setting up all the previews variables
img_h = 1536
img_w = img_h


def meanIoU(y_true, y_pred):
    # get predicted class from softmax
    y_pred = tf.expand_dims(tf.argmax(y_pred, -1), -1)
    per_class_iou = []

    for i in range(1,3): # exclude the background class 0
      # Get prediction and target related to only a single class (i)
      class_pred = tf.cast(tf.where(y_pred == i, 1, 0), tf.float32)
      class_true = tf.cast(tf.where(y_true == i, 1, 0), tf.float32)
      intersection = tf.reduce_sum(class_true * class_pred)
      union = tf.reduce_sum(class_true) + tf.reduce_sum(class_pred) - intersection
    
      iou = (intersection + 1e-7) / (union + 1e-7)
      per_class_iou.append(iou)

    return tf.reduce_mean(per_class_iou)
from keras import backend as K
def dice_coef_multi(y_true, y_pred, smooth=1e-7):
    y_true_f = K.flatten(K.one_hot(K.cast(y_true, 'int32'), num_classes=3)[...,1:])
    y_pred_f = K.flatten(y_pred[...,1:])
    intersect = K.sum(y_true_f * y_pred_f, axis=-1)
    denom = K.sum(y_true_f + y_pred_f, axis=-1)
    return K.mean((2. * intersect / (denom + smooth)))

def dice_coef_loss_multi(y_true, y_pred):
    return 1 - dice_coef_multi(y_true, y_pred)
  
def rle_encode(img):
    '''
    img: numpy array, 1 - foreground, 0 - background
    Returns run length as string formatted
    '''
    pixels = img.flatten()
    pixels = np.concatenate([[0], pixels, [0]])
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
    runs[1::2] -= runs[::2]
    return ' '.join(str(x) for x in runs)

def create_json(img, img_name, team, crop, file_path):
  # Creating the dict
  submission_dict = {}
  submission_dict[img_name] = {}
  submission_dict[img_name]['shape'] = img.shape
  submission_dict[img_name]['team'] = team
  submission_dict[img_name]['crop'] = crop
  submission_dict[img_name]['segmentation'] = {}

  #RLE encoding
  # crop
  rle_encoded_crop = rle_encode(mask_arr == 1)
  # weed
  rle_encoded_weed = rle_encode(mask_arr == 2)

  submission_dict[img_name]['segmentation']['crop'] = rle_encoded_crop
  submission_dict[img_name]['segmentation']['weed'] = rle_encoded_weed

  # Please notice that in this example we have a single prediction.
  # For the competition you have to provide segmentation for each of
  # the test images.

  # Finally, save the results into the submission.json file
  with open(file_path, 'a') as f:
      json.dump(submission_dict, f)
# Load the model 


model = tf.keras.models.load_model("/content/drive/MyDrive/cp_24.ckpt", custom_objects={'meanIoU':meanIoU})#, 'dice_coef_loss_multi': dice_coef_loss_multi})