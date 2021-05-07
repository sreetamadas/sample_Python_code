# -*- coding: utf-8 -*-
"""
Created on Fri Nov  6 15:43:10 2020
@author: DAR9KOR

Infer on test images - code to be kept inside training_demo folder

"""
import time
import io
import os
import glob
import scipy.misc
import numpy as np
from six import BytesIO
from PIL import Image, ImageDraw, ImageFont

import tensorflow as tf

from object_detection.utils import label_map_util
from object_detection.utils import config_util
from object_detection.utils import visualization_utils as viz_utils
from object_detection.builders import model_builder

import matplotlib
import matplotlib.pyplot as plt
#%matplotlib inline


#############################################################################
def load_image_into_numpy_array(path):
  """Load an image from file into a numpy array.

  Puts image into numpy array to feed into tensorflow graph.
  Note that by convention we put it into a numpy array with shape
  (height, width, channels), where channels=3 for RGB.

  Args:
    path: the file path to the image

  Returns:
    uint8 numpy array with shape (img_height, img_width, 3)
  """
  img_data = tf.io.gfile.GFile(path, 'rb').read()
  image = Image.open(BytesIO(img_data))
  (im_width, im_height) = image.size
  return np.array(image.getdata()).reshape(
      (im_height, im_width, 3)).astype(np.uint8)

##############################################################################

#os.chdir("/content/drive/My Drive/VQI/efficientDet/tensorflow/workspace/training_demo/")

########################################################################
###################     METHOD 1       ##################################
# https://colab.research.google.com/github/TannerGilbert/Tensorflow-Object-Detection-API-Train-Model/blob/master/Tensorflow_2_Object_Detection_Train_model.ipynb#scrollTo=EEX-m3P1yp4y


######  load model (uses a .pb file) & labelmap   ############
## LOAD MODEL
tf.keras.backend.clear_session()
model = tf.saved_model.load(f'./exported-models/efficientdet_d1/saved_model')

## label map
labelmap_path = './annotations/label_map.pbtxt'  #'/content/labelmap.pbtxt'
category_index = label_map_util.create_category_index_from_labelmap(labelmap_path, use_display_name=True)



########   Inferencing   ###########

def run_inference_for_single_image(model, image):
  image = np.asarray(image)
  # The input needs to be a tensor, convert it using `tf.convert_to_tensor`.
  input_tensor = tf.convert_to_tensor(image)
  # The model expects a batch of images, so add an axis with `tf.newaxis`.
  input_tensor = input_tensor[tf.newaxis,...]

  # Run inference
  model_fn = model.signatures['serving_default']
  output_dict = model_fn(input_tensor)

  # All outputs are batches tensors.
  # Convert to numpy arrays, and take index [0] to remove the batch dimension.
  # We're only interested in the first num_detections.
  num_detections = int(output_dict.pop('num_detections'))
  output_dict = {key:value[0, :num_detections].numpy() 
                 for key,value in output_dict.items()}
  output_dict['num_detections'] = num_detections

  # detection_classes should be ints.
  output_dict['detection_classes'] = output_dict['detection_classes'].astype(np.int64)
   
  # Handle models with masks:
  if 'detection_masks' in output_dict:
    # Reframe the the bbox mask to the image size.
    detection_masks_reframed = utils_ops.reframe_box_masks_to_image_masks(
              output_dict['detection_masks'], output_dict['detection_boxes'],
               image.shape[0], image.shape[1])      
    detection_masks_reframed = tf.cast(detection_masks_reframed > 0.5,
                                       tf.uint8)
    output_dict['detection_masks_reframed'] = detection_masks_reframed.numpy()
    
  return output_dict



for image_path in glob.glob('./images/val/*.jpg'):  # using val set, as test set was used as val during training
  #print("")
  #print(image_path)
  start_time=time.time()
  image_np = load_image_into_numpy_array(image_path)
  output_dict = run_inference_for_single_image(model, image_np)
  #print(" Inference time: %s seconds " % (time.time() - start_time))
  print("")
  print('img: ', image_path, ", Inference time: %s seconds " % (time.time() - start_time))
  ## display images with pred boxes
  #viz_utils.visualize_boxes_and_labels_on_image_array(
  #    image_np,
  #    output_dict['detection_boxes'],
  #    output_dict['detection_classes'],
  #    output_dict['detection_scores'],
  #    category_index,
  #    instance_masks=output_dict.get('detection_masks_reframed', None),
  #    use_normalized_coordinates=True,
  #    line_thickness=8)
  #display(Image.fromarray(image_np)) 



##############################################################################
##################   METHOD 2   ################################################
# Roboflow, https://colab.research.google.com/drive/1sLqFKVV94wm-lglFq_0kGo2ciM0kecWD#scrollTo=YnSEZIzl4M10&uniqifier=1

'''
#######  load model (uses checkpoint file) & labelmap  ###########
# LOAD MODEL : recover our saved model
pipeline_config = "./exported-models/efficientdet_d1-round2/pipeline.config"  #pipeline_file


#generally you want to put the last ckpt from training in here (LAST OR BEST?)
model_dir = './models/efficientdet_d1-round2/ckpt-23'
configs = config_util.get_configs_from_pipeline_file(pipeline_config)
model_config = configs['model']
detection_model = model_builder.build(
      model_config=model_config, is_training=False)


# Restore checkpoint
ckpt = tf.compat.v2.train.Checkpoint(
      model=detection_model)
ckpt.restore(os.path.join('./models/efficientdet_d1-round2/ckpt-23'))   ## should this checkpoint be same as above?



def get_model_detection_function(model):
  """Get a tf.function for detection."""

  @tf.function
  def detect_fn(image):
    """Detect objects in image."""

    image, shapes = model.preprocess(image)
    prediction_dict = model.predict(image, shapes)
    detections = model.postprocess(prediction_dict, shapes)

    return detections, prediction_dict, tf.reshape(shapes, [-1])

  return detect_fn

detect_fn = get_model_detection_function(detection_model)


##  map labels for inference decoding  
label_map_path = configs['eval_input_config'].label_map_path
label_map = label_map_util.load_labelmap(label_map_path)
categories = label_map_util.convert_label_map_to_categories(
    label_map,
    max_num_classes=label_map_util.get_max_label_map_index(label_map),
    use_display_name=True)
category_index = label_map_util.create_category_index(categories)
label_map_dict = label_map_util.get_label_map_dict(label_map, use_display_name=True)



########   Inferencing   ###########
import glob
import time

for image_path in glob.glob('./images/sample_val_infer/*.jpg'):
  #print("")
  #print(image_path)
  start_time=time.time()
  image_np = load_image_into_numpy_array(image_path)
  #output_dict = run_inference_for_single_image(model, image_np)
  input_tensor = tf.convert_to_tensor(np.expand_dims(image_np, 0), dtype=tf.float32)
  detections, predictions_dict, shapes = detect_fn(input_tensor)
  
  label_id_offset = 1
  #print(" Inference time: %s seconds " % (time.time() - start_time))
  print("")
  print('img: ', image_path, ", Inference time: %s seconds " % (time.time() - start_time))
  #image_np_with_detections = image_np.copy()  
  #viz_utils.visualize_boxes_and_labels_on_image_array(
  #    image_np_with_detections,
  #    detections['detection_boxes'][0].numpy(),
  #    (detections['detection_classes'][0].numpy() + label_id_offset).astype(int),
  #    detections['detection_scores'][0].numpy(),
  #    category_index,
  #    use_normalized_coordinates=True,
  #    max_boxes_to_draw=200,
  #    min_score_thresh=.5,
  #    agnostic_mode=False,
  #    )
  #plt.figure(figsize=(3,3))  #12,16
  #plt.imshow(image_np_with_detections)
  #plt.show()  
'''  
  


'''
###################################################
###   run detector on a single test image  ###
#it takes a little longer on the first run and then runs at normal speed. 
import random
TEST_IMAGE_PATHS = glob.glob('./images/sample_val_infer/*.jpg')
image_path = random.choice(TEST_IMAGE_PATHS)
print(image_path)
image_np = load_image_into_numpy_array(image_path)

# Things to try:
# Flip horizontally
# image_np = np.fliplr(image_np).copy()

# Convert image to grayscale
# image_np = np.tile(
#     np.mean(image_np, 2, keepdims=True), (1, 1, 3)).astype(np.uint8)

input_tensor = tf.convert_to_tensor(
    np.expand_dims(image_np, 0), dtype=tf.float32)
detections, predictions_dict, shapes = detect_fn(input_tensor)

label_id_offset = 1
image_np_with_detections = image_np.copy()

viz_utils.visualize_boxes_and_labels_on_image_array(
      image_np_with_detections,
      detections['detection_boxes'][0].numpy(),
      (detections['detection_classes'][0].numpy() + label_id_offset).astype(int),
      detections['detection_scores'][0].numpy(),
      category_index,
      use_normalized_coordinates=True,
      max_boxes_to_draw=200,
      min_score_thresh=.5,
      agnostic_mode=False,
)

plt.figure(figsize=(12,16))
plt.imshow(image_np_with_detections)
plt.show()
'''



