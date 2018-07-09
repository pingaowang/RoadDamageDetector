import numpy as np
import os
import argparse
import cv2
import sys
import tarfile
import tensorflow as tf
import zipfile

from collections import defaultdict
from io import StringIO
from matplotlib import pyplot as plt
from PIL import Image

from utils import label_map_util
from utils import visualization_utils as vis_util

parser = argparse.ArgumentParser()
parser.add_argument("test_name", help="Name of the test folder.")
args = parser.parse_args()

TEST_NAME = args.test_name
TEST_PATH = os.path.join("./test_data/", TEST_NAME)
SAVE_PATH = "./result/result_" + TEST_NAME

if not os.path.exists(SAVE_PATH):
    os.makedirs(SAVE_PATH)



# Path to frozen detection graph. This is the actual model that is used for the object detection.
PATH_TO_CKPT =  'trainedModels/ssd_mobilenet_RoadDamageDetector.pb'

# List of the strings that is used to add correct label for each box.
PATH_TO_LABELS = 'trainedModels/crack_label_map.pbtxt'

NUM_CLASSES = 8

detection_graph = tf.Graph()
with detection_graph.as_default():
  od_graph_def = tf.GraphDef()
  with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
    serialized_graph = fid.read()
    od_graph_def.ParseFromString(serialized_graph)
    tf.import_graph_def(od_graph_def, name='')

label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES, use_display_name=True)
category_index = label_map_util.create_category_index(categories)

def img_resize(image):
    return cv2.resize(image, (600, 600))

def load_image_into_numpy_array(image):
  (im_width, im_height) = image.size
  return np.array(image.getdata()).reshape(
      (im_height, im_width, 3)).astype(np.uint8)



##-- Pingao --##

def travel_folder_get_img_path(folder_path):
    ## Given folder path, return a list contains all images' path.
    list_img_path = []
    for subdir, dirs, files in os.walk(folder_path):
        for file in files:
            img_path = os.path.join(subdir, file)
            list_img_path.append(img_path)

    list_img_path = filter(lambda k: '.DS_Store' not in k,list_img_path)
    return [x for x in list_img_path if x is not None]

TEST_IMAGE_PATHS = travel_folder_get_img_path(TEST_PATH)

IMAGE_SIZE = (12, 8)

with detection_graph.as_default():
  with tf.Session(graph=detection_graph) as sess:
    # Definite input and output Tensors for detection_graph
    image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
    # Each box represents a part of the image where a particular object was detected.
    detection_boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
    # Each score represent how level of confidence for each of the objects.
    # Score is shown on the result image, together with the class label.
    detection_scores = detection_graph.get_tensor_by_name('detection_scores:0')
    detection_classes = detection_graph.get_tensor_by_name('detection_classes:0')
    num_detections = detection_graph.get_tensor_by_name('num_detections:0')

    for image_path in TEST_IMAGE_PATHS:
      image = Image.open(image_path)
      # the array based representation of the image will be used later in order to prepare the
      # result image with boxes and labels on it.
      image_np = load_image_into_numpy_array(image)
      image_np = img_resize(image_np)
      # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
      image_np_expanded = np.expand_dims(image_np, axis=0)
      # Actual detection.
      (boxes, scores, classes, num) = sess.run(
          [detection_boxes, detection_scores, detection_classes, num_detections],
          feed_dict={image_tensor: image_np_expanded})
      # Visualization of the results of a detection.
      vis_util.visualize_boxes_and_labels_on_image_array(
          image_np,
          np.squeeze(boxes),
          np.squeeze(classes).astype(np.int32),
          np.squeeze(scores),
          category_index,
          min_score_thresh=0.3,
          use_normalized_coordinates=True,
          line_thickness=8)
      # plt.figure(figsize=IMAGE_SIZE)
      cv2.imwrite(os.path.join(SAVE_PATH, image_path.split('/')[-1]), image_np[:,:,(2,1,0)])
