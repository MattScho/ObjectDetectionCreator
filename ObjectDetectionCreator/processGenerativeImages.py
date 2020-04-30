'''
Use a trained model to generate more training data

Change lines:
24 and 40 for your system configurations
'''

import os
import cv2
import numpy as np
import tensorflow as tf
import sys
import glob

# This is needed since the notebook is stored in the object_detection folder.
sys.path.append("..")

# Import utilites
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as vis_util

global boxes, scores, classes, num, oImage, imageName
PROJECT_PATH = "D:/MajorProjects/ObjectDetectionCreator/"

# Name of the directory containing the object detection module we're using
MODEL_PATH = 'inference_graph'
IMAGE_PATH = PROJECT_PATH + "generativeImages"
TRAIN_IMAGES_PATH = '/data/Images/trainImages/'


# Grab path to current working directory
CWD_PATH = os.getcwd()

# Path to frozen detection graph .pb file, which contains the model that is used
# for object detection.
PATH_TO_CKPT = os.path.join(CWD_PATH, MODEL_PATH, 'frozen_inference_graph.pb')

# Path to label map file
PATH_TO_LABELS = 'D:/MajorProjects/EqSolver/data/utils/labelmap.pbtxt'



# Path to image
PATH_TO_IMAGE = os.path.join(CWD_PATH, IMAGE_PATH)

# Number of classes the object detector can identify
NUM_CLASSES = 17

def writeXML(cutOff):
    baseImageName = imageName[:-4]
    cv2.imwrite(TRAIN_IMAGES_PATH + imageName, oImage)
    xmlName = baseImageName + ".xml"
    height, width, channels = image.shape
    out = "<annotation>\n" + \
        "<filename></filename>\n"+\
        "<size>\n"+\
        "<width>" + str(width) + "</width>\n"+\
        "<height>" + str(height) + "</height>\n"+\
        "<depth>3</depth>\n"+\
        "</size>\n"+\
        "<segmented>0</segmented>\n"
    tempBoxes = range(len(boxes[0][:cutOff]))
    for ind in tempBoxes:
        box = boxes[0][ind]
        class_ = int(classes[0][ind])

        ymin, xmin, ymax, xmax = box
        ymin = str(int(ymin*height))
        ymax = str(int(ymax*height))
        xmin = str(int(xmin*height))
        xmax = str(int(xmax*height))
        out += "<object>\n"+\
        "<name>" + label_map_dict[class_] + "</name>\n"+\
        "<pose>Unspecified</pose>\n"+\
        "<truncated>0</truncated>\n"+\
        "<difficult>0</difficult>\n"+\
        "<bndbox>\n"+\
        "<xmin>" + xmin + "</xmin>\n"+\
        "<ymin>"+ ymin + "</ymin>\n"+\
        "<xmax>" + xmax + "</xmax>\n"+\
        "<ymax>" + ymax + "</ymax>\n"+\
        "</bndbox>" +\
        "</object>\n"

    out += "</annotation>"
    with open(xmlName, "w+") as file:
        file.write(out)
        file.close()


def clickProcess(event, x,y,flags,param):
    if event == cv2.EVENT_LBUTTONDOWN:
        print("Saving Image")
        cv2.destroyWindow("image")
        cutOff = np.argmax(scores<.75)
        if cutOff > 0:
            writeXML(cutOff)
    elif event == cv2.EVENT_RBUTTONUP:
        print("Next Image")
        cv2.destroyWindow("image")

def decideImage(image):
    cv2.namedWindow("image")
    cv2.setMouseCallback("image", clickProcess)
    cv2.imshow("image", image)
    cv2.waitKey(0)


# Load the label map.
# Label maps map indices to category names, so that when our convolution
# network predicts `5`, we know that this corresponds to `king`.
# Here we use internal utility functions, but anything that returns a
# dictionary mapping integers to appropriate string labels would be fine
label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
label_map_dict_ = label_map_util.get_label_map_dict(PATH_TO_LABELS)
label_map_dict = {}
for k in label_map_dict_.keys():
    label_map_dict[label_map_dict_[k]] = k

categories = label_map_util.convert_label_map_to_categories(
    label_map, max_num_classes=NUM_CLASSES, use_display_name=True)
category_index = label_map_util.create_category_index(categories)

# Load the Tensorflow model into memory.
detection_graph = tf.Graph()
with detection_graph.as_default():
    od_graph_def = tf.GraphDef()
    with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
        serialized_graph = fid.read()
        od_graph_def.ParseFromString(serialized_graph)
        tf.import_graph_def(od_graph_def, name='')

    sess = tf.Session(graph=detection_graph)

# Define input and output tensors (i.e. data) for the object detection classifier

# Input tensor is the image
image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')

# Output tensors are the detection boxes, scores, and classes
# Each box represents a part of the image where a particular object was detected
detection_boxes = detection_graph.get_tensor_by_name('detection_boxes:0')

# Each score represents level of confidence for each of the objects.
# The score is shown on the result image, together with the class label.
detection_scores = detection_graph.get_tensor_by_name('detection_scores:0')
detection_classes = detection_graph.get_tensor_by_name('detection_classes:0')

# Number of objects detected
num_detections = detection_graph.get_tensor_by_name('num_detections:0')

# Load image using OpenCV and
# expand image dimensions to have shape: [1, None, None, 3]
# i.e. a single-column array, where each item in the column has the pixel RGB value
for image in glob.glob(IMAGE_PATH + "/*.jpg"):
    imageName = image
    image = cv2.imread(image)
    image = cv2.resize(image, (1024, 1024), interpolation=cv2.INTER_AREA)
    cv2.imwrite(imageName, image)
    oImage = image
    # Perform the actual detection by running the model with the image as input
    (boxes, scores, classes, num) = sess.run(
        [detection_boxes, detection_scores, detection_classes, num_detections],
        feed_dict={image_tensor: np.array([image])})#image_expanded
    # Draw the results of the detection (aka 'visualize the results')

    vis_util.visualize_boxes_and_labels_on_image_array(
        image,
        np.squeeze(boxes),
        np.squeeze(classes).astype(np.int32),
        np.squeeze(scores),
        category_index,
        use_normalized_coordinates=True,
        line_thickness=4,
        min_score_thresh=0.70)


    # All the results have been drawn on the image. Now display the image.
    decideImage(image)
