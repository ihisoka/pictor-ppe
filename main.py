import cv2
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib as mpl

from tensorflow.keras.layers import Input

from src.yolo3.model import *
from src.yolo3.detect import *

from src.utils.image import *
from src.utils.datagen import *
from src.utils.fixes import *

# we create the video capture object cap
cap = cv2.VideoCapture("rtsp://admin:hosinox2023@192.168.1.102:554/cam/realmonitor?channel=1&subtype=0 ! rtph264depay ! h264parse ! omxh264dec ! appsink max-buffers=1 drop=true",cv2.CAP_FFMPEG)

def prepare_model():
    '''
    Prepare the YOLO model
    '''
    global input_shape, class_names, anchor_boxes, num_classes, num_anchors, model

    # shape (height, width) of the imput image
    input_shape = (416, 416*2)
    
#     class_names = ['H', 'V', 'W']
    class_names  = ['Worker','Worker Hat','Worker Vest','Worker Hat Vest']
#     class_names = ['W']

    # anchor boxes
    anchor_boxes = np.array(
            [
#             np.array([[ 76,  59], [ 84, 136], [188, 225]]) /32, # output-1 anchor boxes
#             np.array([[ 25,  15], [ 46,  29], [ 27,  56]]) /16, # output-2 anchor boxes
#             np.array([[ 5,    3], [ 10,   8], [ 12,  26]]) /8   # output-3 anchor boxes
                
            np.array([[ 73, 158], [128, 209], [224, 246]]) /32, # output-1 anchor boxes
            np.array([[ 32,  50], [ 40, 104], [ 76,  73]]) /16, # output-2 anchor boxes
            np.array([[ 6,   11], [ 11,  23], [ 19,  36]]) /8   # output-3 anchor boxes
            ],
            dtype='float64'
        )

    # number of classes and number of anchors
    num_classes = len(class_names)
    num_anchors = anchor_boxes.shape[0] * anchor_boxes.shape[1]

    # input and output
    input_tensor = Input( shape=(input_shape[0], input_shape[1], 3) ) # input
    num_out_filters = ( num_anchors//3 ) * ( 5 + num_classes )        # output

    # build the model
    model = yolo_body(input_tensor, num_out_filters)

    # load weights
    weight_path = r'C:\Users\msi\Documents\GitHub\pictor-ppe\model-data\weights\pictor-ppe-v302-a2-yolo-v3-weights.h5'
    
    model.load_weights( weight_path )

def get_detection(img):
    # save a copy of the img
    act_img = img.copy()

    # shape of the image
    ih, iw = act_img.shape[:2]

    # preprocess the image
    img = letterbox_image(img, input_shape)
    img = np.expand_dims(img, 0)
    image_data = np.array(img) / 255.

    # raw prediction from yolo model
    prediction = model.predict(image_data)

    # process the raw prediction to get the bounding boxes
    boxes = detection(
        prediction,
        anchor_boxes,
        num_classes,
        image_shape = (ih, iw),
        input_shape = (416,416),
        max_boxes = 10,
        score_threshold=0.3,
        iou_threshold=0.45,
        classes_can_overlap=False)

    # convert tensor to numpy
    boxes = boxes[0].numpy()

    # draw the detection on the actual image
    return draw_detection(act_img, boxes, class_names)

if not cap.isOpened():
    raise IOError("We cannot open webcam")

prepare_model()

prev_frame_time = 0
new_frame_time = 0

while True:
    ret, frame = cap.read()
    frame = cv2.flip(frame, 1,1)
    
    # resize our captured frame if we need
    frame = letterbox_image(frame, input_shape)
    # get the detection on the image
    frame = get_detection(frame)
    
    # show us frame with detection
    cv2.imshow("Web cam input", frame)
    if cv2.waitKey(25) & 0xFF == ord("q"):
        cv2.destroyAllWindows()
        break

cap.release()
cv2.destroyAllWindows()