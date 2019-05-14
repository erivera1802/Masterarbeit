import tensorflow as tf
import cv2
import numpy as np

import numpy as np
import tensorflow as tf
from PIL import Image
import time

import yolo_v3
import yolo_v3_tiny

from utils import load_coco_names, draw_boxes, get_boxes_and_inputs, get_boxes_and_inputs_pb, non_max_suppression, \
                  load_graph, letter_box_image

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string(
    'input_img', '', 'Input image')
tf.app.flags.DEFINE_string(
    'output_img', '', 'Output image')
tf.app.flags.DEFINE_string(
    'class_names', 'coco.names', 'File with class names')
tf.app.flags.DEFINE_string(
    'weights_file', 'yolov3.weights', 'Binary file with detector weights')
tf.app.flags.DEFINE_string(
    'data_format', 'NCHW', 'Data format: NCHW (gpu only) / NHWC')
tf.app.flags.DEFINE_string(
    'ckpt_file', './saved_model/model.ckpt', 'Checkpoint file')
tf.app.flags.DEFINE_string(
    'frozen_model', '', 'Frozen tensorflow protobuf model')
tf.app.flags.DEFINE_bool(
    'tiny', False, 'Use tiny version of YOLOv3')

tf.app.flags.DEFINE_integer(
    'size', 416, 'Image size')

tf.app.flags.DEFINE_float(
    'conf_threshold', 0.5, 'Confidence threshold')
tf.app.flags.DEFINE_float(
    'iou_threshold', 0.4, 'IoU threshold')

tf.app.flags.DEFINE_float(
    'gpu_memory_fraction', 1.0, 'Gpu memory fraction to use')

def get_frozen_graph(graph_file):
    """Read Frozen Graph file from disk."""
    with tf.gfile.FastGFile(graph_file, "rb") as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
    return graph_def


def gstreamer_pipeline(capture_width=1280, capture_height=720, display_width=1280, display_height=720, framerate=20,
                       flip_method=0):
    return ('nvarguscamerasrc ! '
            'video/x-raw(memory:NVMM), '
            'width=(int)%d, height=(int)%d, '
            'format=(string)NV12, framerate=(fraction)%d/1 ! '
            'nvvidconv flip-method=%d ! '
            'video/x-raw, width=(int)%d, height=(int)%d, format=(string)BGRx ! '
            'videoconvert ! '
            'video/x-raw, format=(string)BGR ! appsink' % (
            capture_width, capture_height, framerate, flip_method, display_width, display_height))





def show_camera():
    # Window
    while cv2.getWindowProperty('CSI Camera', 0) >= 0:
        ret_val, img = cap.read();

        image = cv2.resize(img, (300, 300))
        scores, boxes, classes, num_detections = tf_sess.run([tf_scores, tf_boxes, tf_classes, tf_num_detections],
                                                             feed_dict={tf_input: image[None, ...]})
        boxes = boxes[0]  # index by 0 to remove batch dimension
        scores = scores[0]
        classes = classes[0]
        num_detections = int(num_detections[0])
        boxes_pixels = []
        for i in range(num_detections):
            # scale box to image coordinates
            box = boxes[i] * np.array([720, 1280, 720, 1280])
            box = np.round(box).astype(int)
            boxes_pixels.append(box)
        boxes_pixels = np.array(boxes_pixels)
        pick = non_max_suppression(boxes_pixels, scores[:num_detections], 0.5)
        # print(pick)

        for i in pick:
            box = boxes_pixels[i]
            box = np.round(box).astype(int)
            # Draw bounding box.
            img = cv2.rectangle(img, (box[1], box[0]), (box[3], box[2]), (0, 255, 0), 2)
            text = "{}:{:.2f}".format(int(classes[i]), scores[i])
            img = cv2.putText(img, text, (box[1], box[0] + 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2,
                              cv2.LINE_AA)
        # print(str(classes))
        cv2.imshow('CSI Camera', img)
        # This also acts as
        keyCode = cv2.waitKey(30) & 0xff
        # Stop the program on the ESC key
        if keyCode == 27:
            break
            cap.release()
            cv2.destroyAllWindows()

    else:
        print('Unable to open camera')

classes = load_coco_names(FLAGS.class_names)
frozenGraph = load_graph(FLAGS.frozen_model)
# To flip the image, modify the flip_method parameter (0 and 2 are the most common)
print(gstreamer_pipeline(flip_method=0))
cap = cv2.VideoCapture(gstreamer_pipeline(flip_method=0), cv2.CAP_GSTREAMER)
if cap.isOpened():
    window_handle = cv2.namedWindow('CSI Camera', cv2.WINDOW_AUTOSIZE)
# Create session and load graph
gpu_options = tf.GPUOptions(allow_growth=True)
config = tf.ConfigProto(
        gpu_options=gpu_options,
        log_device_placement=False)
tf_sess = tf.Session(graph=frozenGraph,config=config)
#tf.import_graph_def(frozenGraph, name='')

boxes, inputs = get_boxes_and_inputs_pb(frozenGraph)
while cv2.getWindowProperty('CSI Camera', 0) >= 0:
    ret_val, img = cap.read();
    cv2_im = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    pil_im = Image.fromarray(cv2_im)
    img_resized = letter_box_image(pil_im, FLAGS.size, FLAGS.size, 128)
    img_resized = img_resized.astype(np.float32)
    detected_boxes = tf_sess.run(boxes, feed_dict={inputs: [img_resized]})
    filtered_boxes = non_max_suppression(detected_boxes,
                                         confidence_threshold=FLAGS.conf_threshold,
                                         iou_threshold=FLAGS.iou_threshold)
    draw_boxes(filtered_boxes, pil_im, classes, (FLAGS.size, FLAGS.size), True)
    img=np.array(pil_im)
    img=cv2.cvtColor(img,cv2.COLOR_RGB2BGR)
    cv2.imshow('CSI Camera', img)
    keyCode = cv2.waitKey(30) & 0xff
    # Stop the program on the ESC key
    if keyCode == 27:
        break
        cap.release()
        cv2.destroyAllWindows()
# decode the results into a list of tuples (class, description, probability)
# (one such list for each sample in the batch)
# print('Predicted:', decode_predictions(preds, top=3)[0])
#show_camera()
tf_sess.close()
