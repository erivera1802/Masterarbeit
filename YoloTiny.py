# MIT License
# Copyright (c) 2019 JetsonHacks
# See license
# Using a CSI camera (such as the Raspberry Pi Version 2) connected to a
# NVIDIA Jetson Nano Developer Kit using OpenCV
# Drivers for the camera and OpenCV are included in the base image
# -*- coding: utf-8 -*-
import cv2
import numpy as np
import tensorflow as tf
from PIL import Image
import time

import yolo_v3
import yolo_v3_tiny

from utils import load_coco_names, draw_boxes, get_boxes_and_inputs, get_boxes_and_inputs_pb, non_max_suppression, \
                  load_graph, letter_box_image
# gstreamer_pipeline returns a GStreamer pipeline for capturing from the CSI camera
# Defaults to 1280x720 @ 60fps
# Flip the image by setting the flip_method (most common values: 0 and 2)
# display_width and display_height determine the size of the window on the screen


def gstreamer_pipeline (capture_width=1280, capture_height=720, display_width=1280, display_height=720, framerate=60, flip_method=0) :
	return ('nvarguscamerasrc ! '
	'video/x-raw(memory:NVMM), '
	'width=(int)%d, height=(int)%d, '
	'format=(string)NV12, framerate=(fraction)%d/1 ! '
	'nvvidconv flip-method=%d ! '
	'video/x-raw, width=(int)%d, height=(int)%d, format=(string)BGRx ! '
	'videoconvert ! '
	'video/x-raw, format=(string)BGR ! appsink' % (capture_width,capture_height,framerate,flip_method,display_width,display_height))

def show_camera(sess,boxes,inputs):
	# To flip the image, modify the flip_method parameter (0 and 2 are the most common)
    classes = load_coco_names(FLAGS.class_names)
    print(gstreamer_pipeline(flip_method=0))
    cap = cv2.VideoCapture(gstreamer_pipeline(flip_method=0), cv2.CAP_GSTREAMER)
    if cap.isOpened():
        window_handle = cv2.namedWindow('CSI Camera', cv2.WINDOW_AUTOSIZE)
        while cv2.getWindowProperty('CSI Camera', 0) >= 0:
            ret_val, img = cap.read();
            cv2_im = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            pil_im = Image.fromarray(cv2_im)
            img_resized = letter_box_image(pil_im, FLAGS.size, FLAGS.size, 128)
            img_resized = img_resized.astype(np.float32)
            detected_boxes = sess.run(boxes, feed_dict={inputs: [img_resized]})
            filtered_boxes = non_max_suppression(detected_boxes,
                                                 confidence_threshold=FLAGS.conf_threshold,
                                                 iou_threshold=FLAGS.iou_threshold)
            draw_boxes(filtered_boxes, pil_im, classes, (FLAGS.size, FLAGS.size), True)
            img=np.array(pil_im)
            img=cv2.cvtColor(img,cv2.COLOR_RGB2BGR)
            cv2.imshow('CSI Camera', img)
            keyCode = cv2.waitKey(30) & 0xff
            if keyCode == 27:
                break
        cap.release()
        cv2.destroyAllWindows()
    else:
        print('Unable to open camera')




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

def main(argv=None):

    #gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=FLAGS.gpu_memory_fraction)
    gpu_options = tf.GPUOptions(allow_growth=True)
    config = tf.ConfigProto(
        gpu_options=gpu_options,
        log_device_placement=False,
    )

    img = Image.open(FLAGS.input_img)

    classes = load_coco_names(FLAGS.class_names)

    if FLAGS.frozen_model:

        t0 = time.time()
        frozenGraph = load_graph(FLAGS.frozen_model)
        print("Loaded graph in {:.2f}s".format(time.time()-t0))

        boxes, inputs = get_boxes_and_inputs_pb(frozenGraph)

        with tf.Session(graph=frozenGraph, config=config) as sess:
            t0 = time.time()
            show_camera(sess,boxes,inputs)

    else:
        if FLAGS.tiny:
            model = yolo_v3_tiny.yolo_v3_tiny
        else:
            model = yolo_v3.yolo_v3

        boxes, inputs = get_boxes_and_inputs(model, len(classes), FLAGS.size, FLAGS.data_format)

        saver = tf.train.Saver(var_list=tf.global_variables(scope='detector'))

        with tf.Session(config=config) as sess:
            t0 = time.time()
            saver.restore(sess, FLAGS.ckpt_file)
            print('Model restored in {:.2f}s'.format(time.time()-t0))

            t0 = time.time()


    #filtered_boxes = detected_boxes
    #print("Predictions found in {:.2f}s".format(time.time() - t0))

    #draw_boxes(filtered_boxes, img, classes, (FLAGS.size, FLAGS.size), True)
    #img.save(FLAGS.output_img)


if __name__ == '__main__':
    tf.app.run()
