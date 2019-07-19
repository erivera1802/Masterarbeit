#!/usr/bin/python
#
# Copyright (c) 2019, NVIDIA CORPORATION. All rights reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a
# copy of this software and associated documentation files (the "Software"),
# to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense,
# and/or sell copies of the Software, and to permit persons to whom the
# Software is furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL
# THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
# FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.
#

import jetson.inference
import jetson.utils

import argparse
import numpy as np
from numpy.linalg import inv
import time
import cv2


class TrackingAlgorithm:

    def __init__(self):
        # Iteration, necessary to asign an iteration to the data
        self.count = 0

        # Is the first time an object is recognized?
        self.first_time = True

        # Variable to add new objects
        self.consecutive = 0
        self.discarded=[]

        # Dictionary with recognized objects
        self.objects = dict()

        # Kalman filter
        # Time difference
        self.dt = 0.2
        self.A = np.array([[1,0,self.dt,0],[0,1,0,self.dt],[0,0,1,0],[0,0,0,1]])
        self.P0 = 1*np.eye(4)
        self.Q = np.array([[1,0,0,0],[0,1,0,0],[0,0,10,0],[0,0,0,10]])
        self.H = np.array([[1, 0, 0, 0], [0, 1, 0, 0]])
        self.R = 0.1*np.eye(2)

        # Counting
        self.counter = 0
        self.roi = 500

        # Saving
        #fourcc = cv2.VideoWriter_fourcc(*'MP4V')
        #self.writeVideo = cv2.VideoWriter('outSSD.mp4', fourcc, 30.0, (1280, 720))

    def prepare_image(self, img):
        img_resized = cv2.resize(img, (300, 300))
        return img_resized

    def non_max_suppression(self,boxes, probs=None, overlapThresh=0.3):
        """Non-max suppression

        Arguments:
            boxes {np.array} -- a Numpy list of boxes, each one are [x1, y1, x2, y2]
        Keyword arguments
            probs {np.array} -- Probabilities associated with each box. (default: {None})
            nms_threshold {float} -- Overlapping threshold 0~1. (default: {0.3})

        Returns:
            list -- A list of selected box indexes.
        """
        # if there are no boxes, return an empty list
        if len(boxes) == 0:
            return []

        # if the bounding boxes are integers, convert them to floats -- this
        # is important since we'll be doing a bunch of divisions
        if boxes.dtype.kind == "i":
            boxes = boxes.astype("float")

        # initialize the list of picked indexes
        pick = []

        # grab the coordinates of the bounding boxes
        x1 = boxes[:, 0]
        y1 = boxes[:, 1]
        x2 = boxes[:, 2]
        y2 = boxes[:, 3]

        # compute the area of the bounding boxes and sort the bounding
        # boxes by the bottom-right y-coordinate of the bounding box
        area = (x2 - x1 + 1) * (y2 - y1 + 1)
        idxs = np.argsort(y2)

        # keep looping while some indexes still remain in the indexes
        # list
        while len(idxs) > 0:
            # grab the last index in the indexes list and add the
            # index value to the list of picked indexes
            last = len(idxs) - 1
            i = idxs[last]
            pick.append(i)

            # find the largest (x, y) coordinates for the start of
            # the bounding box and the smallest (x, y) coordinates
            # for the end of the bounding box
            xx1 = np.maximum(x1[i], x1[idxs[:last]])
            yy1 = np.maximum(y1[i], y1[idxs[:last]])
            xx2 = np.minimum(x2[i], x2[idxs[:last]])
            yy2 = np.minimum(y2[i], y2[idxs[:last]])

            # compute the width and height of the bounding box
            w = np.maximum(0, xx2 - xx1 + 1)
            h = np.maximum(0, yy2 - yy1 + 1)

            # compute the ratio of overlap
            overlap = (w * h) / area[idxs[:last]]

            # delete all indexes from the index list that have
            idxs = np.delete(idxs, np.concatenate(([last],
                                                   np.where(overlap > overlapThresh)[0])))

        # return only the bounding boxes indexes
        return pick
    # Take the boxes, supress the non_max, draw the boxes and show the images
    def draw_and_show(self,detections):
        # Do all the processing of the boxes and objects

        img =self.draw_boxes_and_objects(detections)
        # Draw the line where the objects are counted

        img = cv2.line(img, (0, self.roi), (1280, self.roi), (0, 0, 255))
        # Draw the number of objects
        img = cv2.putText(img,str(self.counter), org = (0,self.roi),fontFace = cv2.FONT_HERSHEY_SIMPLEX,
                         color =(255,255,255), fontScale = 3, thickness = 3)
        #self.writeVideo.write(img)

        cv2.imshow('Video', img)

#     # Process every detected box, that means: Draw the boxes in the images, and create and update tracked objects
    def draw_boxes_and_objects(self,detections):
        if not self.objects:
            self.first_time = True
        img = np.zeros((720, 1280,3))
        for detection in detections:
            farb = colors_array[detection.ClassID-1]
            farb1 = int(farb[0])
            farb2 = int(farb[1])
            farb3 = int(farb[2])
            arr = (farb1,farb2,farb3)
            img = cv2.rectangle(img, (int(detection.Left), int(detection.Top)), (int(detection.Right),int(detection.Bottom)), arr, 2)
            img = cv2.putText(img,classesFile[detection.ClassID-1],(int(detection.Left), int(detection.Top)),cv2.FONT_HERSHEY_SIMPLEX, 1, arr)
            self.tracking_objects(detection.Center)
        # For all detected objects
        if self.objects:
            for key in self.objects.keys():
                # r is the probability of existence, and it determines the radius of the circle
                r = int(self.update(key))
                gate = 100
                #(key)
                img =cv2.circle(img,(self.objects[key]['State'][0],self.objects[key]['State'][1]),r,(255, 0, 0),3)
                # Reset 'Update' parameter of all the actual objects
                if self.objects[key]['Past'] ==False and self.objects[key]['Present'] ==True:
                    self.counter = self.counter+1
                    img = cv2.line(img, (0, self.roi), (1280, self.roi), (255, 255, 255),3)
                self.objects[key]['Past'] = self.objects[key]['Present']
                self.objects[key]['Update'] = False
            for key in [key for key in self.objects if self.objects[key]['Prob'] < 0.3]:
                del self.objects[key]
                self.discarded.append(key)
        return img
            # Update the probability of existence of an object through a Binary Bayes Filter
    def update(self,key):

        # If not detected, probability of existence of an object is 0.3
        sensor = 0.3

        # If detected, probability of existence of an object is 0.7
        if self.objects[key]['Update']:
            sensor = 0.7
        if not self.objects[key]['Update']:
            self.objects[key]['State'], self.objects[key]['P'] = self.predictKalman(self.objects[key]['State'],
                                                                                    self.objects[key]['P'], self.A,
                                                                                    self.Q)
        # Binary Bayes Filter
        l = np.log(sensor / (1 - sensor))
        l_past = np.log(self.objects[key]['Prob'] / (1 - self.objects[key]['Prob']))
        L = l + l_past

        # Corrected the 'Static' asumption
        if np.abs(L) > 5:
            L = 5 * np.sign(L)

        # Get the probability of existence of the box
        P = 1 - 1 / (1 + np.exp(L))

        # Radius of the ellipse
        r = 15 * P

        # Update object probability
        self.objects[key]['Prob'] = P
        return r
#
    def tracking_objects(self,center):
        # Size of the gate to accept a position into an existent object
        gate = 100

        # Dictionary for one object
        # Features:
        # X: Position in X
        # Y: Position in Y
        # Prob: Existence probability
        # Update: Was the object detected in the actual iteration?

        obj = dict()
        # Change coordinates from x0, y0, x1, y1 to x, y, width, height
        x = center[0]
        y = center[1]

        # If it is the first time, a new object has to initialize the dictionary
        if  self.first_time:
            self.first_time = False
            obj['Prob'] = 0.5
            obj['Update'] = True
            obj['P'] = self.P0
            zustand = np.array([x, y, 0, 0])
            obj['State'] = np.expand_dims(zustand,axis = 1)
            if obj['State'][1]<self.roi:
                obj['Past'] = False
                obj['Present'] = False
            else:
                obj['Past'] = True
                obj['Present'] = True

            if not self.discarded:
                self.objects[self.consecutive] = obj
                self.consecutive = self.consecutive + 1
            else:
                index = self.discarded.pop(0)
                self.objects[index] = obj

        else:
            for key in self.objects.keys():
                actualX = x
                actualY = y
                distance = np.sqrt((self.objects[key]['State'][0] - actualX) ** 2 + (self.objects[key]['State'][1] - actualY) ** 2)

                # If the distance is smaller than the gate, the measurement is the new position of the object
                if distance < gate:
                    meas = np.array([actualX, actualY, 0, 0])
                    meas =np.expand_dims(meas,axis = 1)
                    self.objects[key]['State'] = meas
                    if self.objects[key]['State'][1] < self.roi:
                        self.objects[key]['Present'] = False
                    else:
                        self.objects[key]['Present'] = True
                    self.objects[key]['Update'] = True
                    newObject = False
                    break

                # If not, a new object must be created
                else:
                    newObject = True

            # Create a new object with the position of the actual measurement
            if newObject:
                obj['Prob'] = 0.5
                obj['Update'] = True
                obj['P'] = self.P0
                zustand = np.array([x, y, 0, 0])
                obj['State'] = np.expand_dims(zustand, axis=1)
                if obj['State'][1] < self.roi:
                    obj['Past'] = False
                    obj['Present'] = False
                else:
                    obj['Past'] = True
                    obj['Present'] = True
                if not self.discarded:
                    self.objects[self.consecutive] = obj
                    self.consecutive = self.consecutive + 1
                else:
                    index = self.discarded.pop(0)
                    self.objects[index] = obj

    # Change coordinates from x0, y0, x1, y1 to x, y, width, height
    def change_coordinates(self,box):
        width = box[3]-box[1]
        height = box[2]-box[0]
        x = box[1]+width/2
        y = box[0]+height/2
        return x,y
#
#     # # Function to change from cv2 to pil and resizing
#
    def predictKalman(self,x, P, A, Q):
        x = A @ x
        P = A @ P @ A.T + Q
        return (x, P)


    def updateKalman(self,x, z, P, H, R):
        K = P @ H.T @ inv(H @ P @ H.T + R)

        x = x + K @ (z - H @ x)
        P = (np.eye(len(x)) - K @ H) @ P
        return x, P
def colors(classes):
    farbe =dict()
    for i,classe in enumerate(classes):
        farbe[i] = tuple(np.random.randint(0, 256, 3))
    return farbe

def load_coco_names(file_name):
    names = {}
    with open(file_name) as f:
        for id, name in enumerate(f):
            names[id] = name
    return names
# parse the command line
parser = argparse.ArgumentParser(description="Locate objects in a live camera stream using an object detection DNN.",
						   formatter_class=argparse.RawTextHelpFormatter, epilog=jetson.inference.detectNet.Usage())

parser.add_argument("--network", type=str, default="pednet", help="pre-trained model to load, see below for options")
parser.add_argument("--threshold", type=float, default=0.5, help="minimum detection threshold to use")
parser.add_argument("--camera", type=str, default="0", help="index of the MIPI CSI camera to use (NULL for CSI camera 0)\nor for VL42 cameras the /dev/video node to use.\nby default, MIPI CSI camera 0 will be used.")
parser.add_argument("--width", type=int, default=1280, help="desired width of camera stream (default is 1280 pixels)")
parser.add_argument("--height", type=int, default=720, help="desired height of camera stream (default is 720 pixels)")

opt, argv = parser.parse_known_args()

# load the object detection network
net = jetson.inference.detectNet(opt.network, argv, opt.threshold)

classesFile = load_coco_names('coco.names')
colors_array = colors(classesFile)

# create the camera and display
cap = cv2.VideoCapture('bruecke2.mp4')
# if cap.isOpened():
#     window_handle = cv2.namedWindow('Video', cv2.WINDOW_NORMAL)
#     cv2.resizeWindow('Video', 427, 240)
display = jetson.utils.glDisplay()
tracker = TrackingAlgorithm()
window_handle = cv2.namedWindow('Video', cv2.WINDOW_NORMAL)
cv2.resizeWindow('Video', 427, 240)
# process frames until user exits
previous = time.time()
while cv2.getWindowProperty('Video', 0) >= 0:
    # capture the image
    #img, width, height = camera.CaptureRGBA()
    previous = time.time()
    ret_val, img = cap.read()

    b_channel, g_channel, r_channel = cv2.split(img)

    alpha_channel = np.ones(b_channel.shape, dtype=b_channel.dtype) * 50  # creating a dummy alpha channel image.

    img_RGBA = cv2.merge((r_channel, g_channel, b_channel, alpha_channel))
    height, width, channels = img_RGBA.shape
    img = jetson.utils.cudaFromNumpy(img_RGBA)


    # detect objects in the image (with overlay)
    detections = net.Detect(img, width, height)

    tracker.draw_and_show(detections)
    # render the image
    display.RenderOnce(img, width, height)

    # update the title bar
    display.SetTitle("{:s} | Network {:.0f} FPS".format(opt.network, 1000.0 / net.GetNetworkTime()))
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    # synchronize with the GPU
    if len(detections) > 0:
        jetson.utils.cudaDeviceSynchronize()
    actual = time.time()
    print(actual-previous)
    # print out performance info
    #net.PrintProfilerTimes()


