import tensorflow as tf
import cv2
import numpy as np
from numpy.linalg import inv
from PIL import Image
import time
import yolo_v3
import yolo_v3_tiny
from utils import load_coco_names, draw_boxes, get_boxes_and_inputs, get_boxes_and_inputs_pb, non_max_suppression, \
                  load_graph, letter_box_image, convert_to_original_size
from PIL import ImageDraw, Image

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string(
    'class_names', 'names/coco.names', 'File with class names')

# Function to load the graph from the .pb file
def get_frozen_graph(graph_file):
    """Read Frozen Graph file from disk."""
    with tf.gfile.GFile(graph_file, "rb") as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
    with tf.Graph().as_default() as graph:
        tf.import_graph_def(graph_def, name="prefix")
    return graph

# Function to open the pipeline of the camera
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


# Class defining the tracking algorithm and the necessary variables
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
        self.roi = 300

        # Saving
        fourcc = cv2.VideoWriter_fourcc(*'MP4V')
        self.writeVideo = cv2.VideoWriter('outSSD_10Fps.mp4', fourcc, 10.0, (1280, 720))

    def prepare_image(self, img):
        cv2_im = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
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
    def draw_and_show(self,detected_boxes,scores,classes,num_detections,pil_im):
        # filtered_boxes = non_max_suppression(detected_boxes,
        #                                      confidence_threshold=FLAGS.conf_threshold,
        #                                      iou_threshold=FLAGS.iou_threshold)

        #img = np.array(pil_im)
        #img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        #pick = self.non_max_suppression(boxes_pixels, scores[:num_detections], 0.5)
        #for i in pick:
        #    box = boxes_pixels[i]
        #    box = np.round(box).astype(int)
            # Draw bounding box.
            # img = cv2.rectangle(pil_im, (box[1], box[0]), (box[3], box[2]), (0, 255, 0), 2)
            # img = cv2.putText(img, text, (box[1], box[0] + 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2,
            #                   cv2.LINE_AA)
        # img = cv2.putText(img,str(self.counter), org = (0,self.roi),fontFace = cv2.FONT_HERSHEY_SIMPLEX,
        #                   color =(255,255,255), fontScale = 3, thickness = 3)
        img =self.draw_boxes_and_objects(boxes, pil_im, classes, scores, num_detections)
        img = cv2.putText(img,str(self.counter), org = (0,self.roi),fontFace = cv2.FONT_HERSHEY_SIMPLEX,
                          color =(255,255,255), fontScale = 3, thickness = 3)
        self.writeVideo.write(img)
        #cv2.imshow('CSI Camera', img)

    # Process every detected box, that means: Draw the boxes in the images, and create and update tracked objects
    def draw_boxes_and_objects(self,boxes, img, cls_names, scores, num_detections):
        if not self.objects:
            self.first_time = True
        # Box processing, changing from Yolo format
        pick = self.non_max_suppression(boxes_pixels, scores[:num_detections], 0.5)
        for i in pick:
            box = boxes_pixels[i]
            box = np.round(box).astype(int)
            # Draw bounding box.
            # farb = colors_array[classes[i]]
            # farb1 = int(farb[0])
            # farb2 = int(farb[1])
            # farb3 = int(farb[2])
            # arr = (farb1,farb2,farb3)
            img = cv2.rectangle(img, (box[1], box[0]), (box[3], box[2]), (255,0,0), 2)
            img = cv2.putText(img,classesFile[classes[i]-1],(box[1], box[0]),cv2.FONT_HERSHEY_SIMPLEX, 1, (255,0,0))
            self.tracking_objects(box)
        # For all detected objects
        if self.objects:
            for key in self.objects.keys():
                # r is the probability of existence, and it determines the radius of the circle
                r = int(self.update(key))
                gate = 100
                #(key)
                img =cv2.circle(img,(self.objects[key]['State'][0],self.objects[key]['State'][1]),r,(255, 0, 0),1)
                #draw.text((self.objects[key]['State'][0] - 2*r,  self.objects[key]['State'][1] - 2*r), str(key))
                if self.objects[key]['Update']:
                    # draw.ellipse((self.objects[key]['State'][0] - r, self.objects[key]['State'][1] - r, self.objects[key]['State'][0] + r,
                    #               self.objects[key]['State'][1] + r),
                    #              fill=(255, 0, 0, 255))
                    img = cv2.circle(img,(self.objects[key]['State'][0],self.objects[key]['State'][1]),r,(255,0,0))
                else :
                    img = cv2.circle(img, (self.objects[key]['State'][0], self.objects[key]['State'][1]), r,
                                     (255, 255, 0))
                #     draw.ellipse((self.objects[key]['State'][0] - r, self.objects[key]['State'][1] - r, self.objects[key]['State'][0] + r,
                #                   self.objects[key]['State'][1] + r),
                #                  fill=(255, 255, 0, 50))
                #     draw.ellipse((self.objects[key]['State'][0] - gate, self.objects[key]['State'][1] - gate,
                #                   self.objects[key]['State'][0] + gate,
                #                   self.objects[key]['State'][1] + gate))
                img =cv2.line(img,(0, self.roi), (1280, self.roi),(0,0,255))
                #print(len(self.objects),self.objects[key]['Update'])
                #if self.objects[key]['Prob'] < 0.2:
                #    del self.objects[key]
                #    self.discarded.append(key)
                # Reset 'Update' parameter of all the actual objects
                if self.objects[key]['Past'] ==False and self.objects[key]['Present'] ==True:
                    self.counter = self.counter+1
                    img = cv2.line(img, (0, self.roi), (1280, self.roi), (255, 0, 0))
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
        #     self.objects[key]['State'], self.objects[key]['P'] = self.predictKalman(self.objects[key]['State'], self.objects[key]['P'],
        #                                                                         self.A, self.Q)
        #     self.objects[key]['X'] = self.objects[key]['State'][0]
        #     self.objects[key]['Y'] = self.objects[key]['State'][1]
        #     self.objects[key]['vX'] = self.objects[key]['State'][2]
        #     self.objects[key]['vY'] = self.objects[key]['State'][3]
        # Binary Bayes Filter
        l = np.log(sensor / (1 - sensor))
        l_past = np.log(self.objects[key]['Prob'] / (1 - self.objects[key]['Prob']))
        L = l + l_past

        # Corrected the 'Static' asumption
        if np.abs(L) > 5:
            L = 5 * np.sign(L)

        # Get the probability of existence of the box
        P = 1 - 1 / (1 + np.exp(L))
        #print(P, L, l, l_past)

        # Radius of the ellipse
        r = 15 * P

        # Update object probability
        self.objects[key]['Prob'] = P

        # Reset 'Update' parameter of all the actual objects
        #self.objects[key]['Update'] = False
        return r

    def tracking_objects(self,box):
        # Size of the gate to accept a position into an existent object
        gate = 50

        # Dictionary for one object
        # Features:
        # X: Position in X
        # Y: Position in Y
        # Prob: Existence probability
        # Update: Was the object detected in the actual iteration?

        obj = dict()
        # Change coordinates from x0, y0, x1, y1 to x, y, width, height
        x, y = self.change_coordinates(box)

        # If it is the first time, a new object has to initialize the dictionary
        if  self.first_time:
            self.first_time = False
            #obj['X'] = x
            #obj['Y'] = y
            #obj['vX'] = 0
            #obj['vY'] = 0
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
            # Append the object to the dictionary in the 'consecutive' position
            #self.objects[self.consecutive] = obj
            #self.consecutive = self.consecutive + 1
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
                #actualxy = np.array([x,y,self.objects[key]['vX'],self.objects[key]['vY']])
                #actualxy =np.expand_dims(actualxy,axis = 1)
                self.objects[key]['State'], self.objects[key]['P'] = self.predictKalman(self.objects[key]['State'], self.objects[key]['P'], self.A, self.Q)
                #print(self.objects[key]['State'][2], self.objects[key]['State'][3])
                # Calculate the distance between the new measurement and all the saved objects
                #distance = np.sqrt((self.objects[key]['X'] - actualX) ** 2 + (self.objects[key]['Y'] - actualY) ** 2)
                distance = np.sqrt((self.objects[key]['State'][0] - actualX) ** 2 + (self.objects[key]['State'][1] - actualY) ** 2)

                # If the distance is smaller than the gate, the measurement is the new position of the object
                if distance < gate:
                    #meas = np.array([actualX, actualY, 0, 0])
                    meas = np.array([actualX, actualY])
                    meas =np.expand_dims(meas,axis = 1)
                    # print(self.objects[key]['State'].shape)
                    # print(meas.shape)
                    # print(self.H.shape)
                    self.objects[key]['State'], self.objects[key]['P'] = self.updateKalman(self.objects[key]['State'], meas, self.objects[key]['P'], self.H, self.R)

                    #self.objects[key]['State'] = meas
                    #self.objects[key]['State'][0] = actualX
                    #self.objects[key]['State'][1] = actualY
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
                #obj['X'] = x
                #obj['Y'] = y
                #obj['vX'] = 0
                #obj['vY'] = 0
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

    # # Function to change from cv2 to pil and resizing

    def predictKalman(self,x, P, A, Q):
        x = A @ x
        P = A @ P @ A.T + Q
        return (x, P)


    def updateKalman(self,x, z, P, H, R):
        K = P @ H.T @ inv(H @ P @ H.T + R)
        print(x.shape,K.shape,z.shape,H.shape)
        x = x + K @ (z - H @ x)
        P = (np.eye(len(x)) - K @ H) @ P
        return x, P

def colors(classes):
    farbe =dict()
    for i,classe in enumerate(classes):
        farbe[i] = tuple(np.random.randint(0, 256, 3))
    return farbe

# Load the classes file and the graph
classesFile = load_coco_names(FLAGS.class_names)
pb_fname = "./model/coco_vehicle_90k.pb"
print('Loading graph')
frozenGraph = get_frozen_graph(pb_fname)

colors_array = colors(classesFile)

# Prepare the cv2 window
#if cap.isOpened():
#    window_handle = cv2.namedWindow('CSI Camera', cv2.WINDOW_AUTOSIZE)

# Create session and load graph
# Configure the tensorflow session, especially with allow_growth, so it doesnt fails to get memory
gpu_options = tf.GPUOptions(allow_growth=True)
config = tf.ConfigProto(
        gpu_options=gpu_options,
        log_device_placement=False)

# Initialize the session: ACHTUNG!! This is the more efficient way, in comparison to with tf.Session as sess:
#I am not sure why, but that way freezes the Nanoboard and make the loading process really slow
tf_sess = tf.Session(graph=frozenGraph,config=config)

# Get the names of the inputs and outputs of the networks
tf_input = tf_sess.graph.get_tensor_by_name('prefix/image_tensor:0')
tf_scores = tf_sess.graph.get_tensor_by_name('prefix/detection_scores:0')
tf_boxes = tf_sess.graph.get_tensor_by_name('prefix/detection_boxes:0')
tf_classes = tf_sess.graph.get_tensor_by_name('prefix/detection_classes:0')
tf_num_detections = tf_sess.graph.get_tensor_by_name('prefix/num_detections:0')

# Saving data to debug ant test algorithms
#f = open('data.txt','w')
#f.write('X0,Y0,X1,Y1,Indice\n')

tracker = TrackingAlgorithm()
cap = cv2.VideoCapture('bruecke10.mp4')
# if cap.isOpened():
#     window_handle = cv2.namedWindow('Video', cv2.WINDOW_AUTOSIZE)

print(cap.get(3), cap.get(4))
# While you get something from the camera
#while cv2.getWindowProperty('Video', 0) >= 0:
while True:
    if tracker.first_time == False:
        tracker.dt =time.time()-previous
    print('Time')
    print(tracker.dt)
    previous = time.time()
    ret_val, img = cap.read()


    # Prepare the image
    num_rows, num_cols = img.shape[:2]
    rotation_matrix = cv2.getRotationMatrix2D((num_cols / 2, num_rows / 2), 180, 1)
    #img = cv2.warpAffine(img, rotation_matrix, (num_cols, num_rows))
    img_resized = tracker.prepare_image(img)
    # Run the network
    scores, boxes, classes, num_detections = tf_sess.run([tf_scores, tf_boxes,tf_classes, tf_num_detections],
                                                         feed_dict={tf_input: img_resized[None, ...]})
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

    # Show the detected image and process tracking
    tracker.draw_and_show(boxes_pixels,scores,classes,num_detections,img)
    # tracker.count = tracker.count + 1

    # Check if the window should be closed
    keyCode = cv2.waitKey(30) & 0xff

    # Stop the program on the ESC key
    if keyCode == 27:
        break
        cap.release()
        cv2.destroyAllWindows()

# Close the tf Session
tf_sess.close()