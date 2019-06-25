import tensorflow as tf
import cv2
import numpy as np
from PIL import Image
import time
import yolo_v3
import yolo_v3_tiny
from utils import load_coco_names, draw_boxes, get_boxes_and_inputs, get_boxes_and_inputs_pb, non_max_suppression, \
                  load_graph, letter_box_image, convert_to_original_size
from PIL import ImageDraw, Image

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string(
    'class_names', 'coco.names', 'File with class names')
tf.app.flags.DEFINE_string(
    'weights_file', 'yolov3.weights', 'Binary file with detector weights')
tf.app.flags.DEFINE_string(
    'data_format', 'NCHW', 'Data format: NCHW (gpu only) / NHWC')

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

# Function to load the graph from the .pb file
def get_frozen_graph(graph_file):
    """Read Frozen Graph file from disk."""
    with tf.gfile.FastGFile(graph_file, "rb") as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
    return graph_def

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

        # Initialize object to save video
        fourcc = cv2.VideoWriter_fourcc(*'MJPG')
        self.writeVideo = cv2.VideoWriter('output.avi', fourcc, 15.0, (int(cap.get(3)), int(cap.get(4))))

    # Take the boxes, supress the non_max, draw the boxes and show the images
    def draw_and_show(self,detected_boxes,pil_im):
        filtered_boxes = non_max_suppression(detected_boxes,
                                             confidence_threshold=FLAGS.conf_threshold,
                                             iou_threshold=FLAGS.iou_threshold)
        self.draw_boxes_and_objects(filtered_boxes, pil_im, classes, (FLAGS.size, FLAGS.size), True)
        img = np.array(pil_im)
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        cv2.imshow('CSI Camera', img)
        self.writeVideo.write(img)

    # Process every detected box, that means: Draw the boxes in the images, and create and update tracked objects
    def draw_boxes_and_objects(self,boxes, img, cls_names, detection_size, is_letter_box_image):
        draw = ImageDraw.Draw(img)
        if not self.objects:
            self.first_time = True
        # For every detected box from all clases
        for cls, bboxs in boxes.items():
            color = tuple(np.random.randint(0, 256, 3))

            for box, score in bboxs:
                # Box processing, changing from Yolo format
                box = convert_to_original_size(box, np.array(detection_size),
                                               np.array(img.size),
                                               is_letter_box_image)
                # Create and update tracking objects from the boxes
                self.tracking_objects(box)
                # Draw boxes and names
                draw.rectangle(box, outline=colors_array[cls])
                draw.text(box[:2], '{} {:.2f}%'.format(
                    cls_names[cls], score * 100), fill=colors_array[cls])
        # For all detected objects
        if self.objects:
            for key in self.objects.keys():
                # r is the probability of existence, and it determines the radius of the circle
                r = self.update(key)
                print(key)
                draw.ellipse((self.objects[key]['X'] - r, self.objects[key]['Y'] - r, self.objects[key]['X'] + r, self.objects[key]['Y'] + r),
                             fill=(255, 0, 0, 255))
                #if self.objects[key]['Prob'] < 0.2:
                #    del self.objects[key]
                #    self.discarded.append(key)
            for key in [key for key in self.objects if self.objects[key]['Prob'] < 0.3]:
                del self.objects[key]
                self.discarded.append(key)
            # Update the probability of existence of an object through a Binary Bayes Filter
    def update(self,key):

        # If not detected, probability of existence of an object is 0.3
        sensor = 0.3

        # If detected, probability of existence of an object is 0.7
        if self.objects[key]['Update']:
            sensor = 0.7

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
        self.objects[key]['Update'] = False
        return r

    def tracking_objects(self,box):


        # Size of the gate to accept a position into an existent object
        gate = 300

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
            obj['X'] = x
            obj['Y'] = y
            obj['Prob'] = 0.5
            obj['Update'] = True

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

                # Calculate the distance between the new measurement and all the saved objects
                distance = np.sqrt((self.objects[key]['X'] - actualX) ** 2 + (self.objects[key]['Y'] - actualY) ** 2)

                # If the distance is smaller than the gate, the measurement is the new position of the object
                if distance < gate:
                    self.objects[key]['X'] = actualX
                    self.objects[key]['Y'] = actualY
                    self.objects[key]['Update'] = True
                    newObject = False
                    break

                # If not, a new object must be created
                else:
                    newObject = True

            # Create a new object with the position of the actual measurement
            if newObject:
                obj['X'] = x
                obj['Y'] = y
                obj['Prob'] = 0.5
                obj['Update'] = True
                if not self.discarded:
                    self.objects[self.consecutive] = obj
                    self.consecutive = self.consecutive + 1
                else:
                    index = self.discarded.pop(0)
                    self.objects[index] = obj

    # Change coordinates from x0, y0, x1, y1 to x, y, width, height
    def change_coordinates(self,box):
        width = box[2]-box[0]
        height = box[3]-box[1]
        x = box[0]+width/2
        y = box[1]+height/2
        return x,y

    # Function to change from cv2 to pil and resizing
    def prepare_image(self,img):
        cv2_im = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        pil_im = Image.fromarray(cv2_im)
        img_resized = letter_box_image(pil_im, FLAGS.size, FLAGS.size, 128)
        img_resized = img_resized.astype(np.float32)
        return img_resized,pil_im

def colors(classes):
    farbe =dict()
    for i,classe in enumerate(classes):
        farbe[i] = tuple(np.random.randint(0, 256, 3))
    return farbe

# Load the classes file and the graph
classes = load_coco_names(FLAGS.class_names)
frozenGraph = load_graph(FLAGS.frozen_model)

colors_array = colors(classes)

# Initialize the pipeline for the camera
# To flip the image, modify the flip_method parameter (0 and 2 are the most common)
print(gstreamer_pipeline(flip_method=2))
cap = cv2.VideoCapture(gstreamer_pipeline(flip_method=2), cv2.CAP_GSTREAMER)

# Prepare the cv2 window
if cap.isOpened():
    window_handle = cv2.namedWindow('CSI Camera', cv2.WINDOW_AUTOSIZE)

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
boxes, inputs = get_boxes_and_inputs_pb(frozenGraph)

# Saving data to debug ant test algorithms
#f = open('data.txt','w')
#f.write('X0,Y0,X1,Y1,Indice\n')

tracker = TrackingAlgorithm()

# While you get something from the camera
while cv2.getWindowProperty('CSI Camera', 0) >= 0:
    ret_val, img = cap.read()


    # Prepare the image
    num_rows, num_cols = img.shape[:2]
    rotation_matrix = cv2.getRotationMatrix2D((num_cols / 2, num_rows / 2), 180, 1)
    img = cv2.warpAffine(img, rotation_matrix, (num_cols, num_rows))
    img_resized, pil_im = tracker.prepare_image(img)

    # Run the network
    detected_boxes = tf_sess.run(boxes, feed_dict={inputs: [img_resized]})

    # Show the detected image and process tracking
    tracker.draw_and_show(detected_boxes,pil_im)
    tracker.count = tracker.count + 1

    # Check if the window should be closed
    keyCode = cv2.waitKey(30) & 0xff

    # Stop the program on the ESC key
    if keyCode == 27:
        break
        tracker.writeVideo.release()
        cap.release()
        cv2.destroyAllWindows()

# Close the tf Session
tf_sess.close()
