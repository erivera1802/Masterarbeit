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

# Take the boxes, supress the non_max, draw the boxes and show the images
def draw_and_show(detected_boxes,pil_im, count,consecutive):
    global first_time
    filtered_boxes = non_max_suppression(detected_boxes,
                                         confidence_threshold=FLAGS.conf_threshold,
                                         iou_threshold=FLAGS.iou_threshold)
    draw_save_boxes(filtered_boxes, pil_im, classes, (FLAGS.size, FLAGS.size), True, count,consecutive)
    img = np.array(pil_im)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    cv2.imshow('CSI Camera', img)

def draw_save_boxes(boxes, img, cls_names, detection_size, is_letter_box_image, count,consecutive):
    draw = ImageDraw.Draw(img)
    gate = 300
    global first_time
    #print(type(boxes.items()))
    if (bool(boxes.items())== False):
        print('0,0,0,0,'+str(count))
        f.write('0,0,0,0,0,'+str(count)+'\n')
    for cls, bboxs in boxes.items():
        color = tuple(np.random.randint(0, 256, 3))

        for box, score in bboxs:
            box = convert_to_original_size(box, np.array(detection_size),
                                           np.array(img.size),
                                           is_letter_box_image)
            #print(box)
            #str1 = ','.join(str(e) for e in box)
            #print(str1+','+str(cls)+','+str(count)+'\n')
            #f.write(str1+','+str(cls)+','+str(count)+'\n')
            obj = dict()
            x, y = change_coordinates(box)
            #print(first_time)
            if first_time == False:

                first_time= True
                obj['X'] = x
                obj['Y'] = y
                obj['Prob'] = 0.4
                obj['Update'] = True
                objects[consecutive] = obj
                consecutive = consecutive + 1
            else:
                for key in objects.keys():
                    actualX = x
                    actualY = y
                    distance = np.sqrt((objects[key]['X'] - actualX) ** 2 + (objects[key]['Y'] - actualY) ** 2)

                    if distance < gate:
                        objects[key]['X'] = actualX
                        objects[key]['Y'] = actualY
                        objects[key]['Update'] = True
                        newObject = False
                        break
                    else:
                        newObject = True

                if newObject == True:
                    obj['X'] = x
                    obj['Y'] = y
                    obj['Prob'] = 0.5
                    obj['Update'] = True
                    objects[consecutive] = (obj)
                    consecutive = consecutive + 1
            # for key in objects.keys():
            #     sensor = 0.3
            #     if objects[key]['Update'] == True:
            #         sensor = 0.7
            #     l = np.log(sensor / (1 - sensor))
            #     l_past = np.log(objects[key]['Prob']/(1-objects[key]['Prob']))
            #     L = l+l_past
            #     P = 1 - 1 / (1 + np.exp(L))
            #     #print(sensor)
            #     #print('Hey')
            #     #print(objects[key]['Prob'])
            #     #print(P,L,l,l_past)
            #     r =15*P
            #     objects[key]['Prob'] = P
            #     #print(objects[key])
            #     print(objects[key]['Prob'])
            #     objects[key]['Update'] = False
            #     draw.ellipse((objects[key]['X'] - r, objects[key]['Y'] - r, objects[key]['X'] + r, objects[key]['Y'] + r), fill=(255, 0, 0, 255))
            #print(objects)
            draw.rectangle(box, outline=color)
            draw.text(box[:2], '{} {:.2f}%'.format(
                cls_names[cls], score * 100), fill=color)
    if objects:
        for key in objects.keys():
            sensor = 0.3
            if objects[key]['Update'] == True:
                sensor = 0.7
            l = np.log(sensor / (1 - sensor))
            l_past = np.log(objects[key]['Prob'] / (1 - objects[key]['Prob']))
            L = l + l_past
            if np.abs(L) > 5:
                L = 5 * np.sign(L)
            P = 1 - 1 / (1 + np.exp(L))
            # print(sensor)
            # print('Hey')
            # print(objects[key]['Prob'])
            print(P,L,l,l_past)
            r = 15 * P
            objects[key]['Prob'] = P
            # print(objects[key])
            #print(objects[key]['Prob'])
            objects[key]['Update'] = False
            draw.ellipse((objects[key]['X'] - r, objects[key]['Y'] - r, objects[key]['X'] + r, objects[key]['Y'] + r),
                         fill=(255, 0, 0, 255))

def change_coordinates(box):
    width = box[2]-box[0]
    height = box[3]-box[1]
    x = box[0]+width/2
    y = box[1]+height/2
    return x,y

# Function to change from cv2 to pil and resizing
def prepare_image(img):
    cv2_im = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    pil_im = Image.fromarray(cv2_im)
    img_resized = letter_box_image(pil_im, FLAGS.size, FLAGS.size, 128)
    img_resized = img_resized.astype(np.float32)
    return img_resized,pil_im

# Load the classes file and the graph
classes = load_coco_names(FLAGS.class_names)
frozenGraph = load_graph(FLAGS.frozen_model)
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
# While you get something from the camera
f = open('data.txt','w')
f.write('X0,Y0,X1,Y1,Indice\n')
count = 0

first_time = False
consecutive = 0
objects = dict()
while cv2.getWindowProperty('CSI Camera', 0) >= 0:
    ret_val, img = cap.read();
    # Prepare the image
    num_rows, num_cols = img.shape[:2]
    rotation_matrix = cv2.getRotationMatrix2D((num_cols / 2, num_rows / 2), 180, 1)
    img = cv2.warpAffine(img, rotation_matrix, (num_cols, num_rows))
    img_resized, pil_im = prepare_image(img)
    # Run the network
    detected_boxes = tf_sess.run(boxes, feed_dict={inputs: [img_resized]})
    # Show the detected image
    draw_and_show(detected_boxes,pil_im, count,consecutive)
    count = count + 1
    # Check if the window should be closed
    keyCode = cv2.waitKey(30) & 0xff
    # Stop the program on the ESC key
    if keyCode == 27:
        break
        cap.release()
        cv2.destroyAllWindows()
# Close the tf Session
tf_sess.close()
