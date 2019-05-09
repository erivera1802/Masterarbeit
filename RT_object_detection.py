import tensorflow as tf
import cv2
import numpy as np
def get_frozen_graph(graph_file):
    """Read Frozen Graph file from disk."""
    with tf.gfile.FastGFile(graph_file, "rb") as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
    return graph_def

def gstreamer_pipeline (capture_width=1280, capture_height=720, display_width=1280, display_height=720, framerate=60, flip_method=0) :
	return ('nvarguscamerasrc ! '
	'video/x-raw(memory:NVMM), '
	'width=(int)%d, height=(int)%d, '
	'format=(string)NV12, framerate=(fraction)%d/1 ! '
	'nvvidconv flip-method=%d ! '
	'video/x-raw, width=(int)%d, height=(int)%d, format=(string)BGRx ! '
	'videoconvert ! '
	'video/x-raw, format=(string)BGR ! appsink' % (capture_width,capture_height,framerate,flip_method,display_width,display_height))
def non_max_suppression(boxes, probs=None, nms_threshold=0.3):
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

    # compute the area of the bounding boxes and grab the indexes to sort
    # (in the case that no probabilities are provided, simply sort on the
    # bottom-left y-coordinate)
    area = (x2 - x1 + 1) * (y2 - y1 + 1)
    idxs = y2

    # if probabilities are provided, sort on them instead
    if probs is not None:
        idxs = probs

    # sort the indexes
    idxs = np.argsort(idxs)

    # keep looping while some indexes still remain in the indexes list
    while len(idxs) > 0:
        # grab the last index in the indexes list and add the index value
        # to the list of picked indexes
        last = len(idxs) - 1
        i = idxs[last]
        pick.append(i)

        # find the largest (x, y) coordinates for the start of the bounding
        # box and the smallest (x, y) coordinates for the end of the bounding
        # box
        xx1 = np.maximum(x1[i], x1[idxs[:last]])
        yy1 = np.maximum(y1[i], y1[idxs[:last]])
        xx2 = np.minimum(x2[i], x2[idxs[:last]])
        yy2 = np.minimum(y2[i], y2[idxs[:last]])

        # compute the width and height of the bounding box
        w = np.maximum(0, xx2 - xx1 + 1)
        h = np.maximum(0, yy2 - yy1 + 1)

        # compute the ratio of overlap
        overlap = (w * h) / area[idxs[:last]]

        # delete all indexes from the index list that have overlap greater
        # than the provided overlap threshold
        idxs = np.delete(idxs, np.concatenate(([last],
                                               np.where(overlap > nms_threshold)[0])))
    # return only the bounding boxes indexes
    return pick


def show_camera():
	
	# Window
	while cv2.getWindowProperty('CSI Camera',0) >= 0:
		ret_val, img = cap.read();
		
		image = cv2.resize(img, (300, 300))
		scores, boxes, classes, num_detections = tf_sess.run([tf_scores, tf_boxes, 			tf_classes, tf_num_detections], feed_dict={tf_input: image[None, ...]})
		boxes = boxes[0]  # index by 0 to remove batch dimension
		scores = scores[0]
		classes = classes[0]
		num_detections = int(num_detections[0])
		boxes_pixels = []
		for i in range(num_detections):
		    # scale box to image coordinates
		    box = boxes[i] * np.array([720,1280, 720, 1280])
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
		    text="{}:{:.2f}".format(int(classes[i]),scores[i])
		    img = cv2.putText(img,text,(box[1],box[0]+70),cv2.FONT_HERSHEY_SIMPLEX,1,(255,255,255),2,cv2.LINE_AA)
		#print(str(classes))
		cv2.imshow('CSI Camera',img)
		# This also acts as
		keyCode = cv2.waitKey(30) & 0xff
		# Stop the program on the ESC key
		if keyCode == 27:
			break
			cap.release()
			cv2.destroyAllWindows()
			
	else:
		print('Unable to open camera')

pb_fname = "./model/trt_graph_object.pb"
trt_graph = get_frozen_graph(pb_fname)

input_names = ['image_tensor']
# To flip the image, modify the flip_method parameter (0 and 2 are the most common)
print(gstreamer_pipeline(flip_method=0))
cap = cv2.VideoCapture(gstreamer_pipeline(flip_method=0), cv2.CAP_GSTREAMER)
if cap.isOpened():
	window_handle = cv2.namedWindow('CSI Camera', cv2.WINDOW_AUTOSIZE)
# Create session and load graph
tf_config = tf.ConfigProto()
tf_config.gpu_options.allow_growth = True
tf_sess = tf.Session(config=tf_config)
tf.import_graph_def(trt_graph, name='')

tf_input = tf_sess.graph.get_tensor_by_name(input_names[0] + ':0')
tf_scores = tf_sess.graph.get_tensor_by_name('detection_scores:0')
tf_boxes = tf_sess.graph.get_tensor_by_name('detection_boxes:0')
tf_classes = tf_sess.graph.get_tensor_by_name('detection_classes:0')
tf_num_detections = tf_sess.graph.get_tensor_by_name('num_detections:0')


# decode the results into a list of tuples (class, description, probability)
# (one such list for each sample in the batch)
#print('Predicted:', decode_predictions(preds, top=3)[0])
show_camera()
tf_sess.close()
