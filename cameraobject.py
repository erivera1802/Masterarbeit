output_names = ['Logits/Softmax']
input_names = ['input_1']
import time
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input, decode_predictions
import numpy as np
import cv2
from PIL import Image
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

def show_camera():
	# To flip the image, modify the flip_method parameter (0 and 2 are the most common)
	print(gstreamer_pipeline(flip_method=0))
	cap = cv2.VideoCapture(gstreamer_pipeline(flip_method=0), cv2.CAP_GSTREAMER)
	if cap.isOpened():
		window_handle = cv2.namedWindow('CSI Camera', cv2.WINDOW_AUTOSIZE)
	# Window
	while cv2.getWindowProperty('CSI Camera',0) >= 0:
		ret_val, img = cap.read();
		
		
		destRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
		pil_im = Image.fromarray(destRGB)
		img2=pil_im.resize((224,224))
		x = image.img_to_array(img2)
		x = np.expand_dims(x, axis=0)
		x = preprocess_input(x)
		feed_dict = {input_tensor_name: x}
		preds = tf_sess.run(output_tensor, feed_dict)
		#print('Predicted:', decode_predictions(preds, top=3)[0])
		predict=decode_predictions(preds, top=1)[0]
		predict=predict[0][1:3]
		#print(predict)
		cv2.putText(img,str(predict),(10,700),cv2.FONT_HERSHEY_SIMPLEX,2,(255,255,255),2,cv2.LINE_AA)
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
trt_graph = get_frozen_graph('./model/trt_graph.pb')

# Create session and load graph
tf_config = tf.ConfigProto()
tf_config.gpu_options.allow_growth = True
tf_sess = tf.Session(config=tf_config)
tf.import_graph_def(trt_graph, name='')


# Get graph input size
for node in trt_graph.node:
    if 'input_' in node.name:
        size = node.attr['shape'].shape
        image_size = [size.dim[i].size for i in range(1, 4)]
        break
print("image_size: {}".format(image_size))


# input and output tensor names.
input_tensor_name = input_names[0] + ":0"
output_tensor_name = output_names[0] + ":0"

print("input_tensor_name: {}\noutput_tensor_name: {}".format(
    input_tensor_name, output_tensor_name))

output_tensor = tf_sess.graph.get_tensor_by_name(output_tensor_name)



# Optional image to test model prediction.
img_path = './elefant.jpg'

img = image.load_img(img_path, target_size=image_size[:2])
x = image.img_to_array(img)
x = np.expand_dims(x, axis=0)
x = preprocess_input(x)



# decode the results into a list of tuples (class, description, probability)
# (one such list for each sample in the batch)
#print('Predicted:', decode_predictions(preds, top=3)[0])
show_camera()
