''' 
 download pretrained model
 from http://download.tensorflow.org/models/object_detection/ssdlite_mobilenet_v2_coco_2018_05_09.tar.gz

 use tf2onnx/opset v10 to convert onnx model
 ref. https://github.com/onnx/tensorflow-onnx
'''
import numpy as np
import onnxruntime as rt
import cv2
from PIL import Image,ImageDraw

img = Image.open('test.jpg')
# read image size
W = img.size[0]
H = img.size[1]
# print([W, H])
X = img.resize((300, 300)) #for ssdlite mobilenet v2 coco

model_file = "models/ssdlite_mobilenet_v2.onnx"
print('\nwaiting a monent for loading onnx file\n')
sess = rt.InferenceSession(model_file)
input_name = sess.get_inputs()[0].name
print('\ncomplete load onnx\n')

X = np.expand_dims(X, axis=0)

out = sess.run(None, {input_name: X.astype(np.uint8)})
boxes = out[0][0][:] * np.array([H, W, H, W]) # for original
scores = out[1][0][:]
classes = out[2][0][:]
count = out[3][:]
#print(boxes)
#print(scores)
#print(classes)
#print(count)

draw = ImageDraw.Draw(img)

for i in range (0, int(count)) :
    if int(classes[i]) == 1 and scores[i] > 0.5:
       y0, x0, y1, x1 = boxes[i]
       draw.rectangle([(int(x0), int(y0)), (int(x1), int(y1))], None, (0, 0, 255))

img.save("result.png")

cv2.namedWindow('Result Image', cv2.WINDOW_NORMAL)
img = cv2.imread('result.png')
cv2.imshow('Result Image', img)
cv2.waitKey(0)
cv2.destroyAllWindows()

print("end program")

