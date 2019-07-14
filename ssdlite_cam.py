''' 
 download pretrained model
 from http://download.tensorflow.org/models/object_detection/ssdlite_mobilenet_v2_coco_2018_05_09.tar.gz

 use tf2onnx/opset v10 to convert onnx model
 ref. https://github.com/onnx/tensorflow-onnx
'''
from imutils.video import FileVideoStream
import numpy as np
import onnxruntime as rt
import cv2


def preprocess(img):
    img = cv2.resize(img, (300, 300), interpolation = cv2.INTER_LINEAR)
    img = np.array(img).astype('uint8')
    return np.expand_dims(img, axis=0)

def draw_obj_image(img, boxes, labels, scores, count, score_threshold=0.5):
    for box, label, score in zip(boxes, labels, scores):
        if count < 0:
           break
        count = count - 1
        if int(label) == 1 and score > score_threshold:
            img = cv2.rectangle(img, (int(box[1]), int(box[0])), (int(box[3]), int(box[2])), (255, 0, 0), 2)

    return img

model_file = "models/ssdlite_mobilenet_v2.onnx"
print('\nwaiting a monent for loading onnx file\n')
sess = rt.InferenceSession(model_file)
input_name = sess.get_inputs()[0].name
print('\ncomplete load onnx\n')

fvs = FileVideoStream('2.mp4').start()
#cam = cv2.VideoCapture('2.mp4')
count = 5
skipFrame = 5

while fvs.more():
    img = fvs.read()
    count += 1
    if count % skipFrame != 0:
        continue

    # read image size
    H, W = img.shape[:2]
    X = preprocess(img)

    out = sess.run(None, {input_name: X.astype(np.uint8)})
    boxes = out[0][0][:] * np.array([H, W, H, W]) # for original
    scores = out[1][0][:]
    classes = out[2][0][:]
    count = out[3][:]
    draw_obj_image(img, boxes, classes, scores, count)
    cv2.imshow('display', img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


print("end program")

