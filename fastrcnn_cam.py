''' 
 to [onnx/mode](https://github.com/onnx/models/tree/master/faster_rcnn)
  donwload onnx file, current the example testing the below version
  
  [Faster R-CNN R-50-FPN -- opset v10](https://onnxzoo.blob.core.windows.net/models/opset_10/faster_rcnn/faster_rcnn_R_50_FPN_1x.tar.gz)

  wget https://onnxzoo.blob.core.windows.net/models/opset_10/faster_rcnn/faster_rcnn_R_50_FPN_1x.tar.gz 
  tar zxvf faster_rcnn_R_50_FPN_1x.tar.gz


'''
import numpy as np
import onnxruntime as rt
import warnings
import cv2

warnings.filterwarnings('ignore')

classes = [
    "__background", "person", "bicycle", "car", "motorcycle",
    "airplane", "bus", "train", "truck", "boat", "traffic light",
    "fire hydrant", "stop sign", "parking meter", "bench", "bird",
    "cat", "dog", "horse", "sheep", "cow", "elephant", "bear","zebra",
    "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
    "skis", "snowboard", "sports ball", "kite", "baseball bat",
    "baseball glove", "skateboard", "surfboard", "tennis racket",
    "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl",
    "banana", "apple", "sandwich", "orange", "broccoli", "carrot",
    "hot dog", "pizza", "donut", "cake", "chair", "couch", "potted plant",
    "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote",
    "keyboard", "cell phone", "microwave", "oven", "toaster", "sink",
    "refrigerator", "book", "clock", "vase", "scissors", "teddy bear",
    "hair drier", "toothbrush",
]

print('\nwaiting a monent for loading onnx file\n')
sess = rt.InferenceSession("onnx_models/faster_rcnn/faster_rcnn_R_50_FPN_1x.onnx")
input_name = sess.get_inputs()[0].name
print('\ncomplete load onnx\n')


def preprocess(img):
    H, W = img.shape[:2]
    ratio = 600.0 / min(W, H)
    #print((W, H))
    #print((int(ratio * W), int(ratio * H)))    
    img = cv2.resize(img, (int(W * ratio), int(H * ratio)), interpolation = cv2.INTER_LINEAR)
    img = np.array(img).astype('float32')
    #img.swapaxes(1,2).swapaxes(0,1)
    img = np.transpose(img, [2, 0, 1])
    #print(img[1])
    mean_vec = np.array([102.9801, 115.9465, 122.7717])

    for i in range(img.shape[0]):
        img[ i, :, :] = img[i, :, :] - mean_vec[i]

    import math
    padded_h = int(math.ceil(img.shape[1] / 32) * 32)
    padded_w = int(math.ceil(img.shape[2] / 32) * 32)
    #print((padded_h, padded_w))
    padded_img = np.zeros((3, padded_h, padded_w), dtype=np.float32)
    padded_img [:, :img.shape[1], :img.shape[2]] = img
    img = padded_img
    return img

def draw_obj_image(img, boxes, labels, scores, score_threshold=0.7):
    H, W = img.shape[:2]
    ratio = 600.0 / min(W, H)
    #print(ratio)
    boxes /= ratio

    for box, label, score in zip(boxes, labels, scores):
        if label == 1 and score > score_threshold:
            img = cv2.rectangle(img, (box[0], box[1]), (box[2], box[3]), (255, 0, 0), 2)
            #  ax.annotate(classes[label] + ':' + str(np.round(score, 2)), (box[0], box[1]), color='w', fontsize=12)

    return img

cam = cv2.VideoCapture('2.mp4')
count = 5
skipFrame = 5

while (1):
    ret_val, img = cam.read()
    count += 1
    if count % skipFrame != 0:
        continue

    if ret_val:
            X = preprocess(img)
    else:
        break
    boxes, labels, scores = sess.run(None, {input_name: X.astype(np.float32)})
    #print ((boxes))
    #print ((labels))
    #print ((scores))
    img = draw_obj_image(img, boxes, labels, scores, score_threshold=0.7)
    cv2.imshow('display', img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

print("end program")
