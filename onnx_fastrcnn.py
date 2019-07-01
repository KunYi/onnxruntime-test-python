''' 
 to [onnx/mode](https://github.com/onnx/models/tree/master/faster_rcnn)
  donwload onnx file, current the example testing the below version
  
  [Faster R-CNN R-50-FPN -- opset v10](https://onnxzoo.blob.core.windows.net/models/opset_10/faster_rcnn/faster_rcnn_R_50_FPN_1x.tar.gz)

  wget https://onnxzoo.blob.core.windows.net/models/opset_10/faster_rcnn/faster_rcnn_R_50_FPN_1x.tar.gz 
  tar zxvf faster_rcnn_R_50_FPN_1x.tar.gz


'''
import numpy as np
import onnxruntime as rt
import cv2
from PIL import Image,ImageDraw
import matplotlib.pyplot as plt
import matplotlib.patches as patches


print('\nwaiting a monent for loading onnx file\n')
sess = rt.InferenceSession("faster_rcnn_R_50_FPN_1x.onnx")
input_name = sess.get_inputs()[0].name
print('\ncomplete load onnx\n')

def preprocess(image):
    # Resize
    ratio = 800.0 / min(image.size[0], image.size[1])
    image = image.resize((int(ratio * image.size[0]), int(ratio * image.size[1])), Image.BILINEAR)

    # Convert to BGR
    image = np.array(image)[:, :, [2, 1, 0]].astype('float32')

    # HWC -> CHW
    image = np.transpose(image, [2, 0, 1])

    # Normalize
    mean_vec = np.array([102.9801, 115.9465, 122.7717])
    for i in range(image.shape[0]):
        image[i, :, :] = image[i, :, :] - mean_vec[i]

    # Pad to be divisible of 32
    import math
    padded_h = int(math.ceil(image.shape[1] / 32) * 32)
    padded_w = int(math.ceil(image.shape[2] / 32) * 32)

    padded_image = np.zeros((3, padded_h, padded_w), dtype=np.float32)
    padded_image[:, :image.shape[1], :image.shape[2]] = image
    image = padded_image

    return image

img = Image.open('test.jpg')
X = preprocess(img)

out = sess.run(None, {input_name: X.astype(np.float32)})
boxes = out[0][:]
labels = out[1][:]
scores = out[2][:]
classes = [
        "__background",
        "person",
        "bicycle",
        "car",
        "motorcycle",
        "airplane",
        "bus",
        "train",
        "truck",
        "boat",
        "traffic light",
        "fire hydrant",
        "stop sign",
        "parking meter",
        "bench",
        "bird",
        "cat",
        "dog",
        "horse",
        "sheep",
        "cow",
        "elephant",
        "bear",
        "zebra",
        "giraffe",
        "backpack",
        "umbrella",
        "handbag",
        "tie",
        "suitcase",
        "frisbee",
        "skis",
        "snowboard",
        "sports ball",
        "kite",
        "baseball bat",
        "baseball glove",
        "skateboard",
        "surfboard",
        "tennis racket",
        "bottle",
        "wine glass",
        "cup",
        "fork",
        "knife",
        "spoon",
        "bowl",
        "banana",
        "apple",
        "sandwich",
        "orange",
        "broccoli",
        "carrot",
        "hot dog",
        "pizza",
        "donut",
        "cake",
        "chair",
        "couch",
        "potted plant",
        "bed",
        "dining table",
        "toilet",
        "tv",
        "laptop",
        "mouse",
        "remote",
        "keyboard",
        "cell phone",
        "microwave",
        "oven",
        "toaster",
        "sink",
        "refrigerator",
        "book",
        "clock",
        "vase",
        "scissors",
        "teddy bear",
        "hair drier",
        "toothbrush",
]

def display_objdetect_image(image, boxes, labels, scores, score_threshold=0.7):
    # Resize boxes
    ratio = 800.0 / min(image.size[0], image.size[1])
    boxes /= ratio

    _, ax = plt.subplots(1, figsize=(12,9))
    image = np.array(image)
    ax.imshow(image)

    # Showing boxes with score > 0.7
    for box, label, score in zip(boxes, labels, scores):
        if label == 1 and score > score_threshold:
            rect = patches.Rectangle((box[0], box[1]), box[2] - box[0], box[3] - box[1], linewidth=1, edgecolor='b', facecolor='none')
            ax.annotate(classes[label] + ':' + str(np.round(score, 2)), (box[0], box[1]), color='w', fontsize=12)
            ax.add_patch(rect)
    plt.show()

display_objdetect_image(img, boxes, labels, scores)

print("end program")
