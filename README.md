## TinyYolo v2/SSDLite Mobilenet v2/Faster RCNN on ONNX Runtime/OpenVINO EP

all example code test pass on Docker container

---

### Preperation

to build our backend, [ONNX runtime/OpenVINO EP
to docker image](https://github.com/KunYi/Edge-Analytics-FaaS/tree/master/Azure-IoT-Edge/OnnxRuntime)

---
### Example

* onnx_tiny_yolov2.py for inference single picture on TINY YOLO v2 model
* onnx_fastrcnn.py for inference single picture on Faster RCNN model
* onnx_ssdmobilenetv2.py for inference single picture on SSD Lite Mobilenet V2 coco model
* fastrcnn_cam.py for video inference on Faster RCNN model
* ssdlite_cam.py for video inference on SSD Lite Mobilenet v2 Coco
