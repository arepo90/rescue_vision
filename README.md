> Forked project, original README:

# Real-Time Object Detection with YOLO and OpenCV

This project performs real-time hazmat detection using a YOLO model (Darknet format) with OpenCV's DNN module. It detects hazmats from a live camera feed, draws bounding boxes and centroids, and displays normalized coordinates and FPS.

## Features

- Real-time inference using OpenCV
- Bounding box drawing with class label and confidence
- Centroid calculation and normalized coordinate display
- FPS counter overlay
- Optional OpenVINO backend for Intel-based optimization

## Requirements

- Python 3.6+
- OpenCV (with DNN support)
- NumPy

You can install the dependencies with:

```bash
pip install opencv-python numpy
```

## Usage

Make sure to set the correct `CAMERA_INDEX` in the script depending on your system (usually 0 or 2).

## Output

- Bounding boxes with class names and confidence scores
- Centroids marked with dots and labeled with pixel and normalized coordinates
- Real-time FPS indicator
- Press `q` to exit

## Example Output

```
Object: "flammable-solid: 0.94"
Centroid: (250, 310) → [0.40, 0.52]
FPS: 27.6
```

## Optional: OpenVINO Acceleration

Improve inference speed by enabling OpenVINO:

### TODO

- [ ] Install OpenVINO Toolkit from [Intel's official website](https://www.intel.com/content/www/us/en/developer/tools/openvino-toolkit/download.html)
- [ ] Run the OpenVINO environment setup script:
  ```bash
  source /opt/intel/openvino/bin/setupvars.sh
  ```
- [ ] Uncomment the following lines in the script:
  ```python
  # model.setPreferableBackend(cv2.dnn.DNN_BACKEND_INFERENCE_ENGINE)
  # model.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
  ```
- [ ] Comment out the OpenCV backend lines if OpenVINO is active:
  ```python
  # model.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
  # model.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
  ```

## Notes

- Centroid coordinates are calculated and normalized with respect to the image size.
