#!/bin/bash


if [ -z "$1" ]; then
    echo "Using default path to OpenVINO model downloader"

    downloader_path="/opt/intel/openvino/deployment_tools/tools/model_downloader"

else
    echo "Looking for the model downloader at " "$1"

    downloader_path="$1"
fi


# Install the model downloader dependencies
python3 -mpip install --user -r "$downloader_path/requirements.in"


downloader_path="$downloader_path/downloader.py"

# Download the face detection model
python $downloader_path --name face-detection-retail-0005 --precisions FP32,FP16,FP32-INT8 --output_dir models

# Download the facial landmark detection model
python $downloader_path --name facial-landmarks-35-adas-0002 --precisions FP32,FP16,FP32-INT8 --output_dir models

# Download the head pose estimation model
python $downloader_path --name head-pose-estimation-adas-0001 --precisions FP32,FP16,FP32-INT8 --output_dir models

# Download the gaze estimation model
python $downloader_path --name gaze-estimation-adas-0002 --precisions FP32,FP16,FP32-INT8 --output_dir models


