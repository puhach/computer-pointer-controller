# Computer Pointer Controller

Human vision performs a variety of tasks to interpret the surrounding environment. Many of them have been researched and automated by deep learning. This project combines several such models from the Intel Distribution of OpenVINO Toolkit to control a mouse pointer using eye gaze. The first step is to identify faces and extract a face from an input video stream captured from a webcam or a video file. Then we extract facial landmarks and find the orientation of the face by means of a head pose estimation model. Knowing the head pose and facial landmarks, we can find the orientation of the eye gaze using a gaze estimation model. Finally, the mouse pointer is moved in the direction of the eye gaze. 


## Project Set Up and Installation

### Install the Intel Distribution of OpenVINO Toolkit

The project requires Intel OpenVINO 2020.1 or newer. Older versions should work too, but it's not guaranteed.
Refer to [this](https://docs.openvinotoolkit.org/2020.1/_docs_install_guides_installing_openvino_linux.html#install-openvino) manual for a step-by-step installation guide.


### Install Python libraries

Creating an isolated environment for the project is recommended. It can be done by the following command:
```
conda create -n computer-pointer-controller python=3.6
``` 

Now activate the environment:
```
conda activate computer-pointer-controller
```

To install necessary Python packages run:
```
pip install -r requirements.txt
```

### Download pre-trained models from OpenVINO Model Zoo

The project requires the following models:
* face-detection-retail-0005
* facial-landmarks-35-adas-0002
* head-pose-estimation-adas-0001
* gaze-estimation-adas-0002

Model files are expected to reside in <project-folder>/models.

For convenience there is a download_models.sh script in the project root folder which automatically installs the OpenVINO Model Downloader requirements and downloads all the necessary models. Assuming that OpenVINO is installed to the default location, simply run:
```
./download_models.sh
```

In case OpenVINO is installed to a different path, the script should be provided with a Model Downloader directory as an argument:
```
./download_models.sh /opt/intel/openvino/deployment_tools/tools/model_downloader
```

Alternatively, these models can be downloaded via Model Downloader (<OPENVINO_INSTALL_DIR>/deployment_tools/open_model_zoo/tools/downloader) one-by-one:
```
./downloader.py --name face-detection-retail-0005 --precisions FP32,FP16,FP32-INT8 --output_dir <project-folder>/models
./downloader.py --name facial-landmarks-35-adas-0002 --precisions FP32,FP16,FP32-INT8 --output_dir <project-folder>/models
./downloader.py --name head-pose-estimation-adas-0001 --precisions FP32,FP16,FP32-INT8 --output_dir <project-folder>/models
./downloader.py --name gaze-estimation-adas-0002 --precisions FP32,FP16,FP32-INT8 --output_dir <project-folder>/models
```

Consult [this](https://docs.openvinotoolkit.org/2020.1/_tools_downloader_README.html) page for additional details about OpenVINO Model Downloader usage.

### Project directory structure

The project tree should finally look like this:
```
├── bin
│   └── demo.mp4
├── download_models.sh
├── models
│   └── intel
│       ├── face-detection-retail-0005
│       ├── facial-landmarks-35-adas-0002
│       ├── gaze-estimation-adas-0002
│       └── head-pose-estimation-adas-0001
├── README.md
├── requirements.txt
└── src
    ├── face_detection.py
    ├── facial_landmarks_detection.py
    ├── gaze_estimation.py
    ├── generic_model.py
    ├── head_pose_estimation.py
    ├── helpers.py
    ├── input_feeder.py
    ├── main.py
    └── mouse_controller.py
```    



## Demo

Before launching the app make sure to activate the project's virtual environment (see "Project Set Up and Installation" section for guidelines on creating the virtual environment). Also you need to have OpenVINO initialized. Assuming the default installation path, it can be done by this command:
```
source /opt/intel/openvino/bin/setupvars.sh -pyver 3.6
```

The main script is located in the _src_ folder. In order to run the application you need to provide the input file:
```
python main.py --input ../bin/demo.mp4
```

Or you can use a webcam stream:
```
python main.py --input cam
```

Other command line arguments are optional. See the "Documentation" section for details.

