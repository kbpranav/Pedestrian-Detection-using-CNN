# Real-time Pedestrian Detection using CNN for Autonomous Vehicles

The design of a real-time pedestrian detection system using CNN for autonomous vehicles is proposed and the system is designed from scratch without using any standard module/libraries available for object detection.

Convolution Neural Networks performs the combined task of feature extraction and classification, thus being considered as the most preferred algorithm for designing several pattern recognition systems.

The real-time video is captured from camera as individual frames of size 640×480 pixels. Each frame is then split into sub-frames of size 160×480 pixels with 50% overlap, resized to 80×160 pixels and fed to the proposed CNN model for detecting pedestrians.

The proposed model achieved recognition accuracies ranging between 96.73 – 100% based on dataset employed and also the position of the pedestrian.

Download the dataset from here : https://drive.google.com/drive/folders/1uxtnKJe6Wxvb59HSW1EgoQ-tC9oCWkZV?usp=sharing and store it in folder named "database", which should be in the same directory as classifier.py

Necessary packages to run:
1. Keras
2. Tensorflow
3. Opencv-Python
4. Numpy
5. Matplotlib
6. tkinter
7. Anaconda Spyder Environment for better experience(Optional)

Run: classifier.py
1. Click load Architecture
2. Click Train Model( If not trained before)
3. Click Load trained model
4. Click detect pedestrains 

NOTE: If running for the first time, train the model. Else, skip the training step.

For reference : https://ieeexplore.ieee.org/document/9161768

