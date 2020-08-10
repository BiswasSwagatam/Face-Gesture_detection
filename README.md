# Face-Gesture_detection
OpenCV and Deep learning project to detect a hand gesture and consequently detecting the face of the user in a webcam video stream

Instructions - 
1. Use the collect_data_face.py file to collect training images of "user's" face
2 Similarly, use the collect_data_hand.py file to collect training images of the gesture you want to use
3. Inside the train.py module, follow #comment lines to train separate models for detecting hand and face simultaneously
4. Run the predict .py file to see the output

Additional info -
1. The model used for training is based upon VGG16(keras pretrained model) architecture
2. Try to collect as much data for training as you can in different conditions to make the model more robust and adaptive to the surroundings
3. Before running collect_data_hand/collect_data_face.py files make sure to create the folders necessary 
 
 Model in action - 
 https://youtu.be/iQtRLJRMDxs
