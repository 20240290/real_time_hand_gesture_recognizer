# Joseph Patambag
# September 12, 2024

#import pkg_resources
import config.constants as const
import sys
#load main file

"""

- Python 3.x
- OpenCV
- NumPy
- TensorFlow
- Keras
- Mediapipe

"""

def check_dependencies():
    #function to check if the dependencies are installed and let the user know that it needs to be installed
    
    # dependencies = ['cv2', '', '']
    
    # try:
    #     import cv2
    #     print("cv2 library is installed")
    # except ImportError:
    #     print("cv2 library is not installed")

    # try:
    #     import numpy
    #     print("numpy library is installed")

    # except ImportError:
    #     print("numpy library is not installed")

    pass

def initilizeApp():
    from classes.RthgrMod import MLModelTrainer as trainer
    from classes.RthgrMod import MLDatasetGenerator as generator
    from classes.RthgrMod import MLHandRecognizer as gesture

    #initialize all classes
    
    ml_trainer = trainer(const.DATA_DIRECTORY)
    ml_generator = generator()
    ml_recognizer = gesture(const.ML_MODEL)


    msg = "\nWelcome to Real Time Hand Gesture Recognizer. \n"
    msg += "\nPlease choose from the following options: \n"
    msg += "\n1. Capture data set. \n2. Train the data set. \n3. Recognize the gesture. \n4. End the program.\n\n"
    
    try:
        prompt = input(msg)
        if prompt == '1':
            print("Initialize data set to train.")
            ml_generator.capture_gesture()
        elif prompt == "2":
             print("Training your data set.")
             ml_trainer.train_model()
        elif prompt == "3":
            ml_recognizer.start_stream()
        else:
            exit()
    except KeyboardInterrupt:
        exit();    
         

if __name__ == '__main__':
    #Execute when the module is not initialized from an import statement
    #check_dependencies()
    initilizeApp()
# import cv2
# import numpy as np
# import tensorflow as tf
# import mediapipe as mp

