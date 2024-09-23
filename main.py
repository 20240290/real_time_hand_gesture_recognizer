# Joseph Patambag
# September 12, 2024

#import pkg_resources
import sys
import subprocess
import shutil
from pathlib import Path
import config.constants as const


def check_dependencies():
    """
    Method to check if the necessary dependencies are installed and let the user know that it needs to be installed.

    Parameter
    ---------
    None

    Return
    Bool
        Check if all dependencies are installed/
    """
    missing_dependencies = []

    try:
        import cv2
        print("cv2 library is installed")
    except ImportError:
       missing_dependencies.append('cv2')

    try:
        import numpy
        print("numpy library is installed")

    except ImportError:
       missing_dependencies.append('numpy')

    try:
        import tensorflow
        print("tensorflow library is installed")

    except ImportError:
        missing_dependencies.append('tensorflow')

    try:
        import mediapipe
        print("mediapipe library is installed")

    except ImportError:
        missing_dependencies.append('mediapipe')

    if len(missing_dependencies) == 0:
        return True
    else:
        for item in missing_dependencies:
            install(item)
        else:
            return True    
    
def install(package):
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", package])
        print(f"Successfully installed {package}")
    except subprocess.CalledProcessError as e:
        print(f"Failed to install {package}: {e}")

def clearDataSet(directory):
    """
    Use this to clear the assets in the data set folder.

    Parameter
    ---------
    directory
        Location of the folder.

    Return
    ------
    None    
    """
    #Iteratate each directory and remove the images.
    for item in Path(directory).iterdir():
            if item.is_dir():
                shutil.rmtree(item)

    #call menu initialization            
    mainApp()            

def completion(msg):
    """
    Callback method to display the 
    """
    if msg == "exit":
       #prompt = menu()
       mainApp()     

def mainApp():
    """
    Entry point of the application.

    Parameter
    ---------
    None
    
    Return
    -------
    None
    """
    #class import
    from classes.RthgrMod import MLModelTrainer as trainer
    from classes.RthgrMod import MLDatasetGenerator as generator
    from classes.RthgrMod import MLHandRecognizer as gesture

    #initialize all classes
    
    ml_trainer = trainer(const.DATA_DIRECTORY, completion)
    ml_generator = generator(completion)
    ml_recognizer = gesture(const.ML_MODEL, completion)

    msg = "\nWelcome to Real Time Hand Gesture Recognizer Project. \n"
    msg += "\nPlease choose from the following options: \n"
    msg += "\n1. Capture data set. \n2. Train the data set. \n3. Recognize the gesture."
    msg += "\n4. Clear dataset.\n5. End the program.\n\n"
    
    try:
        prompt = input(msg)
        if prompt == "1":
            print("Initialize data set to train.")
            ml_generator.capture_gesture()
        elif prompt == "2":
             print("Training your data set.")
             ml_trainer.train_model()
        elif prompt == "3":
            ml_recognizer.start_stream()
        elif prompt == "4":
            clearDataSet(const.VIDEO_DATA_DIRECTORY)  
        else:
            exit()
    except KeyboardInterrupt:
        exit();    
         

if __name__ == '__main__':
#Execute when the module is not initialized from an import statement
   if check_dependencies():
        mainApp()        
# import cv2
# import numpy as np
# import tensorflow as tf
# import mediapipe as mp

