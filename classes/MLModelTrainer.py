# Joseph Patambag
# September 12, 2024


import cv2 as cv
import numpy as np
import os
from sklearn.model_selection import train_test_split as trainer
from keras.utils import to_categorical as to_cat
from keras.models import Sequential as seq
from keras.layers import Conv2D as conv, MaxPooling2D as mxp, Flatten as flt, Dense as dens
import config.constants as const

class MLModelTrainer:
    """
    This class will be responsible for training the assets captured.
    """
    # variable to hold image size
    img_size = 0

    # data set source directory
    source_dir = ""

    def __init__(self, source, callback):
        """
        Default class initializer.

        Parameter
        ---------
        source: String
            The source directory of the data set.
        callback: Method    
            Call back function.
        
        Return
        ------
        None
        """
        # assign the source directory
        self.source_dir = source
        self.img_size = const.IMAGE_SIZE
        self.callback = callback
    
    def configure(self):
        """
        Method to configure the machine learning function.

        Paramters
        ---------
        None

        Return
        ------
        Tuple (snapshots, tags)
        Snapshots:
            Array of images captures.
        Tags:
            Array of labels generatef from captured assets.
        """
    
        # variable to hold array of snapshots
        snapshots = []

        # variable to hold the snapshot tags
        tags = []

        # iterate list of directory inside the location
        for tag, gesture in enumerate(os.listdir(self.source_dir)):
            captured_dir = os.path.join(self.source_dir, gesture)
            
            #check if the path is directory
            if os.path.isdir(captured_dir):
                 # iterate each name in the captured images
                for name in (os.listdir(captured_dir)):
                    image_path = os.path.join(captured_dir, name)
                    image = cv.imread(image_path, cv.IMREAD_GRAYSCALE)
                    if image is not None:
                        image = cv.resize(image, (self.img_size, self.img_size))
                        snapshots.append(image)
                    tags.append(tag)
            else:
                print("Skipping files that are not directory.")        
           
        snapshots = np.array(snapshots)
        tags = np.array(tags)

        return snapshots, tags

    def split_train(self):
        """
        Method to split and train the assets.

        Parameter
        ---------
        None

        Return
        ------
        Tuple (X_train, X_test, y_train, y_test)
        X_train, X_test, y_train, y_test:
            Object generate.
        """
        # tuples to get the raw images and tags

        snapshots, tags = self.configure()
        snapshots = snapshots.reshape(snapshots.shape[0], 
                                      self.img_size, 
                                      self.img_size, 
                                      1)
        snapshots = snapshots / 255.0
        tags = to_cat(tags)

        if len(tags) > len(snapshots):
            diff = len(tags) - len(snapshots)
            new_tags = tags[:-diff]
            X_train, X_test, y_train, y_test = trainer(snapshots, 
                                                   new_tags, 
                                                   test_size=const.TRAINER_TEST_SIZE, 
                                                   random_state=const.TRAINER_TEST_RANDOM_STATE)
            return X_train, X_test, y_train, y_test
        else:
            X_train, X_test, y_train, y_test = trainer(snapshots, 
                                                   tags, 
                                                   test_size=const.TRAINER_TEST_SIZE, 
                                                   random_state=const.TRAINER_TEST_RANDOM_STATE)
            return X_train, X_test, y_train, y_test 
         
    
    def train_model(self):
        """
        Train the dataset.

        Parameter
        ---------
        None

        Return
        ------
        None
        """
        
        # Define the CNN model
        model = seq([
            conv(32, (3, 3), activation=const.RELU_ACTIVATION, 
                 input_shape=(self.img_size, 
                              self.img_size, 
                              1)),
            mxp((2, 2)),
            conv(64, (3, 3), 
                 activation=const.RELU_ACTIVATION),
            mxp((2, 2)),
            flt(),
            dens(128, 
                 activation=const.RELU_ACTIVATION),
            dens(len(os.listdir(self.source_dir)), 
                 activation=const.SOFTMAX)  # Number of gestures
        ])

        model.compile(optimizer=const.OPTIMIZER, 
                      loss=const.LOSS, 
                      metrics=[const.ACCURACY])
        model.summary()

        trainer = self.split_train()

       
        # Train the model
        model.fit(trainer[0], 
                  trainer[2], 
                  epochs=10, 
                  validation_data=(trainer[1], trainer[3]))

        # Save the model
        model.save(const.ML_MODEL)
        self.callback('exit')