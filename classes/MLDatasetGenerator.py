# Joseph Patambag
# September 12, 2024

import cv2 as cv
import os
import sys
import numpy as np
import mediapipe as mp
import config.constants as const

class MLDatasetGenerator:
    """
    This class will enable the video camera capture from cv.

    Parameter
    ----------
    None

    Return
    ------
    None
    """

    #class variables
    gestures = const.ML_GESTURE_MODEL
    mp_hands = any
    hand_gesture = any
    mp_drawing = any

    def __init__(self, callback) -> None:
        """
        Default class initializer.

        Parameter
        --------
        callback: Method    
            Call back function.

        Return
        ------
        None
        
        """

        # Initialize MediaPipe hands module for hand detection and landmark estimation
        self.mp_hands = mp.solutions.hands
        self.hand_gesture = self.mp_hands.Hands(static_image_mode=False, 
                               max_num_hands=1, min_detection_confidence=0.5, 
                               min_tracking_confidence=0.5)
        self.mp_drawing = mp.solutions.drawing_utils
        self.callback = callback
        self.initialize_directories()
        
    def initialize_directories(self):
        """
        Method to check the diretories are existing in the dataset directory.
        """
        # Create directories if they don't exist
        if not os.path.exists(const.VIDEO_DATA_DIRECTORY):
            os.makedirs(const.VIDEO_NUM_CAPTURE)
        for gesture in self.gestures:
            gesture_dir = os.path.join(const.VIDEO_DATA_DIRECTORY, 
                                       gesture)
            if not os.path.exists(gesture_dir):
                os.makedirs(gesture_dir)
        #self.capture_gesture()            
    
    def capture_hand_gesture(self, gesture):
        """
        Start the camera to capture the dataset to be processed.

        Parameter
        ---------
        None
        
        Return
        ------
        None
        
        """
        cap = cv.VideoCapture(0)
        count = 0
        
        if not cap.isOpened():
            print("Error: Could not open camera.")
            exit()

        print(f"Stream started.... \n")
        key = input(f"Press 'g' to capture {gesture} gesture. Press 'ctl+c' to quit.\n")
        try:
             while True:
                ret, frame = cap.read()
                if not ret:
                    break

                # Flip the frame horizontally for a later selfie-view display
                frame = cv.flip(frame, 1)

                # Convert the image from BGR to RGB
                image_rgb = cv.cvtColor(frame, cv.COLOR_BGR2RGB)

                # To improve performance, optionally mark the image as not writeable to pass by reference
                image_rgb.flags.writeable = False

                # Perform hand detection and landmark estimation
                results = self.hand_gesture.process(image_rgb)

                # Check if hand(s) detected
                if results.multi_hand_landmarks:
                    for hand_landmarks in results.multi_hand_landmarks:
                        # Draw hand landmarks (skeleton) on the image
                        self.mp_drawing.draw_landmarks(frame, hand_landmarks, self.mp_hands.HAND_CONNECTIONS)

                        # Initialize min and max values for bounding box
                        x_min, y_min = frame.shape[1], frame.shape[0]
                        x_max, y_max = 0, 0

                        # Iterate through all landmarks to find bounding box
                        for landmark in hand_landmarks.landmark:
                            x, y = int(landmark.x * frame.shape[1]), int(landmark.y * frame.shape[0])
                            if x < x_min:
                                x_min = x
                            if x > x_max:
                                x_max = x
                            if y < y_min:
                                y_min = y
                            if y > y_max:
                                y_max = y


                        # Crop hand region based on detected bounding box
                        hand_crop = frame[y_min:y_max, x_min:x_max]


                        if hand_crop is None or hand_crop.size == 0:
                            print("Captured image is empty!")
                            break  
                        
                        # Resize cropped hand region
                        hand_resized = cv.resize(hand_crop, 
                                                (const.VIDEO_IMAGE_SIZE, const.VIDEO_IMAGE_SIZE))

                        # Save the resized hand image to the dataset directory
                        image_path = os.path.join(const.VIDEO_DATA_DIRECTORY, gesture, f'{count}.jpg')
                        cv.imwrite(image_path, hand_resized)
                        print(f"Captured snapshot {count} for {gesture} max {const.VIDEO_NUM_CAPTURE}")
                        count += 1

                        # Stop capturing after num_images
                        if count == const.VIDEO_NUM_CAPTURE:
                            break

                # Display the frame
                cv.imshow('Hand Gesture Capture', frame)

                # Wait for user input
                key = cv.waitKey(1) & 0xFF
                if key == ord('q'):
                    print(f"q data set")    
                    break
        except Exception as e:
            print(f"An error occurred: {e}")        
        except KeyboardInterrupt:  
            cap.release()
            cv.destroyAllWindows()    
            self.callback('exit')     
        finally:
            print(f"finally data set")  
            cap.release()
            cv.destroyAllWindows()

    def capture_gesture(self):
        """
        Start to captre the list of gesture added.

        Parameter
        --------
        None

        Return
        ------
        None
        """

        #iterate each gesture and capture the snapshot.
        for gesture in self.gestures:
            print("gesture is ", gesture)
            self.capture_hand_gesture(gesture)
        else:
            self.callback('exit')  
                