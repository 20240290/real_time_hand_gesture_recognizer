# Joseph Patambag
# September 16, 2024

import cv2 as cv
import numpy as np
import tensorflow as tf
import mediapipe as mp
import sys
import config.constants as const

class MLHandRecognizer:
    """
    Class to handle the real time gesture recognition for the trained model.

    Parameter
    ----------
    None

    Return
    ------
    None
    """

    #class variables
    gestures = const.ML_GESTURE_MODEL
    gesture_names = const.ML_GESTURE_MODEL
    mp_hands = any
    hand_gesture = any
    mp_drawing = any
    model = any
    
    
    def __init__(self, source, callback):
        """
        Default class initializer.

        Parameter
        --------
        source: Path
            Path location of the model.
        callback: Method    
            Call back function.    

        Return
        ------
        None
        
        """
        self.model = tf.keras.models.load_model(source)
        self.mp_hands = mp.solutions.hands
        self.hand_gesture = self.mp_hands.Hands(static_image_mode=False, 
                                           max_num_hands=2, 
                                           min_detection_confidence=0.5)
        self.mp_drawing = mp.solutions.drawing_utils
        self.callback = callback

    # Function to preprocess frame
    def preprocess_frame(self, frame):
        gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        resized = cv.resize(gray, (const.IMAGE_SIZE, const.IMAGE_SIZE))  # Ensure image size is (64, 64)
        reshaped = resized.reshape(1, const.IMAGE_SIZE, const.IMAGE_SIZE, 1)  # Reshape to match (1, 64, 64, 1)
        return reshaped / 255.0  # Normalize pixel values to [0, 1]

    def draw_hand_landmarks(self, frame, landmarks):
        if landmarks:
            # Draw connections between nodes (landmarks)
            connections = self.mp_hands.HAND_CONNECTIONS
            for connection in connections:
                start_index = connection[0]
                end_index = connection[1]
                
                # Extract start and end landmarks
                start_landmark = landmarks.landmark[start_index]
                end_landmark = landmarks.landmark[end_index]
                
                # Calculate pixel coordinates
                start_point = (int(start_landmark.x * frame.shape[1]), 
                               int(start_landmark.y * frame.shape[0]))
                end_point = (int(end_landmark.x * frame.shape[1]), 
                             int(end_landmark.y * frame.shape[0]))
                
                # Draw line
                cv.line(frame, start_point, end_point, (255, 255, 255), 2)  # White color, thickness 2
            
            # Draw landmarks (nodes) as circles
            for landmark in landmarks.landmark:
                x = int(landmark.x * frame.shape[1])
                y = int(landmark.y * frame.shape[0])
                cv.circle(frame, (x, y), 5, (0, 0, 255), -1)  # Red color, filled circle     

    def start_stream(self):
        # Start the webcam
        cap = cv.VideoCapture(0)
    
        if not cap.isOpened():
            print("Error: Could not open camera.")
            exit()

        try:
            print(f"Stream started.... \n")
            key = input(f"Press 's' to start, 'q' to quit.\n")
            
            while True:
                if key == 's':
                    ret, frame = cap.read()

                    if not ret:
                        print("Failed to capture frame from webcam. Exiting...")
                        break
                    # Hand pose estimation
                    frame_rgb = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
                    results = self.hand_gesture.process(frame_rgb)

                    if results.multi_hand_landmarks:
                        for hand_landmarks in results.multi_hand_landmarks:
                            self.draw_hand_landmarks(frame, hand_landmarks)

                            # Extract bounding box of the hand region
                            x_min = int(min([lm.x for lm in hand_landmarks.landmark]) * frame.shape[1])
                            x_max = int(max([lm.x for lm in hand_landmarks.landmark]) * frame.shape[1])
                            y_min = int(min([lm.y for lm in hand_landmarks.landmark]) * frame.shape[0])
                            y_max = int(max([lm.y for lm in hand_landmarks.landmark]) * frame.shape[0])

                            # Ensure bounding box coordinates are within frame dimensions
                            x_min = max(0, x_min)
                            y_min = max(0, y_min)
                            x_max = min(frame.shape[1], x_max)
                            y_max = min(frame.shape[0], y_max)

                            # Crop hand region
                            hand_crop = frame[y_min:y_max, x_min:x_max]

                            if hand_crop.size > 0:
                                try:
                                    preprocessed_frame = self.preprocess_frame(hand_crop)

                                    # Predict the gesture
                                    predictions = self.model.predict(preprocessed_frame)
                                    gesture = self.gesture_names[np.argmax(predictions)]

                                    # Display the gesture
                                    cv.putText(frame, f'Gesture: {gesture}', (x_min, y_min - 10), cv.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2, cv.LINE_AA)
                                except Exception as e:
                                    print(f"Error during prediction: {e}")

                    # Display the frame
                    cv.imshow('Hand Gesture Recognition', frame)

                    if cv.waitKey(1) & 0xFF == ord('q'): 
                        self.callback('exit') 
                        break
                else:
                    cap.release()
                    cv.destroyAllWindows()  
                    self.callback('exit') 
                    break 

        except KeyboardInterrupt:
            self.callback('exit')
            sys.exit()