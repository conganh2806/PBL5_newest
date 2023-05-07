import cv2
import urllib.request
import numpy as np

import face_recognition
import math
from sklearn import neighbors

import os
import os.path
import pickle
from background_task import background
import requests
from student_management_app.models import Attendance, Students
	

def predict(X_img, knn_clf=None, model_path=None, distance_threshold=0.6):
    """
    Recognizes faces in given image using a trained KNN classifier

    :param knn_clf: (optional) a knn classifier object. if not specified, model_save_path must be specified.
    :param model_path: (optional) path to a pickled knn classifier. if not specified, model_save_path must be knn_clf.
    :param distance_threshold: (optional) distance threshold for face classification. the larger it is, the more chance
           of mis-classifying an unknown person as a known one.
    :return: a list of names and face locations for the recognized faces in the image: [(name, bounding box), ...].
        For faces of unrecognized persons, the name 'unknown' will be returned.
    """
    

    if knn_clf is None and model_path is None:
        raise Exception(
            "Must supply knn classifier either thourgh knn_clf or model_path")

    # Load a trained KNN model (if one was passed in)
    if knn_clf is None:
        with open(model_path, 'rb') as f:
            knn_clf = pickle.load(f)

    # Load image file and find face locations
    X_face_locations = face_recognition.face_locations(X_img)

    # If no faces are found in the image, return an empty result.
    if len(X_face_locations) == 0:
        return []

    # Find encodings for faces in the test iamge
    faces_encodings = face_recognition.face_encodings(
        X_img, known_face_locations=X_face_locations)

    # Use the KNN model to find the best matches for the test face
    closest_distances = knn_clf.kneighbors(faces_encodings, n_neighbors=1)
    are_matches = [closest_distances[0][i][0] <=
                   distance_threshold for i in range(len(X_face_locations))]
    
    

    # Predict classes and remove classifications that aren't within the threshold
    return [(pred, loc) if rec else ("unknown", loc) for pred, loc, rec in zip(knn_clf.predict(faces_encodings), X_face_locations, are_matches)]

@background(schedule=10)
def getFrame(requests):
    url = 'http://192.168.248.126/cam-hi.jpg'
    while True:
        img_resp = urllib.request.urlopen(url)
        imgnp = np.array(bytearray(img_resp.read()), dtype=np.uint8)
        frame = cv2.imdecode(imgnp, -1)
        print("OK")
        width = 640
        height = 480
        dim = (width, height)

        # resize image
        resized = cv2.resize(frame, dim, interpolation=cv2.INTER_AREA)

        rgb_frame = resized[:, :, ::-1]

        predictions = predict(
            rgb_frame, model_path="trained_knn_model.clf", distance_threshold=0.4)
        #Xu li diem danh 
        for name, (top, right, bottom, left) in predictions:
            print(name)
  
    
    
    
    #     for name, (top, right, bottom, left) in predictions:
    #         # Draw a box around the face
    #         cv2.rectangle(resized, (left, top),
    #                       (right, bottom), (0, 0, 255), 2)

    #         # Draw a label with a name below the face
    #         cv2.rectangle(resized, (left, bottom - 35),
    #                       (right, bottom), (0, 0, 255), cv2.FILLED)
    #         font = cv2.FONT_HERSHEY_DUPLEX
    #         cv2.putText(resized, name, (left + 6, bottom - 6),
    #                     font, 1.0, (255, 255, 255), 1)

    #     cv2.imshow('Camera', frame)

    #     key = cv2.waitKey(5)
    #     if key == ord('q'):
    #         break

    # cv2.destroyAllWindows()



class FaceDetect(object):
    def __init__(self):
        # change the IP address below according to the
        # IP shown in the Serial monitor of Arduino code
        self.url = 'http://192.168.1.96/cam-hi.jpg'
 
    
    def __del__(self):
        cv2.destroyAllWindows()
        
        
    def get_frame(self):
        
		# grab the frame from the threaded video stream
        img_resp = urllib.request.urlopen(self.url)
        imgnp = np.array(bytearray(img_resp.read()), dtype=np.uint8)
        frame = cv2.imdecode(imgnp, -1)
        #frame = cv2.flip(frame,1)

		# resize the frame to have a width of 600 pixels (while
		# maintaining the aspect ratio), and then grab the image
		# dimensions
        width = 640
        height = 480
        dim = (width, height)

		# # resize image
        resized = cv2.resize(frame, dim, interpolation=cv2.INTER_AREA)

        rgb_frame = resized[:, :, ::-1]

        predictions = predict( rgb_frame, model_path="trained_knn_model.clf", distance_threshold=0.4)
        
        for name, (top, right, bottom, left) in predictions:
            #Draw a box around the face
            cv2.rectangle(resized, (left, top),
                        (right, bottom), (0, 0, 255), 2)

            # Draw a label with a name below the face
            cv2.rectangle(resized, (left, bottom - 35),
                        (right, bottom), (0, 0, 255), cv2.FILLED)
            font = cv2.FONT_HERSHEY_DUPLEX
            cv2.putText(resized, name, (left + 6, bottom - 6),
                        font, 1.0, (255, 255, 255), 1)
            
            ser.write(name.encode())
            
            
        ret, jpeg = cv2.imencode('.jpg', resized)
            
        return jpeg.tobytes()  
        
        
  
  
  