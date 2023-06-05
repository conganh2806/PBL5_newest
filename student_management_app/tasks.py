import datetime
import math
import os
import os.path
import pickle
import urllib.request
from timeit import default_timer as timer

import cv2
import face_recognition
import numpy as np
import pyrebase
from background_task import background
from face_recognition.face_recognition_cli import image_files_in_folder
from sklearn import neighbors

from student_management_app.models import (
    Attendance,
    AttendanceReport,
    CustomUser,
    Students,
    Subjects,
)

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


firebaseConfig = {
    "apiKey": "AIzaSyAq7-ziABaQCTxfeOlMIbv8jvfQk2B7lmQ",
    "authDomain": "pbl5-94125.firebaseapp.com",
    "databaseURL": "https://pbl5-94125-default-rtdb.asia-southeast1.firebasedatabase.app",
    "projectId": "pbl5-94125",
    "storageBucket": "pbl5-94125.appspot.com",
    "messagingSenderId": "42461525472",
    "appId": "1:42461525472:web:e0519d8a1a0e0644f1e785",
    "measurementId": "G-K47401613X",
    "serviceAccount": "serviceAccount.json",
}

firebase = pyrebase.initialize_app(firebaseConfig)
storage = firebase.storage()


def get_all_student():
    students = []
    # students_model = CustomUser.objects.filter(user_type=3)
    # for student in students_model:
    #     students.append(student.username)

    all_files = storage.child("Image").list_files()
    for file in all_files:
        print(file.name)
        student_folder = file.name.split("/")[1]
        if student_folder not in students:
            print(student_folder)
            students.append(student_folder)
        print(students)
    return students


def download_from_firebase():
    print("Downloading image to train from firebase ... ")
    student_folders = get_all_student()  # Lay cac thu muc tren storage ma duoc nop len

    for student_folder in student_folders:
        if(student_folder != ''):
            print("Student folder", student_folder)
            ab = str(1)
            student_folder_id = CustomUser.objects.get(username=student_folder).students.id
            print(student_folder_id)
        #     # local_directory = (
        #     #     "C:/Users/ncanh/OneDrive/Documents/GitHub/PBL5_newest/train_images/"
        #     #     + student_folder
        #     # )
            local_directory = (
                "train_images" + "/" + student_folder + "_" + str(student_folder_id)
            )

            if not os.path.exists(local_directory):
                os.mkdir(local_directory)
            all_files = storage.child("Image").child(student_folder).list_files()
            for file in all_files:
                student_folder_name = file.name.split("/")[1]
                if student_folder_name == student_folder:
                    # local_directory = (
                    #     "C:/Users/ncanh/OneDrive/Documents/GitHub/PBL5_newest/train_images/"
                    #     + student_folder_name
                    #     + "/"
                    #     + ab
                    #     + ".jpg"
                    # )
                    local_path = (
                        "train_images"
                        + "/"
                        + student_folder_name
                        + "_"
                        + str(student_folder_id)
                        + "/"
                        + ab
                        + ".jpg"
                    )
                    print(local_path)

                    file.download_to_filename(local_path)
                    x = int(ab)
                    ab = str(x + 1)
                    print("Download ", file.name, " done")
                else:
                    continue

        print("Downloaded all image!")


def train(
    train_dir,
    model_save_path=None,
    n_neighbors=None,
    knn_algo="ball_tree",
    verbose=False,
):
    """
    Trains a k-nearest neighbors classifier for face recognition.
    :param train_dir: directory that contains a sub-directory for each known person, with its name.
     (View in source code to see train_dir example tree structure)
     Structure:
        <train_dir>/
        ├── <person1>/
        │   ├── <somename1>.jpeg
        │   ├── <somename2>.jpeg
        │   ├── ...
        ├── <person2>/
        │   ├── <somename1>.jpeg
        │   └── <somename2>.jpeg
        └── ...
    :param model_save_path: (optional) path to save model on disk
    :param n_neighbors: (optional) number of neighbors to weigh in classification. Chosen automatically if not specified
    :param knn_algo: (optional) underlying data structure to support knn.default is ball_tree
    :param verbose: verbosity of training
    :return: returns knn classifier that was trained on the given data.
    """
    X = []
    y = []

    # Loop through each person in the training set
    for class_dir in os.listdir(train_dir):
        print(class_dir)
        if not os.path.isdir(os.path.join(train_dir, class_dir)):
            continue
        print(os.path.join(train_dir, class_dir))

        # Loop through each training image for the current person
        for img_path in image_files_in_folder(os.path.join(train_dir, class_dir)):
            print(img_path)
            image = face_recognition.load_image_file(img_path)
            face_bounding_boxes = face_recognition.face_locations(image)

            if len(face_bounding_boxes) != 1:
                # If there are no people (or too many people) in a training image, skip the image.
                if verbose:
                    print(
                        "Image {} not suitable for training: {}".format(
                            img_path,
                            "Didn't find a face"
                            if len(face_bounding_boxes) < 1
                            else "Found more than one face",
                        )
                    )
            else:
                # Add face encoding for current image to the training set
                X.append(
                    face_recognition.face_encodings(
                        image, known_face_locations=face_bounding_boxes
                    )[0]
                )
                y.append(class_dir)
                print(class_dir + " Added to database")

    # Determine how many neighbors to use for weighting in the KNN classifier
    if n_neighbors is None:
        n_neighbors = int(round(math.sqrt(len(X))))
        if verbose:
            print("Chose n_neighbors automatically:", n_neighbors)

    # Create and train the KNN classifier
    knn_clf = neighbors.KNeighborsClassifier(
        n_neighbors=n_neighbors, algorithm=knn_algo, weights="distance"
    )
    knn_clf.fit(X, y)

    # Save the trained KNN classifier
    if model_save_path is not None:
        with open(model_save_path, "wb") as f:
            pickle.dump(knn_clf, f)

    return knn_clf


def training():
    
    print("Training KNN classifier...")

    train_dir = "train_images"
    model_save_path = "trained_knn_model.clf"

    classifier = train(
        train_dir,
        model_save_path,
        verbose=True,
    )
    print("Training complete!")
    

def preprocessing():
    download_from_firebase()
    training()


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
            "Must supply knn classifier either thourgh knn_clf or model_path"
        )

    # Load a trained KNN model (if one was passed in)
    if knn_clf is None:
        with open(model_path, "rb") as f:
            knn_clf = pickle.load(f)

    # Load image file and find face locations
    X_face_locations = face_recognition.face_locations(X_img)

    # If no faces are found in the image, return an empty result.
    if len(X_face_locations) == 0:
        return []

    # Find encodings for faces in the test iamge
    faces_encodings = face_recognition.face_encodings(
        X_img, known_face_locations=X_face_locations
    )

    # Use the KNN model to find the best matches for the test face
    closest_distances = knn_clf.kneighbors(faces_encodings, n_neighbors=1)
    are_matches = [
        closest_distances[0][i][0] <= distance_threshold
        for i in range(len(X_face_locations))
    ]

    # Predict classes and remove classifications that aren't within the threshold
    return [
        (pred, loc) if rec else ("student_unknown", loc)
        for pred, loc, rec in zip(
            knn_clf.predict(faces_encodings), X_face_locations, are_matches
        )
    ]


@background(schedule=0)
def detect_face():
    preprocessing()
    url = "http://192.168.1.3/cam-hi.jpg"
    print("Running bg task .... ")
    
    start_time= datetime.datetime.now()
    
    while True:
        
        img_resp = urllib.request.urlopen(url)
        imgnp = np.array(bytearray(img_resp.read()), dtype=np.uint8)
        frame = cv2.imdecode(imgnp, -1)
        
        width = 640
        height = 480
        dim = (width, height)

        # resize image
        resized = cv2.resize(frame, dim, interpolation=cv2.INTER_AREA)

        rgb_frame = resized[:, :, ::-1]

        predictions = predict(
            rgb_frame, model_path="trained_knn_model.clf", distance_threshold=0.4
        )
        print("OK")
        # Xu li diem danh
        for studentName_id, (top, right, bottom, left) in predictions:
            print("Xu li done")
            id = studentName_id.split("_")[1]
            print("Id student: ", id)
            if id != "unknown":
                student = Students.objects.get(id=id)
                print("Student course id: ", student.course_id_id)
                subject_model = Subjects.objects.filter(course_id=student.course_id_id)
                print("subject model: ", subject_model)
                for subject in subject_model:
                    print("Subject id", subject.id)
                    attendance_date = datetime.datetime.now().date()
                    print(student.session_year_id.id)
                    if not Attendance.objects.filter(subject_id=subject,attendance_date=attendance_date,session_year_id=student.session_year_id).exists():
                        if not AttendanceReport.objects.filter(student_id=student, created_at=attendance_date).exists():
                            attendance=Attendance(subject_id=subject,attendance_date=attendance_date,session_year_id=student.session_year_id) 
                            attendance.save()
                            print("Done save attendance")
                            attendance_report=AttendanceReport(student_id=student,attendance_id=attendance,status=True)
                            attendance_report.save()
                            print("Done save attendance report")
                        else:
                            print("test")
                    else:
                        print("Student already check attendance today")
                        pass
            else:
                print("Student not found!")
        detect_face_running_time = datetime.datetime.now()
        
        elapsed_time =  detect_face_running_time - start_time
        print(elapsed_time.seconds)
        if elapsed_time.seconds >= 86400:
            break
         
       


class FaceDetect(object):
    def __init__(self):
        # change the IP address below according to the
        # IP shown in the Serial monitor of Arduino code
        self.url = "http://192.168.1.9/cam-hi.jpg"

    def __del__(self):
        cv2.destroyAllWindows()

    def get_frame(self):
        # grab the frame from the threaded video stream
        img_resp = urllib.request.urlopen(self.url)
        imgnp = np.array(bytearray(img_resp.read()), dtype=np.uint8)
        frame = cv2.imdecode(imgnp, -1)
        # frame = cv2.flip(frame,1)

        # resize the frame to have a width of 600 pixels (while
        # maintaining the aspect ratio), and then grab the image
        # dimensions
        width = 640
        height = 480
        dim = (width, height)

        # # resize image
        resized = cv2.resize(frame, dim, interpolation=cv2.INTER_AREA)

        rgb_frame = resized[:, :, ::-1]

        predictions = predict(
            rgb_frame, model_path="trained_knn_model.clf", distance_threshold=0.4
        )

        for name, (top, right, bottom, left) in predictions:
            # Draw a box around the face
            cv2.rectangle(resized, (left, top), (right, bottom), (0, 0, 255), 2)

            # Draw a label with a name below the face
            cv2.rectangle(
                resized, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED
            )
            font = cv2.FONT_HERSHEY_DUPLEX
            cv2.putText(
                resized, name, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1
            )

            # ser.write(name.encode())

        ret, jpeg = cv2.imencode(".jpg", resized)

        return jpeg.tobytes()
