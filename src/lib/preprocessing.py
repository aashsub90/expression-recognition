import math
'''
    This file contains all the preprocessing functions used by different models
'''
import cv2
import sys
import dlib
import numpy as np
sys.path.append('../../lib/dlib')
from keras.utils import to_categorical

def normalize_data(X_train, X_test):

    # Normalize pixel values
    X_train = X_train / 255
    X_test = X_test / 255

    return (X_train, X_test)


def get_landmarks(image):
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor(
        "../../lib/shape_predictor_68_face_landmarks.dat")
    detections = detector(image, 1)
    data = {}
    for k, d in enumerate(detections):  # For all detected face instances individually
        # Draw Facial Landmarks with the predictor class
        shape = predictor(image, d)
        xlist = []
        ylist = []
        for i in range(1, 68):  # Store X and Y coordinates in two lists
            xlist.append(float(shape.part(i).x))
            ylist.append(float(shape.part(i).y))
        xmean = np.mean(xlist)
        ymean = np.mean(ylist)
        xcentral = [(x-xmean) for x in xlist]
        ycentral = [(y-ymean) for y in ylist]
        landmarks_vectorised = []
        for x, y, w, z in zip(xcentral, ycentral, xlist, ylist):
            landmarks_vectorised.append(w)
            landmarks_vectorised.append(z)
            meannp = np.asarray((ymean, xmean))
            coornp = np.asarray((z, w))
            dist = np.linalg.norm(coornp-meannp)
            landmarks_vectorised.append(dist)
            landmarks_vectorised.append((math.atan2(y, x)*360)/(2*math.pi))
        data['landmarks_vectorised'] = landmarks_vectorised
    if len(detections) < 1:
        data['landmarks_vectorised'] = "error"
    return data


def make_sets(rawData, path, extract_landmarks=True):
    training_data = []
    training_labels = []
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
#     prediction_data = []
#     prediction_labels = []
    print('Reading from path: {}\n'.format(path))
    print(rawData.label.unique())
    for i, j in zip(rawData['file_name'], rawData['label']):
        print('Reading image: {}\n'.format(i))
        image = cv2.imread(path+'/'+i)  # open image
        if image is None:
            print("Warning - Could not read input image")
            continue
        else:
            # convert to grayscale
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            clahe_image = clahe.apply(gray)
            if(extract_landmarks):
                data = get_landmarks(cv2.resize(clahe_image, None, fx=0.1, fy=0.1))
                #data = get_landmarks(cv2.resize(clahe_image,(48,48)))
                if data['landmarks_vectorised'] == "error":
                    print("No face detected on this one")
                else:
                    # append image array to training data list
                    training_data.append(data['landmarks_vectorised'])
                    training_labels.append(j)
            else:
                print(np.array(clahe_image).shape)
                training_data.append(cv2.resize(clahe_image, None, fx=0.1, fy=0.1))
                training_labels.append(to_categorical(j, num_classes=None))

    return training_data, training_labels
