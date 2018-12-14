
# coding: utf-8

# In[ ]:


import pickle
import dlib
import cv2
import imutils
import glob
import random
import math
import numpy as np
import itertools
from sklearn.svm import SVC
import sys
sys.path.append('/Users/niki/dlib')


# In[ ]:


emotions = ['angry', 'fear', 'happy', 'sad', 'neutral', 'surprise']
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")


# In[ ]:


c = 1.0
kernel = 'linear'
degree = 3
gamma = 'auto_deprecated'
coef0 = 0.0
tol = 1e-3
size = 200
max_iter = -1
function = 'ovr'
probability = True
shrink = True
verbose = False


# In[ ]:


#data = {}
def get_(emotion):
    files = glob.glob("images/%s/*" % emotion)
    random.shuffle(files)
    training = files[:int(len(files)*0.8)]
    prediction = files[-int(len(files)*0.2):]
    return training, prediction


'''def get_landmarks(image):
    #     image = cv2.imread(img)
    #     image = imutils.resize(image, width=500)
    #     gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # detect faces in the grayscale image
    #     rects = detector(gray, 1)
    data = {}
    detections = detector(image, 1)
    for k, d in enumerate(detections):
        shape = predictor(image, d)
        xlist = []
        ylist = []
        vectorised_landmarks = []
        for i in range(1, 68):
            xlist.append(float(shape.part(i).x))
            ylist.append(float(shape.part(i).y))
        x_mean = np.mean(xlist)
        y_mean = np.mean(ylist)
        x_central = [(x-x_mean) for x in xlist]
        y_central = [(y-y_mean) for y in ylist]
        for x, y, w, z in zip(x_central, y_central, xlist, ylist):
            vectorised_landmarks.append(w)
            vectorised_landmarks.append(z)
            mean_np = np.asarray((y_mean, x_mean))
            coornp = np.asarray((z, w))
            dist = np.linalg.norm(coornp - mean_np)
            vectorised_landmarks.append(dist)
            vectorised_landmarks.append((math.atan2(y, x)*360)/(2*math.pi))
        data['vectorised_landmarks'] = vectorised_landmarks

    if len(detections) < 1:
        data['vectorised_landmarks'] = 'error'
    return data'''


def get_landmarks(X):
    #     image = cv2.imread(img)
    #     image = imutils.resize(image, width=500)
    #     gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # detect faces in the grayscale image
    #     rects = detector(gray, 1)
    landmark_data = []
    for image in X:
        detections = detector(image, 1)
        for k, d in enumerate(detections):
            shape = predictor(image, d)
            xlist = []
            ylist = []
            vectorised_landmarks = []
            for i in range(1, 68):
                xlist.append(float(shape.part(i).x))
                ylist.append(float(shape.part(i).y))
            x_mean = np.mean(xlist)
            y_mean = np.mean(ylist)
            x_central = [(x-x_mean) for x in xlist]
            y_central = [(y-y_mean) for y in ylist]
            for x, y, w, z in zip(x_central, y_central, xlist, ylist):
                vectorised_landmarks.append(w)
                vectorised_landmarks.append(z)
                mean_np = np.asarray((y_mean, x_mean))
                coornp = np.asarray((z, w))
                dist = np.linalg.norm(coornp - mean_np)
                vectorised_landmarks.append(dist)
                vectorised_landmarks.append((math.atan2(y, x)*360)/(2*math.pi))

            data['vectorised_landmarks'] = vectorised_landmarks

        if len(detections) >= 1:
            landmark_data.append(data['vectorised_landmarks'])
            #data['vectorised_landmarks'] = 'error'
    return landmark_data


# In[ ]:

'''def make_sets():
    training_data = []
    training_labels = []
    prediction_data = []
    prediction_labels = []
    
    for emotion in emotions:
        print("working on %s" %emotion)
        training, prediction = get_files(emotion)
        
        for item in training:
            image = cv2.imread(item) 
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            clahe_image = clahe.apply(gray)
            get_landmarks(clahe_image)
            if data['vectorised_landmarks'] == 'error':
                print('no face detected on this one')
            else:
                training_data.append(data['vectorised_landmarks'])
                training_labels.append(emotions.index(emotion))
        
        for item in prediction:
            image = cv2.imread(item)
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            clahe_image = clahe.apply(gray)
            get_landmarks(clahe_image)
            if data['vectorised_landmarks'] == "error":
                print("no face detected on this one")
            else:
                prediction_data.append(data['vectorised_landmarks'])
                prediction_labels.append(emotions.index(emotion))
                
    return training_data, training_labels, prediction_data, prediction_labels
'''

# In[ ]:


'''accur_lin = []
for i in range(0, 10):
    print("Making sets %s" %i)
    training_data, training_labels, prediction_data, prediction_labels = make_sets()
    print('training labels', training_labels)
    npar_train = np.array(training_data)
#     length = len(npar_train)
#     npar_train = npar_train.reshape((length,1))
#     print(npar_train.shape)
#     print(type(npar_train))
    npar_trainlabs = np.array(training_labels)
#     print('labels',len(npar_trainlabs))
    print("training SVM linear %s" %i) #train SVM
    clf.fit(npar_train, npar_trainlabs)
    print("getting accuracies %s" %i)
    npar_pred = np.array(prediction_data)
    pred_lin = clf.score(npar_pred, prediction_labels)
    print("linear: ", pred_lin)
    accur_lin.append(pred_lin)
print("Mean value lin svm: %s" %np.mean(accur_lin)) '''


# In[ ]:


# save the model to disk
filename = 'finalized_model.sav'
pickle.dump(clf, open(filename, 'wb'))


# In[ ]:


# if __name__ == "__main__":
#     input_Data = get_landmarks('img.jpg')
# #     x, y = input_Data.merge_Train_Test()
#     print(input_Data)

def generate_model(X_train, Y_train):
    train_data = np.array(get_landmarks(X_train))
    clf = SVC(kernel=kernel, probability=probability, tol=tol)
    # SVC(C = c, kernel=kernel, degree=degree, gamma= gamma,
    #     coef0=coef0, shrinking = shrink, probability= probability, tol= tol, cache_size= size,
    #     class_weight=None, verbose = verbose, max_iter= max_iter, decision_function_shape = function, random_state=None)
    clf.fit(train_data, Y_train)
    return clf


def main(data_name):
    clf = SVC(kernel='linear', probability=True, tol=1e-3)
    #, verbose = True) #Set the classifier as a support vector machines with polynomial kernel

    if(data_name == "fer2013"):
        name = "fer2013"
        path = ''
        data_file_path = "../../../data/fer2013/fer2013.csv"

    # Obtain the data from the path provided
    data = get_icvMEFED(name=name, data_file_path=data_file_path)
    training_data, training_labels = make_sets(data,path)
    npar_train = np.array(training_data) #Turn the training set into a numpy array for the classifier
    npar_trainlabs = np.array(training_labels)
    print("training SVM linear %s" %i) #train SVM
    clf.fit(npar_train, training_labels)
    print('training done')

#     # Generate training and test sets from the data
#     X_train, Y_train, X_test, Y_test = generate_data_split(
#         data=data, num_of_classes=7, name=name)

#     # Pre-process the image data
#     X_train, X_test = normalize_data(X_train, X_test)

    # Generate or load trained model
    model = generate_model(X_train, Y_train)

    # Evaluate model
    #evaluate_model(model, X_train, Y_train, X_test, Y_test)

    # Save model to disk
    save_model(name="svm", model=model)

    # Predict for a given sample
    #sample_image_path = 'sample.png'
    #predict(model, sample_image_path)


if __name__ == "__main__":
    # main(sys.argv[1])
    main()
