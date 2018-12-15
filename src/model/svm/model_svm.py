
# coding: utf-8

# In[ ]:


from lib import get_data, save_model
from lib.preprocessing import make_sets, get_landmarks, normalize_data
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


# In[ ]:


emotions = ['angry', 'fear', 'happy', 'sad', 'neutral', 'surprise']

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


def get_(emotion):
    files = glob.glob("images/%s/*" % emotion)
    random.shuffle(files)
    training = files[:int(len(files)*0.8)]
    prediction = files[-int(len(files)*0.2):]
    return training, prediction


def evaluate_model(model, X_train, Y_train, X_test, Y_test):

    train_score = model.score(X_train, Y_train)
    test_score = model.score(X_test, Y_test)
    print(X_test.shape)
    predictions = model.predict(X_test)
    print(predictions)
    print(train_score, test_score)
    #print("Train loss: {}".format(train_score[0]))
    #print("Test loss: {}".format(test_score[0]))

    #print("Train accuracy: {}".format(train_score[1]))
    #print("Test accuracy: {}".format(test_score[1]))


def generate_model(X_train, Y_train):
    #train_data = np.array(get_landmarks(X_train))
    clf = SVC(kernel=kernel, probability=probability, tol=tol, max_iter=10)
    # SVC(C = c, kernel=kernel, degree=degree, gamma= gamma,
    #     coef0=coef0, shrinking = shrink, probability= probability, tol= tol, cache_size= size,
    #     class_weight=None, verbose = verbose, max_iter= max_iter, decision_function_shape = function, random_state=None)
    clf.fit(X_train, Y_train)
    return clf


def main(data_name):
    #clf = SVC(kernel='linear', probability=True, tol=1e-3)
    # , verbose = True) #Set the classifier as a support vector machines with polynomial kernel

    if(data_name == "fer2013"):
        name = "fer2013"
        data_file_path = "../../../data/fer2013/fer2013.csv"
    else:
        name = "icv_mefed"
        train_path = '../../../data/icv_mefed/training/'
        test_path = '../../../data/icv_mefed/testing/'

    # Obtain the data from the path provided
    train_data = get_data(name=name, data_file_path=train_path+'training.txt')
    training_data, training_labels = make_sets(train_data, train_path)
    test_data = get_data(name=name, data_file_path=test_path+'testing.txt')
    testing_data, testing_labels = make_sets(test_data, test_path)

    # Turn the training set into a numpy array for the classifier
    X_train = np.array(training_data)
    Y_train = np.array(training_labels)
    X_test = np.array(testing_data)
    Y_test = np.array(testing_labels)

    # Pre-process the image data
    X_train, X_test = normalize_data(X_train, X_test)

    # Generate or load trained model
    model = generate_model(X_train, Y_train)

    # Evaluate model
    evaluate_model(model, X_train, Y_train, X_test, Y_test)

    # Save model to disk
    save_model(name="svm", model=model)


if __name__ == "__main__":
    main(sys.argv[1])
