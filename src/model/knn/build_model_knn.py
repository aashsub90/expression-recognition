import tarfile
import pandas as pd
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import sklearn.preprocessing as pre
from lib import get_data, generate_data_split, load_model, save_model
from lib.preprocessing import make_sets, get_landmarks, normalize_data
import sys
from sklearn.model_selection import cross_val_score

def find_k(X_train, Y_train):
    myList = list(range(1, 20))
    # subsetting just the odd ones
    neighbors = list(filter(lambda x: x % 2 != 0, myList))
    cv_scores = []
    # perform 10-fold cross validation
    for k in neighbors:
        knn = KNeighborsClassifier(n_neighbors=k, n_jobs=-1)
        scores = cross_val_score(knn, X_train, Y_train, cv=10, scoring='accuracy')
    cv_scores.append(scores.mean())
    MSE = [1 - x for x in cv_scores]
    # determining best k
    optimal_k = neighbors[MSE.index(min(MSE))]
    print("The optimal number of neighbors is %d" % optimal_k)
    # plot misclassification error vs k
    fig = plt.figure()
    plt.plot(neighbors, MSE)
    plt.xlabel('Number of Neighbors K')
    plt.ylabel('Misclassification Error')
    fig.savefig('knn_cross_val_plot.png')


def generate_model(X_train, Y_train):
    # X_train = X_train.reshape(X_train.shape[0], X_train.shape[1]*X_train.shape[2]*X_train.shape[3])
    # print(X_train.shape)
    neigh = KNeighborsClassifier(n_neighbors=2, weights='distance')
    neigh.fit(X_train, Y_train)
    return(neigh)


def evaluate_model(model, X_train, Y_train, X_test, Y_test):
    # X_test = X_test.reshape(X_test.shape[0], X_test.shape[1]*X_test.shape[2]*X_test.shape[3])
    train_score = model.score(X_train, Y_train)
    test_score = model.score(X_test, Y_test)
    print(X_test.shape)
    predictions = model.predict(X_test)
    print(predictions)
    print(train_score, test_score)


def main(data_name):
    if(data_name == "fer2013"):
        name = "fer2013"
        data_file_path = "../../../data/fer2013/fer2013.csv"
    else:
        name = "icv_mefed"
        train_path = "../../../data/icv_mefed/training/"
        test_path = "../../../data/icv_mefed/testing/"

    # Obtain the data from the path provided
    # data = get_data(name=name, data_file_path=data_file_path)

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

    # Generate training and test sets from the data
    # X_train, Y_train, X_test, Y_test = generate_data_split(data=data, num_of_classes=7, name=name)

    # Pre-process the image data
    X_train, X_test = normalize_data(X_train, X_test)

    #find_k(X_train,Y_train)
    # Generate or load trained model
    model = generate_model(X_train, Y_train)

    # Evaluate model
    evaluate_model(model, X_train, Y_train, X_test, Y_test)

    # Save model
    save_model(name="knn", model=model)


if __name__ == "__main__":
    main(sys.argv[1])
