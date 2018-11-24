import tarfile
import pandas as pd
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt
import sklearn.preprocessing as pre
from lib.preprocessing import normalize_data
from lib import get_data, generate_data_split, load_model, save_model
import sys

'''df = pd.read_csv(
    "/Users/Iris/SJSU/Fall_2018/CMPE_257/Project/Group/local/data/fer2013.csv")
df["Usage"].value_counts()
train = df[["emotion", "pixels"]][df["Usage"] == "Training"]
train.isnull().sum()
train['pixels'] = train['pixels'].apply(lambda im: np.fromstring(im, sep=' '))
x_train = np.vstack(train['pixels'].values)
y_train = np.array(train["emotion"])

x_train.shape, y_train.shape

public_test_df = df[["emotion", "pixels"]][df["Usage"] == "PublicTest"]
public_test_df["pixels"] = public_test_df["pixels"].apply(
    lambda im: np.fromstring(im, sep=' '))
x_test = np.vstack(public_test_df["pixels"].values)
y_test = np.array(public_test_df["emotion"])

# normalize
x_train = pre.normalize(x_train, axis=1)
x_test = pre.normalize(x_test, axis=1)'''


def generate_model(X_train, Y_train):
    X_train = X_train.reshape(
        X_train.shape[0], X_train.shape[1]*X_train.shape[2]*X_train.shape[3])
    print(X_train.shape)
    neigh = KNeighborsClassifier(n_neighbors=7, weights='distance')
    neigh.fit(X_train, Y_train)
    return(neigh)


def evaluate_model(model, X_train, Y_train, X_test, Y_test):
    X_test = X_test.reshape(
        X_test.shape[0], X_test.shape[1]*X_test.shape[2]*X_test.shape[3])
    print(model.score(X_test, Y_test))


def main(data_name):
    if(data_name == "fer2013"):
        name = "fer2013"
        data_file_path = "../../../data/fer2013/fer2013.csv"

    # Obtain the data from the path provided
    data = get_data(name=name, data_file_path=data_file_path)

    # Generate training and test sets from the data
    X_train, Y_train, X_test, Y_test = generate_data_split(
        data=data, num_of_classes=7, name=name)

    # Pre-process the image data
    X_train, X_test = normalize_data(X_train, X_test)

    # Generate or load trained model
    model = generate_model(X_train, Y_train)

    # Evaluate model
    evaluate_model(model, X_train, Y_train, X_test, Y_test)

    # Save model
    save_model(name="knn", model=model)


if __name__ == "__main__":
    main(sys.argv[1])
