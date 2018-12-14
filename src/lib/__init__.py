'''
This file contains common functions used across models.
'''

# Import required libraries
import numpy as np
from keras.models import model_from_yaml
import _pickle as cPickle
from keras.utils import to_categorical


def get_data(name, data_file_path):
    '''
        Purpose: Function that loads the data from the given path
        Arguements: Name of source, Path to source
        Return: raw data containing images and labels
    '''
    if(name == 'fer2013'):
        print("Reading the fer2013 dataset...")
        with open(data_file_path) as fp:
            data = fp.readlines()
        print("Labels for the dataset are: {}".format(data[0].split(",")))
        data = np.array(data[1:])
        print("Number of images read: {}".format(data.size))
        print("Returning the fer2013 dataset...")
    return data

def get_icvMEFED(name, data_file_path):
    if name == 'icvMEFED':
        rawData = pd.read_excel(data_file_path)
        return rawData


def generate_data_split(data, num_of_classes=7, name='fer2013'):
    '''
        Purpose: Function that splits the data into training and test sets
        Arguements: data, number of classes, dataset name
        Return: raw data containing images and labels
    '''
    X_train = []
    Y_train = []
    X_test = []
    Y_test = []
    if(name == 'fer2013'):
        for point in data:
            try:
                emotion, image, instance_type = point.split(",")
                image_pixels = np.array(image.split(" "), 'float32')
                emotion = to_categorical(emotion, num_of_classes)
                if(instance_type.strip() == 'Training'):
                    X_train.append(image_pixels)
                    Y_train.append(emotion)
                else:
                    X_test.append(image_pixels)
                    Y_test.append(emotion)
            except Exception as e:
                print("ERROR - {}".format(e))

        X_train = np.array(X_train, 'float32')
        Y_train = np.array(Y_train, 'float32')
        X_test = np.array(X_test, 'float32')
        Y_test = np.array(Y_test, 'float32')
        return (X_train, Y_train, X_test, Y_test)
    return


def load_model(name='cnn'):
    '''
        Purpose: Load the saved model
        Arguements: Model name
        Return: Loaded model
    '''
    if(name == 'cnn'):
        with open("model_cnn.yaml", "r") as yaml_file:
            loaded_model_yaml = yaml_file.read()
        model = model_from_yaml(loaded_model_yaml)
        model.load_weights("model_cnn.h5")
        # model.compile(loss='categorical_crossentropy',optimizer = Adam(), metrics = ['accuracy'])
        print("Loaded model from disk")
    else:
        with open('model_'+name+'.pkl', 'rb') as fid:
            model = cPickle.load(model, fid)
    return (model)


def save_model(name, model):
    '''
        Purpose: Save the trained model
        Arguements: Model name, model
        Return: N/A
    '''
    if(name == 'cnn'):
        model_yaml = model.to_yaml()
        with open("model_cnn.yaml", "w") as yaml_file:
            yaml_file.write(model_yaml)
        # serialize weights to HDF5
        model.save_weights("model_cnn.h5")
    else:
        with open('model_'+name+'.pkl', 'wb') as fid:
            cPickle.dump(model, fid)
    print("Saved model to disk.")
