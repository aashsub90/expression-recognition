# Import necessary libraries

from lib.preprocessing import normalize_data
from lib import get_data, generate_data_split, load_model, save_model
import os.path
import numpy as np
from keras import Sequential
from keras.layers import Conv2D, MaxPooling2D, AveragePooling2D, Dense, Flatten, Dropout
from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing import image
from keras.optimizers import Adam
from keras.models import model_from_yaml
#import matplotlib.image as mpimg
#import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
import sys
from lib.preprocessing import make_sets, get_landmarks, normalize_data

import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'


def generate_model(X_train, Y_train):
    model = Sequential()

    model.add(Conv2D(filters=64, kernel_size=(5, 5), strides=(1, 1), padding='valid', dilation_rate=(1, 1), activation='relu', use_bias=True, kernel_initializer='glorot_uniform',
                     bias_initializer='zeros', kernel_regularizer=None, bias_regularizer=None, activity_regularizer=None, kernel_constraint=None, bias_constraint=None, input_shape=(346, 518, 1)))
    model.add(MaxPooling2D(pool_size=(5, 5), strides=(
        2, 2), padding='valid', data_format=None))

    model.add(Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='valid', dilation_rate=(1, 1), activation='relu', use_bias=True, kernel_initializer='glorot_uniform',
                     bias_initializer='zeros', kernel_regularizer=None, bias_regularizer=None, activity_regularizer=None, kernel_constraint=None, bias_constraint=None, input_shape=(346, 518, 1)))
    model.add(Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='valid', dilation_rate=(1, 1), activation='relu', use_bias=True, kernel_initializer='glorot_uniform',
                     bias_initializer='zeros', kernel_regularizer=None, bias_regularizer=None, activity_regularizer=None, kernel_constraint=None, bias_constraint=None, input_shape=(346, 518, 1)))
    model.add(Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='valid', dilation_rate=(1, 1), activation='relu', use_bias=True, kernel_initializer='glorot_uniform',
                     bias_initializer='zeros', kernel_regularizer=None, bias_regularizer=None, activity_regularizer=None, kernel_constraint=None, bias_constraint=None, input_shape=(346, 518, 1)))
    model.add(AveragePooling2D(pool_size=(3, 3), strides=(2, 2)))

    model.add(Conv2D(128, (3, 3), activation='relu'))
    model.add(Conv2D(128, (3, 3), activation='relu'))
    model.add(AveragePooling2D(pool_size=(3, 3), strides=(2, 2)))

    model.add(Flatten())

    model.add(Dense(1024, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(1024, activation='relu'))
    model.add(Dropout(0.2))

    model.add(Dense(50, activation='softmax'))

    # Fit the model with the training data
    model = fit_model(model, X_train, Y_train)
    return model


def fit_model(model, X_train, Y_train, batch_size=2, epochs=1, loss_function='categorical_crossentropy'):
    data_generator = ImageDataGenerator(featurewise_center=False, samplewise_center=False, featurewise_std_normalization=False, samplewise_std_normalization=False, zca_whitening=False, zca_epsilon=1e-06, rotation_range=0, width_shift_range=0.0, height_shift_range=0.0,
                                        brightness_range=None, shear_range=0.0, zoom_range=0.0, channel_shift_range=0.0, fill_mode='nearest', cval=0.0, horizontal_flip=False, vertical_flip=False, rescale=None, preprocessing_function=None, data_format=None, validation_split=0.0, dtype=None)
    # print(np.array(X_train).reshape(
    #   1, len(X_train[0]), len(X_train[0][0]), len(X_train)))
    # X_train = np.array(X_train)
    X_train = np.array(X_train).reshape(
        len(X_train), len(X_train[0]), len(X_train[0][0]), 1)
    training_data_generator = data_generator.flow(
        X_train, Y_train, batch_size=batch_size)
    model.compile(loss=loss_function, optimizer=Adam(), metrics=['accuracy'])
    model.fit_generator(training_data_generator,
                        steps_per_epoch=batch_size, epochs=epochs)
    return model


def evaluate_model(model, X_train, Y_train, X_test, Y_test):

    X_train = np.array(X_train).reshape(
        len(X_train), len(X_train[0]), len(X_train[0][0]), 1)
    X_test = np.array(X_train).reshape(
        len(X_train), len(X_train[0]), len(X_train[0][0]), 1)
    train_score = model.evaluate(X_train, Y_train, verbose=0)
    test_score = model.evaluate(X_test, Y_test, verbose=0)

    print("Train loss: {}".format(train_score[0]))
    print("Test loss: {}".format(test_score[0]))

    print("Train accuracy: {}".format(train_score[1]))
    print("Test accuracy: {}".format(test_score[1]))


def show_image(image_path):
    try:
        image = mpimg.imread(image_path)
        imgplot = plt.imshow(image)
        plt.show()
    except:
        print("ERROR - Unable to read and plot the provided image")
        exit()


def emotion_display(predicted_label):
    emotion_classes = ['angry', 'disgust', 'fear',
                       'happy', 'sad', 'surprise', 'neutral']
    y_ticks = np.arange(len(emotion_classes))
    print("Predicted emotion is: {}".format(emotion_classes[predicted_label]))
    plt.bar(y_ticks, emotion_classes, align='center', alpha=0.5)
    plt.xticks(y_ticks, emotion_classes)
    plt.ylabel('percentage')
    plt.title('emotion')
    plt.show()


def predict(model, image_path):
    img = image.load_img(
        image_path, color_mode='grayscale', target_size=(48, 48))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x /= 255
    predicted_label = model.predict(x)
    print("Prediction: {}".format(predicted_label))
    # Draw image provided
    show_image(image_path)
    # Draw emotion pie chart
    emotion_display(predicted_label)


'''def generate_confusion_matrix(predictions, Y_test):
    predicted_list = []
    actual_list = []
    for i in predictions:
        predicted_list.append(np.argmax(i))
    for i in Y_test:
        actual_list.append(np.argmax(i))
    conf_matrix = confusion_matrix(actual_list, predicted_list)
    plt.figure()
    plt.imshow(conf_matrix, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Confusion matrix')
    plt.colorbar()
    classes = ['angry', 'disgust', 'fear',
               'happy', 'sad', 'surprise', 'neutral']
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)
    thresh = conf_matrix.max() / 2.
    for i, j in itertools.product(range(conf_matrix.shape[0]), range(conf_matrix.shape[1])):
        plt.text(j, i, format(cm[i, j], 'd'),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()'''


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
    training_data, training_labels = make_sets(
        train_data, train_path, extract_landmarks=False)
    test_data = get_data(name=name, data_file_path=test_path+'testing.txt')
    testing_data, testing_labels = make_sets(
        test_data, test_path, extract_landmarks=False)

    # Generate training and test sets from the data
    # X_train, Y_train, X_test, Y_test = generate_data_split(data=data, num_of_classes=7, name=name)

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
    save_model(name="cnn", model=model)

    # Predict for a given sample
    # sample_image_path = 'sample.png'
    # predict(model, sample_image_path)


if __name__ == "__main__":
    main(sys.argv[1])

