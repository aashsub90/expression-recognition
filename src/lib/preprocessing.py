'''
    This file contains all the preprocessing functions used by different models
'''


def normalize_data(X_train, X_test):

    # Normalize pixel values
    X_train = X_train / 255
    X_test = X_test / 255

    X_train = X_train.reshape(X_train.shape[0], 48, 48, 1).astype('float32')
    X_test = X_test.reshape(X_test.shape[0], 48, 48, 1).astype('float32')

    return (X_train, X_test)
