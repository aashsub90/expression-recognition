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


def get_landmarks(image):
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
    detections = detector(image, 1)
    data = {}
    for k,d in enumerate(detections): #For all detected face instances individually
        shape = predictor(image, d) #Draw Facial Landmarks with the predictor class
        xlist = []
        ylist = []
        for i in range(1,68): #Store X and Y coordinates in two lists
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
            meannp = np.asarray((ymean,xmean))
            coornp = np.asarray((z,w))
            dist = np.linalg.norm(coornp-meannp)
            landmarks_vectorised.append(dist)
            landmarks_vectorised.append((math.atan2(y, x)*360)/(2*math.pi))
        data['landmarks_vectorised'] = landmarks_vectorised
    if len(detections) < 1:
        data['landmarks_vestorised'] = "error"
    return data
        
        
        
def make_sets(path,rawData):
    training_data = []
    training_labels = []
#     prediction_data = []
#     prediction_labels = []
    for i,j in zip(rawData['name'], rawData['label']):
        image = cv2.imread(path+'/'+i) #open image
        if image is None:
            print("Could not read input image")
            exit()
        else:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) #convert to grayscale
            clahe_image = clahe.apply(gray)
            get_landmarks(clahe_image)
            if data['landmarks_vectorised'] == "error":
                print("no face detected on this one")
            else:
                training_data.append(data['landmarks_vectorised']) #append image array to training data list
                training_labels.append(j)
            
    return training_data, training_labels