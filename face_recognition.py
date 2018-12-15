
# coding: utf-8

# In[ ]:


# Import required libraries
import cv2
import numpy as np
from keras.preprocessing import image as img
from keras.models import model_from_yaml
import pickle
import sys
sys.path.insert(0, '/Users/Iris/SJSU/Fall_2018/CMPE_257/Project/Group/repo/expression-recognition/src')
print(sys.path)
from lib.preprocessing import get_landmarks


# In[ ]:


def opencv_init():
    '''
        Purpose: Creates the classifier and video capture object
        Arguements: N/A
        Return: classifier and video capture object
    '''
    face_detector_object = cv2.CascadeClassifier('/Users/Iris/DeveloperTools/anaconda3/envs/expression-recognition/share/OpenCV/haarcascades/haarcascade_frontalface_default.xml')
    video_capture_object = cv2.VideoCapture(0)
    return (face_detector_object, video_capture_object)


# In[ ]:


def opencv_cleanup(cap):
    '''
        Purpose: Release resource held by opencv
        Arguements: OpenCV video object
        Return: N/A
    '''
    cap.release()
    cv2.destroyAllWindows()


# In[ ]:


def load_model():
    '''
        Purpose: Load the saved model
        Arguements: N/A
        Return: Loaded model
    '''
    '''with open("model_cnn.yaml", "r") as yaml_file:
        loaded_model_yaml = yaml_file.read()
    model = model_from_yaml(loaded_model_yaml)
    model.load_weights("model_cnn.h5")
    print("Loaded model from disk")
    return (model)'''
    model = pickle.load(open('/Users/Iris/SJSU/Fall_2018/CMPE_257/Project/Group/repo/expression-recognition/src/model/svm/model_svm.pkl', 'rb'))
    #model = pickle.load(open('/Users/Iris/SJSU/Fall_2018/CMPE_257/Project/Group/repo/expression-recognition/src/model/knn/model_knn.pkl', 'rb'))
    return(model)


# In[ ]:


def emotion_detection(face_detect, video_capture, model):
    '''
        Purpose: Begin video cature and perform emotion detection of all faces detected
        Arguements: face_detect, video_capture and model references
        Return: N/A
    '''
    emotion_classes = {'1_1':'angry', '1_2':'contemptly angry', '1_3':'disgustingly angry','1_4':'fearfully angry','1_5':'happily angry', '1_6':'sadly angry','1_7':'surprsingly angry',  '2_1':'angrily contempt', '2_2':'contempt','2_3':'disgustingly contempt','2_4':'fearfully contempt','2_5':'happily contempt','2_6':'sadly contempt', '2_7':'surprisingly contempt', '3_1':'angrily disgusted','3_2':'contemptly disgusted','3_3':'disgust','3_4':'fearfully disgusted','3_5':'happily disgusted','3_6':'sadly disgusted','3_7':'surprisingly disgusted', '4_1':'angrily fearful', '4_2':'contemptly fearful', '4_3':'disgustigly fearful','4_4':'fearful', '4_5':'happily fearful', '4_6':'sadly fearful', '4_7':'surprisingly fearful','5_1':'angrily happy', '5_2':'contemptly happy', '5_3':'disgustingly happy', '5_4':'fearfully happy','5_5':'happy', '5_6':'sadly happy','5_7':'surprisingly happy','6_1':'angrily sad', '6_2':'contemptly sad', '6_3':'disgustigly sad', '6_4':'fearfully sad', '6_5':'happily sad','6_6':'sad','6_7':'surprisingly sad', '7_1':'angrily surprised', '7_2':'contemptly surprised', '7_3':'disgustingly surprised','7_4':'fearfully surprised', '7_5':'happily surprised', '7_6':'sadly surprised','7_7':'surprised','N_N':'Neutral'}
    while(True):
        ret, image = video_capture.read()
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        faces = face_detect.detectMultiScale(gray, 1.3, 5)

        print("Detected face positions: {}".format(faces))

        for (x,y,w,h) in faces:
            # Create boundary for the recognirzed face
            cv2.rectangle(image,(x,y),(x+w,y+h),(255,0,0),2)
            # Crop detected faces
            detected_face = image[int(y):int(y+h), int(x):int(x+w)]

            # Transform the detected face to grayscale
            detected_face = cv2.cvtColor(detected_face, cv2.COLOR_BGR2GRAY)
            label = ""
            data = get_landmarks(detected_face)
            if data['landmarks_vectorised'] == "error":
                print("Can't detect landmarks.")
            else:
                # append image array to training data list
                label = model.predict(np.array(data['landmarks_vectorised']).reshape(1,-1))[0]
                emotion = emotion_classes[label]
                print(emotion)
                
            # Resize face image to 48x48
            #detected_face = cv2.resize(detected_face, (48, 48))

            #img_pixels = img.img_to_array(detected_face)
            #img_pixels = np.expand_dims(img_pixels, axis = 0)
            #img_pixels /= 255 
            #img_pixels = np.vstack(img_pixels)
            #print(img_pixels.reshape(48,48))
            # Obtain predictions for the detected face
            #predictions = model.predict(img_pixels.reshape(48,48))
            #predictions = model.predict(img_pixels)

            # Find index of the maximum value 
            # Array - 0: angry, 1:disgust, 2:fear, 3:happy, 4:sad, 5:surprise, 6:neutral
            #max_index = np.argmax(predictions[0])


            # Display the emotion text above rectangle
            cv2.putText(image, emotion, (int(x), int(y)), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)

        cv2.imshow('img',image)
        
        # Break video capture event
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break


# In[ ]:


def main():
    # Intialize opencv and obtain face detection and video capture objects
    face_detect, video_capture = opencv_init()
    
    # Load the model 
    model = load_model()
    
    # Start video capture and emotion recognition
    emotion_detection(face_detect, video_capture, model)
    
    # Deallocate resources used by OpenCV
    opencv_cleanup(video_capture)


# In[ ]:


if __name__ == "__main__":
    main()

