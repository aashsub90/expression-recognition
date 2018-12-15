# expression-recognition
This repository contains code for a expression recognition system as part of CMPE 257 - Machine Learning at San Jose State University.

This repository contains the code for 2 components in the real-time emotion detection system, the face-detection component and emotion reconition component.

Data:

The dataset used for this is the iCV MEFED dataset. The drive link to this dataset has been provided in the submission comments.

1. Face Recognition:


2. Emotion Detection

	- Model training

		CNN
		- cd src/model/cnn
		- python build_model_cnn.py icv_mefed

		SVM
		- cd src/model/svm
		- python model_svm.py icv_mefed

		kNN
		- cd src/model/knn
		- python build_model_knn.py icv_mefed

	In each case, the model is saved to the same folder

3. To start real-time face detection:

	python face_recognition.py
