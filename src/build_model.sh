#!/bin/bash

model=$1
if [ $model == "cnn" ];
then
eval `cd ./model/cnn/ && python build_model_cnn.py $2`
elif [ $model == "knn" ];
then
eval `cd ./model/knn/ && python build_model_knn.py $2`
else
eval `cd ./model/svm/ && python ./model/svm/build_model_svm.py $2`
fi