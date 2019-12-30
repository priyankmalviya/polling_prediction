# Polling_Prediction
This repo deals with the problem of image classification by taking the outcomes of multiple CNN models and predicting the final result from a cumulative decision of all the models.

This work is focused on the Imagenet Dataset. Seven different pre-trained models were used as following :

* DenseNet
* Inception-ResNet-v2
* MobileNet-v2
* NASNet-A-Large
* NASNet-A-Mobile
* VGG-16
* Xception

## Approach
First step was to download the data of the imagenet challenge. To know more on this refer to :-
[Github Repo](https://github.com/mf1024/ImageNet-Datasets-Downloader.git)

Next step was to train these pre trained models on the dataset in hand using transfer learning. To know more about transfer learning following links can be useful :-
* (https://www.tensorflow.org/tutorials/images/transfer_learning)
* (https://keras.io/applications/)

Once our models were trained, the next step was to test them and get accuracy for each of the models. 

Finally once we had all the models trained, our goal was to predict class of each image by taking a poll of each of the trained model on each image and see the change in result. 

