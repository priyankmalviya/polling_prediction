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

## Dataset
Our dataset was downloaded using the same downloader script as referred above. 
The data consisted of 20 different classes with 50 images per class. 

For training the models 35 images were used while rest 15 images were used for testing.


## Getting Started

It is strongly recommended to run this project on any cloud platform as it requires heavy computations as it involves training the model on images. We used Google Colab to run this project. To know more about google colab refer to this link :-

* (https://towardsdatascience.com/getting-started-with-google-colab-f2fff97f594c)
* (https://colab.research.google.com/notebooks/welcome.ipynb#recent=true)

If you want to run this on your local machine, it should have GPU to make it run faster.

### Requirements
* Python 3 [How to install](http://ubuntuhandbook.org/index.php/2017/07/install-python-3-6-1-in-ubuntu-16-04-lts/)
* Keras [How to install](https://github.com/hsekia/learning-keras/wiki/How-to-install-Keras-to-Ubuntu-18.04)




## Execution on Google Colab

To execute this project on google colab. You need a gmail account. Upload the source from this repo on your google drive. Login to your google colab and run the downloader script mentioned above to download imagenet data. 

Once the data is downloaded just run the script train.py to train the seven models :-
Eg. Running the script on google colab 
```
!python /content/drive/My\ Drive/source_train/train.py /path/to/training_images
```

To get the prediction run the script test.py under the folder source_test.
Eg. Running the script on google colab
```
!python /content/drive/My\ Drive/source_test/test.py /path/to/test_images
```

## Things to note

To make the whole project more efficient we are reading the images in batches to avoid overhear of I/O operation. 
In our scripts we are using batch size of 64 images at a time. To change this while running the project go to the scripts *train.py* and *test.py* and change the global variable *BATCH_SIZE* as per your resource availability.

In this project we used 20 different classes to test the results. In case you need to change the number of classes go to the script *train.py* and modify the global variable *TRAIN_CLASSES* as per your need.


## Results

For Dataset 1:
Number of classes - 20
Images per class - 50
 
|Model Type|Accuracy|
|---|---|
|DenseNet|86.93|
|Inception-ResNet-v2|92.96|
|MobileNet-v2|87.43|
|NASNet-A-Large|95.47|
|NASNet-A-Mobile|90.20|
|VGG-16|86.18|
|Xception|95.47|
|Cumulative Polling|96.23|

For Dataset 2:
Number of classes - 25
Images per class - 50
 
|Model Type|Accuracy|
|---|---|
|DenseNet|82.92|
|Inception-ResNet-v2|94.74|
|MobileNet-v2|85.93|
|NASNet-A-Large|97.75|
|NASNet-A-Mobile|93.95|
|VGG-16|91.75|
|Xception|95.73|
|Cumulative Polling|98.23|
