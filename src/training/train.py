from keras.preprocessing.image import ImageDataGenerator
from keras.layers import Dense, GlobalAveragePooling2D
from keras.models import Model
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping, ModelCheckpoint
import densenet as dnet
import inceptionresnetv2 as ipnetV2
import mobilenetv2 as mnetV2
import nasnetlarge as nnet_L
import nasnetmobile as nnet_M
import vgg16
import xception as xcept

BATCH_SIZE = 64
TRAIN_CLASSES = 20
PATH = '/content/train_images'

if __name__ == '__main__':
    dnet.train(PATH,BATCH_SIZE,224,TRAIN_CLASSES,'/content/models/densenet.h5')
    ipnetV2.train(PATH,BATCH_SIZE,299,TRAIN_CLASSES,'/content/models/ipnet.h5')
    mnetV2.train(PATH,BATCH_SIZE,224,TRAIN_CLASSES,'/content/models/mnet.h5')
    nnet_L.train(PATH,BATCH_SIZE,331,TRAIN_CLASSES,'/content/models/nnet_L.h5')
    nnet_M.train(PATH,BATCH_SIZE,224,TRAIN_CLASSES,'/content/models/nnet_M.h5')
    vgg16.train(PATH,BATCH_SIZE,224,TRAIN_CLASSES,'/content/models/vgg.h5')
    xcept.train(PATH,BATCH_SIZE,299,TRAIN_CLASSES,'/content/models/xception.h5')
