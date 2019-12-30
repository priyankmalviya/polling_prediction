from keras.applications.nasnet import NASNetLarge
from keras.applications.nasnet import preprocess_input
import helpers

def train(path,BATCH_SIZE,IMG_SIZE,n_classes,save_path):
    datagenerator = helpers.generate_data(BATCH_SIZE,IMG_SIZE,preprocess_input,path)
    base_model = NASNetLarge(input_shape=(IMG_SIZE, IMG_SIZE,3),
                                               include_top=False,
                                               weights='imagenet')

    compiled_model = helpers.transfer_learning(base_model,n_classes)
    helpers.fit_model(compiled_model,datagenerator,save_path)
