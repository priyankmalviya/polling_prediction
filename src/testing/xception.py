from keras.applications.xception import Xception
from keras.applications.xception import preprocess_input
import helpers


def train(path,BATCH_SIZE,IMG_SIZE,n_classes,save_path):
    datagenerator = helpers.generate_data(BATCH_SIZE,IMG_SIZE,preprocess_input,path)
    base_model = Xception(input_shape=(IMG_SIZE, IMG_SIZE,3),
                                               include_top=False,
                                               weights='imagenet')

    compiled_model = helpers.transfer_learning(base_model,n_classes)
    helpers.fit_model(compiled_model,datagenerator,save_path)
