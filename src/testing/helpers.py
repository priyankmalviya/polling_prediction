from keras.preprocessing.image import ImageDataGenerator
from keras.models import load_model
import pandas as pd


def generate_test_data(batch_size,img_size,preprocess_input,path):
    test_datagen = ImageDataGenerator(
    preprocessing_function = preprocess_input)

    test_data_dir = path
    test_generator = test_datagen.flow_from_directory(
        test_data_dir,
        shuffle=False,
        target_size=(img_size,img_size),
        batch_size=batch_size)

    return test_generator

def get_model(model_path):
    model = load_model(model_path)
    return model

def predict(model,test_generator):
    yhat = model.predict_generator(test_generator,verbose = 1)
    return yhat

def get_best_guess(yhat):
    yhat = yhat.argmax(axis = 1)
    return yhat

def get_accuracy(yhat,test_generator):
    return (sum(yhat == test_generator.classes)/len(yhat))

def get_class_mapping(test_generator):
    dc ={}
    original_mapping = test_generator.class_indices
    for k,v in original_mapping.items():
      dc[v] = k
    return dc

def final_prediction(dc, yhat):
    final_pred = [dc[k] for k in yhat]
    return final_pred

def create_dataframe(col_lst):
    df = pd.DataFrame(columns = col_lst)
    return df

def get_filenames(test_generator):
    return test_generator.filenames

def add_records(df,test_generator,final_pred,model_prefix):
    filenames = get_filenames(test_generator)
    for i in range(0,len(filenames)):
        row = {'image':filenames[i].split('/')[1],'class':filenames[i].split('/')[0],model_prefix:final_pred[i]}
        df = df.append(row,ignore_index = True)

    return df
