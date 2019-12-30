import helpers
from keras.applications.mobilenet_v2 import preprocess_input

def test(batch_size,img_size,test_images_path,model_path):
    test_generator = helpers.generate_test_data(batch_size,img_size,preprocess_input,test_images_path)
    model = helpers.get_model(model_path)
    yhat = helpers.predict(model,test_generator)
    yhat = helpers.get_best_guess(yhat)
    acc_mnet = helpers.get_accuracy(yhat,test_generator)
    class_mapping =  helpers.get_class_mapping(test_generator)
    final_pred = helpers.final_prediction(class_mapping,yhat)
    df_mnet = helpers.create_dataframe(['image','class','mnet'])
    df_mnet = helpers.add_records(df_mnet,test_generator,final_pred,'mnet')

    return df_mnet,acc_mnet
