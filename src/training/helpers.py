from keras.preprocessing.image import ImageDataGenerator
from keras.layers import Dense, GlobalAveragePooling2D
from keras.models import Model
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping, ModelCheckpoint


def generate_data(BATCH_SIZE,IMG_SIZE,preprocess_input,path):
    train_datagen = ImageDataGenerator(
        preprocessing_function = preprocess_input,
        validation_split = 0.2)

    train_data_dir = path
    train_generator = train_datagen.flow_from_directory(
        train_data_dir,
        shuffle=False,
        target_size=(IMG_SIZE, IMG_SIZE),
        batch_size=BATCH_SIZE)

    return train_generator

def transfer_learning(base_model,n_classes):
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(1024, activation='relu')(x)
    predictions = Dense(n_classes, activation='softmax')(x)
    model = Model(inputs=base_model.input, outputs=predictions)
    for layer in base_model.layers:
        layer.trainable = False

    model.compile(optimizer=Adam(lr = 0.001), loss='categorical_crossentropy', metrics= ['accuracy'])
    return model

def initiate_callback(model_path):
    callbacks = [EarlyStopping(monitor='loss', patience=5),
             ModelCheckpoint(filepath=model_path, monitor='loss', save_best_only=True)]
    return callbacks

def fit_model(model,data_generator,model_save_path):
    cb = initiate_callback(model_save_path)
    model.fit_generator(data_generator,epochs=20,validation_steps=20,steps_per_epoch=100,callbacks= cb)
