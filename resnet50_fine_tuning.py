import tensorflow as tf
from keras.callbacks import EarlyStopping
from tensorflow import keras
from tensorflow.keras.optimizers import SGD

if __name__ == "__main__":

    train_ds = tf.keras.preprocessing.image_dataset_from_directory(
        directory="COVID-19_Radiography_Dataset/Training",
        validation_split=0.2,
        subset='training',
        label_mode='categorical',
        seed=1337,
        image_size=(299, 299),
        batch_size=128,
        shuffle=True,
    )

    test_ds = tf.keras.preprocessing.image_dataset_from_directory(
        directory="COVID-19_Radiography_Dataset/Training",
        validation_split=0.2,
        subset='validation',
        label_mode='categorical',
        seed=1337,
        image_size=(299, 299),
        batch_size=128,
        shuffle=True,
    )

    model = keras.model.load_model("resnet50_trained")
    for layer in model.layers[:]:
        layer.trainable = True

    early_stopping = EarlyStopping(monitor='val_loss', patience=10, verbose=0, mode='min')
    model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(filepath=f'final-checkpoint.hdf5',
                                                                   save_weights_only=True, monitor='val_accuracy',
                                                                   mode='max', save_best_only=True)

    # Ponowny trening z niskim learning rate
    model.compile(optimizer=SGD(lr=0.0001, momentum=0.9), loss='categorical_crossentropy',
                  metrics='categorical_accuracy')

    model.fit(train_ds, epochs=20, validation_data=test_ds,
              callbacks=[early_stopping, model_checkpoint_callback])  # na razie 3 epochy, później wiecej
    model.save_weights("model.h5")
    model.save('fine_tuned_model')

