import numpy as np
import tensorflow as tf
from keras.callbacks import EarlyStopping
from tensorflow import keras
from tensorflow.keras.optimizers import SGD
from tensorflow.keras import layers

if __name__ == "__main__":
    train_ds = tf.keras.preprocessing.image_dataset_from_directory(
        directory="COVID-19_Radiography_Dataset/Training",
        validation_split=0.2,
        subset='training',
        label_mode='categorical',
        seed=1337,
        image_size=(299, 299),
        batch_size=16,
        shuffle=True,
    )

    val_ds = tf.keras.preprocessing.image_dataset_from_directory(
        directory="COVID-19_Radiography_Dataset/Training",
        validation_split=0.2,
        subset='validation',
        label_mode='categorical',
        seed=1337,
        image_size=(299, 299),
        batch_size=16,
        shuffle=True,
    )

    model = tf.keras.models.load_model("resnet50_trained")

    for layer in model.layers:
        layer.trainable = True

        if layer.name.startswith('bn'):
            layer.call(layer.input, training=False)

    model.summary()

    early_stopping = EarlyStopping(monitor='loss', patience=10, verbose=0, mode='min')
    model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(filepath=f'final-checkpoint.hdf5',
                                                                   save_weights_only=True, monitor='categorical_accuracy',
                                                                   mode='max', save_best_only=True)

    # Ponowny trening z niskim learning rate
    model.compile(optimizer=SGD(learning_rate=1e-6, momentum=0.9), loss='categorical_crossentropy',
                  metrics='categorical_accuracy')

    model.fit(train_ds, epochs=20, validation_data=val_ds,
              callbacks=[early_stopping, model_checkpoint_callback])

    model.save('fine_tuned_model')

