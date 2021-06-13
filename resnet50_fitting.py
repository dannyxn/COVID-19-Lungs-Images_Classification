import tensorflow as tf
from keras.callbacks import EarlyStopping
from keras.layers.preprocessing.image_preprocessing import Rescaling
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.applications import resnet50

from tensorflow.keras.optimizers import SGD


if __name__ == "__main__":
    train_ds = tf.keras.preprocessing.image_dataset_from_directory(
        directory="COVID-19_Radiography_Dataset/Training",
        validation_split=0.2,
        subset='training',
        label_mode='categorical',
        seed=1337,
        image_size=(299, 299),
        batch_size=8,
        shuffle=True,
    )

    val_ds = tf.keras.preprocessing.image_dataset_from_directory(
        directory="COVID-19_Radiography_Dataset/Training",
        validation_split=0.2,
        subset='validation',
        label_mode='categorical',
        seed=1337,
        image_size=(299, 299),
        batch_size=8,
        shuffle=True,
    )

    test_ds = tf.keras.preprocessing.image_dataset_from_directory(
        directory="COVID-19_Radiography_Dataset/Test",
        label_mode='categorical',
        seed=1337,
        image_size=(299, 299),
        batch_size=8,
        shuffle=True,
    )

    base_model = resnet50.ResNet50(weights="imagenet", input_shape=(299, 299, 3), include_top=False, layers=tf.keras.layers)

    data_augmentation = keras.Sequential(
        [
            layers.experimental.preprocessing.RandomFlip("horizontal"),
            layers.experimental.preprocessing.RandomRotation(0.1),
        ]
    )

    for layer in base_model.layers:
        layer.trainable = False

    inputs = keras.Input(shape=(299, 299, 3))
    new_model = data_augmentation(inputs)
    new_model = base_model(new_model, training=False)
    new_model = keras.layers.Flatten()(new_model)
    outputs = keras.layers.Dense(3, activation='softmax')(new_model)
    model = keras.Model(inputs, outputs)

    early_stopping = EarlyStopping(monitor='loss', patience=10, verbose=0, mode='min')
    model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(filepath='checkpoint.hdf5',
        save_weights_only=True, monitor='val_accuracy', mode='max', save_best_only=True)

    model.compile(optimizer=keras.optimizers.Adam(1e-3), loss='categorical_crossentropy', metrics='categorical_accuracy')
    model.summary()

    model.fit(train_ds, epochs=10, callbacks=[early_stopping, model_checkpoint_callback])
    model.save('resnet50_trained')

