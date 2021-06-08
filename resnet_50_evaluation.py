import tensorflow as tf
import tensorflow_addons as tfa
import numpy as np
from tensorflow import keras


model = keras.models.load_model('trained_model')

test_ds = tf.keras.preprocessing.image_dataset_from_directory(
    directory="COVID-19_Radiography_Dataset/Test",
    label_mode='categorical',
    seed=1337,
    image_size=(299, 299),
    batch_size=32,
    shuffle=True,
)

images, labels = tuple(zip(*test_ds))  # extract separately images and labels

# get rid of batches
labels = np.vstack(labels)
images = np.vstack(images)
predictions = model.predict(images)


metric = tfa.metrics.MultiLabelConfusionMatrix(num_classes=3)
metric.update_state(labels, predictions)
result = metric.result()
