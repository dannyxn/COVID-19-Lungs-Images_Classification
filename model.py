import tensorflow as tf
from keras.callbacks import EarlyStopping
from keras.layers.preprocessing.image_preprocessing import Rescaling
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.optimizers import SGD

# TODO Rozbij sobie dane na dwa katalogi - Test i Training ( do testu wziąłem około 10 % wszystkich danych )
# TODO Zbiłem dwa katalogi danych w jeden - Non-Covid, żeby były 3 klasy i teoretycznie lepsze wyniki, więc sobie też rozbij
# TODO Dzisiaj tego za bardzo nie zrobisz, bo musi się model wytrenować ale będzie trzeba zrobić predykcję na test secie
# TODO Zrobić macierz pomyłek i skonfrontować ją z tą macierzą z artykułu.
# Jak będzie się nam już fajnie trenowało i będzie wszystko finito, to wytrenujemy to wszystko nie tylko dla res neta ale też dla innych

# "Do analizy proszę wykorzystać zarówno macierz pomyłek jak i obliczenie skuteczności, wrażliwości itp. miar jakości klasyfikatora".
# TODO sprawdź co to wrażliwość i to jakoś wylicz, no i jak masz jakieś pomysły na pomiary klasyfikacji to zastosuj
# TODO ewentualnie porównaj optimizery i learning rate'y w pierwszym ficie.
# TODO ewentualnie wypróbuj odblokowanie innego zakresu warstw sieci w fine-tuningu.
train_ds = tf.keras.preprocessing.image_dataset_from_directory(
    directory="COVID-19_Radiography_Dataset/Training",
    validation_split=0.2,
    subset='training',
    label_mode='categorical',
    seed=1337,
    image_size=(299, 299),
    batch_size=32,
    shuffle=True,
)
val_ds = tf.keras.preprocessing.image_dataset_from_directory(
    directory="COVID-19_Radiography_Dataset/Training",
    validation_split=0.2,
    subset='validation',
    label_mode='categorical',
    seed=1337,
    image_size=(299, 299),
    batch_size=32,
    shuffle=True,
)
test_ds = tf.keras.preprocessing.image_dataset_from_directory(
    directory="COVID-19_Radiography_Dataset/Test",
    label_mode='categorical',
    seed=1337,
    image_size=(299, 299),
    batch_size=32,
    shuffle=True,
)

base_model = ResNet50(weights=None, input_shape=(299, 299, 3), include_top=False)

data_augmentation = keras.Sequential(
    [
        layers.experimental.preprocessing.RandomFlip("horizontal"),
        layers.experimental.preprocessing.RandomRotation(0.1),
    ]
)

# Wstepny trening dodatkowych wartw
# warstwy sieci pretrenowanej są zablokowane
base_model.trainable = False
inputs = keras.Input(shape=(299, 299, 3))
x = data_augmentation(inputs)
x = Rescaling(1.0 / 255)(x)
x = base_model(x, training=False)
x = keras.layers.Flatten()(x)
outputs = keras.layers.Dense(3, activation='softmax')(x)
model = keras.Model(inputs, outputs)
#
earlyStopping = EarlyStopping(monitor='val_loss', patience=10, verbose=0, mode='min')
model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(filepath='checkpoint',
    monitor='val_categorical_accuracy', mode='max', save_best_only=True)

model.compile(optimizer=keras.optimizers.Adam(1e-3), loss='categorical_crossentropy',
              metrics=['categorical_accuracy', 'val_categorical_accuracy'])
model.fit(train_ds, epochs=10, callbacks=[earlyStopping, model_checkpoint_callback])
model.save('trained_model')

# ---------------------- Fine tuning ----------------------

# model = keras.models.load_model('trained_model')
for layer in model.layers[:]:
    layer.trainable = True

model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(filepath=f'final-checkpoint',
    monitor='val_categorical_accuracy', mode='max', save_best_only=True)

# Ponowny trening z niskim learning rate
model.compile(optimizer=SGD(learning_rate=0.0001, momentum=0.9), loss='categorical_crossentropy',
              metrics=['categorical_accuracy'])
model.fit(train_ds, epochs=20, validation_data=val_ds, callbacks=[earlyStopping, model_checkpoint_callback])  # na razie 3 epochy, później wiecej
model.save('fine_tuned_model')