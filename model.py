import tensorflow as tf
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
x = keras.layers.GlobalAveragePooling2D()(x)
x = Dense(1024, activation='relu')(x)
outputs = keras.layers.Dense(3, activation='softmax')(x)
model = keras.Model(inputs, outputs)

model.compile(optimizer=keras.optimizers.Adam(1e-3),
              loss='categorical_crossentropy',
              metrics='categorical_accuracy')
model.fit(train_ds, epochs=1, validation_data=val_ds)  # Na razie 1 epoch, później zwiekszymy
model.save('trained_model')

# ---------------------- Fine tuning ----------------------
# Odblokowujemy warstwy pretrenowanego modelu, zaczynając od 250-tej
# to gwałtownie zwiększa dofitowanie
for layer in model.layers[:249]:
    layer.trainable = False
for layer in model.layers[249:]:
    layer.trainable = True

# Ponowny trening z niskim learning rate
model.compile(optimizer=SGD(lr=0.0001, momentum=0.9), loss='categorical_crossentropy', metrics='categorical_accuracy')
model.fit(train_ds, epochs=3, validation_data=val_ds)  # na razie 3 epochy, później wiecej
model.save('fine_tuned_model')
