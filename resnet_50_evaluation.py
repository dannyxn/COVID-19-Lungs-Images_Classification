import tensorflow as tf
import numpy as np
from tensorflow import keras
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix


def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix'):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    cmap = plt.get_cmap("Blues")
    import itertools
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    model = keras.models.load_model('fine_tuned_model')

    test_ds = tf.keras.preprocessing.image_dataset_from_directory(
        directory="COVID-19_Radiography_Dataset/Test",
        label_mode='categorical',
        seed=1337,
        image_size=(299, 299),
        batch_size=16,
        shuffle=True,
    )

    images, labels = tuple(zip(*test_ds))  # extract separately images and labels

    # get rid of batches
    labels = np.vstack(labels)
    images = np.vstack(images)
    predictions = model.predict(images)

    y_true = np.argmax(labels, axis=1)
    y_pred = np.argmax(predictions, axis=1)

    cnf_matrix = confusion_matrix(y_true, y_pred, labels=[0, 1, 2])

    plot_confusion_matrix(cnf_matrix, ["Covid", "Non Covid", "Normal"])