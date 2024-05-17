#Imports
import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['PYTHONIOENCODING'] = 'utf-8'

import matplotlib.pyplot as plt
import pathlib
from PIL import Image
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential

#region Basico
#Lo agrego a una funcion no pq sea necesario sino por si despues necesito definir funciones extras
#esas funciones extras tambien podrian ir en otros archivos. (a todo caso lo cambiamos)
def main():
    # Con esto lo que estoy intentando es tener el repositorio de imagenes a mano.
    data_dir = pathlib.Path('Imagenes')
    # Para mas tarde )? -Gaby
    list_dataset = tf.data.Dataset.list_files(str(data_dir/'*/*.jpg'))
    #Cuento e imprimo
    image_count = len(list(data_dir.glob('*/*.jpg')))
    print("------------------------------------------")
    print("Cantidad de imagenes del directorio: "+ str(image_count))
    print("------------------------------------------")

    #Reconocer grupo de imagenes
    river = list(data_dir.glob('River/*'))
    boca = list(data_dir.glob('Boca/*'))
    #Abre la imagen que reconocio: 
    #image = Image.open(str(river[0]))
    #no se que tan util es excepto para saber en donde estamos parado...
    #image.show()
#endregion

#region Normalización Datos
    #Definicion de datos para entrenar
    batch_size = 32
    img_height = 180 
    img_width = 180

    #Definicion de entrenamiento
    train_dataset = tf.keras.utils.image_dataset_from_directory(
        data_dir,
        validation_split=0, #esto va 0 pq creo que nosotros vamos a darle input cuando vayamos a validar
        seed=123,
        image_size=(img_height, img_width),
        batch_size=batch_size)
    
    #con esto deberia ver los N directorios que podriamos (serian nuestras clases)
    class_names = train_dataset.class_names
    print("----------------------------")
    print("Clases que identifico:")
    print(class_names)
    print("----------------------------")

    #Preparación del dataset para el entrenamiento:
    AUTOTUNE = tf.data.AUTOTUNE
    train_dataset = train_dataset.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
    #normalization_layer = layers.Rescaling(1./255)
#endregion

#region RNA
    #Aca se agregan las capas de nodos de la RNA 
    num_classes = len(class_names)
    model = Sequential([
        layers.Input(shape=(img_height, img_width, 3)), 
        layers.Rescaling(1./255),
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dense(num_classes)
    ])

    model.compile(optimizer='adam',
                loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                metrics=['accuracy'])    
    model.summary()
#endregion

#region Archivo de Validacion
    val_dir = pathlib.Path('Validacion') #--> esto no encuentra nada, pq?
    # list_dataset = tf.data.Dataset.list_files(str(val_dir/'*/*.jpg'))
    # image_count2 = len(list(val_dir.glob('*/*.jpg')))

    val_dataset =  tf.keras.utils.image_dataset_from_directory(
        val_dir,
        image_size=(img_height, img_width),
        batch_size=batch_size)
# endregion



#region Entrenamiento
    #Aca se entrena el proyecto 
    #Esto ya deberia andar si el validation_data (region de arriba) anda
    epochs=10 
    history = model.fit(
        train_dataset,
        validation_data=val_dataset, 
        epochs=epochs
    )

#endregion

    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']

    loss = history.history['loss']
    val_loss = history.history['val_loss']

    epochs_range = range(epochs)

    plt.figure(figsize=(8, 8))
    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, acc, label='Training Accuracy')
    plt.plot(epochs_range, val_acc, label='Validation Accuracy')
    plt.legend(loc='lower right')
    plt.title('Training and Validation Accuracy')

    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, loss, label='Training Loss')
    plt.plot(epochs_range, val_loss, label='Validation Loss')
    plt.legend(loc='upper right')
    plt.title('Training and Validation Loss')
    plt.show()
#Corro el codigo:
main()