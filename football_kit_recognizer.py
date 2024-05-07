#Imports
import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import tensorflow as tf
import pathlib
from PIL import Image

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
    image = Image.open(str(river[0]))
    #Abre la imagen que reconocio: 
    #no se que tan util es excepto para saber en donde estamos parado...
    #image.show()
#endregion

#region Proceso
    #Definicion de datos para entrenar
    batch_size = 32
    img_height = 180 #Qsy
    img_width = 180
    #Definicion de entrenar
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

    #Como seguir: habria que normalizar los datos y generar las distintas capas de RNA.
    #...
#endregion

#Corro el codigo:
main()