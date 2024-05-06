#Imports
import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import tensorflow as tf
import pathlib
#OS commands
os.system('cls')

# Con esto lo que estoy intentando es tener el repositorio de imagenes a mano.
data_dir = pathlib.Path('Imagenes')
#list_ds = tf.data.Dataset.list_files(str(data_dir/'*/*.jpg'))
image_count = len(list(data_dir.glob('*/*.jpg')))
print("Cantidad de imagenes del directorio: "+ str(image_count))