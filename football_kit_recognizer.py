# %%
import os
import sys

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['PYTHONIOENCODING'] = 'utf-8'

import ctypes
import matplotlib.pyplot as plt
import pathlib
import numpy as np
from PIL import Image
import tensorflow as tf
from tensorflow import keras

#sys.stdout.reconfigure(encoding='utf-8')
#sys.stderr.reconfigure(encoding='utf-8')

size_tuple = (150,150)
epochs = 40

# %%
categorias = []
labels = []
imagenes  = []

# %%
categorias =  [archivo for archivo in os.listdir("Imagenes") if not archivo.startswith('.')]
print(categorias)

# %%
x=0
for directorio in categorias:
    imagenes_dir = [imagen for imagen in os.listdir('Imagenes/'+directorio) if not imagen.startswith('.')]
    for imagen in imagenes_dir:
        img = Image.open('Imagenes/'+directorio+'/'+imagen).resize(size_tuple)
        img = np.asarray(img)
        imagenes.append(img)
        labels.append(x)
    x += 1

# %%
print(labels)

# %%
imagenes=np.asanyarray(imagenes)
imagenes.shape
imagenes = imagenes[:,:,:,0]

# %%
model = tf.keras.models.Sequential([
    tf.keras.layers.Flatten(input_shape=size_tuple),
    tf.keras.layers.Dense(128,activation='relu'),
    tf.keras.layers.Dropout(0,25),
    tf.keras.layers.Dense(len(categorias),activation='softmax')
])

# %%
model.compile(optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy'])

# %%
labels = np.array(labels)

# %%
history = model.fit(imagenes, labels, epochs=epochs)

# %%
correcto = 0
incorrecto = 0
root_dir = 'Validacion' 

for subdir, dirs, files in os.walk(root_dir):
    for file in files:
        if not file.startswith('.'):
            im = Image.open(os.path.join(subdir, file)).resize(size_tuple)
            im = np.asarray(im)
            im = im[:,:,0]
            im = np.asarray([im])
            test=im
            predicciones = model.predict(test)  
            print(categorias[np.argmax(predicciones[0])])
            actual_categoria = os.path.basename(subdir)
            if(categorias[np.argmax(predicciones[0])] == actual_categoria):
                correcto += 1
            else:
                incorrecto += 1

total=correcto+incorrecto
porcentajeCorrecto= (correcto*100)/total
print('Validaciones Correctas: '+ str(correcto))
print('Validaciones Incorrectas: '+ str(incorrecto))


# %%
nombre_imagen_de_prueba = os.listdir("Prediccion")[0]
imagen_prueba = "Prediccion/" + nombre_imagen_de_prueba
im = Image.open(imagen_prueba).resize(size_tuple)
im = np.asarray(im)
im = im[:,:,0]
im = np.asarray([im])
im.shape
test=im

# %%
prediccion = model.predict(test)  
print(prediccion)

# %%
img = Image.open(imagen_prueba).resize(size_tuple)
plt.figure()
plt.imshow(img)
plt.show()

# %%
categorias[np.argmax(prediccion[0])]

# %%
model.summary()

# %%
acc = history.history['accuracy']
#val_acc = history.history['val_accuracy']

loss = history.history['loss']
#val_loss = history.history['val_loss']

epochs_range = range(epochs)

plt.figure(figsize=(8, 8))
plt.subplot(1, 2, 1)
plt.plot(epochs_range, acc, label='Training Accuracy')
#plt.plot(epochs_range, val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')

plt.subplot(1, 2, 2)
plt.plot(epochs_range, loss, label='Training Loss')
#plt.plot(epochs_range, val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')
plt.show()


ctypes.windll.user32.MessageBoxW(0, "Validaciones correctas: "+str(correcto)+"\nValidaciones incorrecta: "+str(incorrecto)+"\nPorcentaje Correctas: "+str(porcentajeCorrecto)+"%\nPrediccion de imagen: "+str(categorias[np.argmax(prediccion[0])]), "Resultados", 0)
# %%
#tf.keras.utils.plot_model(model,to_file='rna.png', show_shapes=True,rankdir='LR')


