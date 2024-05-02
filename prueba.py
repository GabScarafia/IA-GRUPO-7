import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import tensorflow as tf

print("Hola Mundo")
print("tensorflow version:", tf.__version__)