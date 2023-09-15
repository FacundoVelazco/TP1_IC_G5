import matplotlib.pyplot as plt
import tensorflow as tf
import utils
import numpy as np

raw_data = utils.getData()

max_open = max(item["open"] for item in raw_data)
min_open = min(item["open"] for item in raw_data)

normalized_data = utils.normalize(raw_data,max_open,min_open)

(train_data_4days,train_labels_4days),(test_data_4days,test_labels_4days) = utils.genTrainDataFourDaysBf(normalized_data)

oculta1 = tf.keras.layers.Dense(units=20,input_shape=[9, ])
oculta2 = tf.keras.layers.Dense(units=40)
oculta3 = tf.keras.layers.Dense(units=20)
salida = tf.keras.layers.Dense(units=1)
modelo = tf.keras.Sequential([oculta1, oculta2, oculta3, salida])

modelo.compile(
    optimizer=tf.keras.optimizers.Adam(0.1),
    loss='mean_squared_error',
    metrics='mse'
)
print("Comenzando entrenamiento...")
history = modelo.fit(train_data_4days, train_labels_4days, epochs=5, verbose=False)

print("Metodo entrenado!")

print(history)