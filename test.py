import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from OriginalData import (target_data, train_data, min_open, max_open)

from sklearn.metrics import mean_squared_error

train = np.array(train_data[:70], dtype=float)
test = np.array(train_data[70:], dtype=float)
target_train = np.array(target_data[:70], dtype=float)
target_test = np.array(target_data[70:], dtype=float)

#  Capas densas son aquellas que tienen conexiones con todas la neuronas
# unit -> unidades o neuronas de la capa
# input_shape cantidad de neuronas de entrada???

# capa = tf.keras.layers.Dense(units=1, input_shape=[1])
# modelo= tf.keras.Sequential([capa])

oculta1 = tf.keras.layers.Dense(units=20,input_shape=[9, ])
oculta2 = tf.keras.layers.Dense(units=40)
oculta3 = tf.keras.layers.Dense(units=20)
salida = tf.keras.layers.Dense(units=1)
modelo = tf.keras.Sequential([oculta1, oculta2, oculta3, salida])

modelo.compile(
    optimizer=tf.keras.optimizers.Adam(0.1),
    loss='mean_squared_error',
    metrics='accuracy'
)
print("Comenzando entrenamiento...")
historial = modelo.fit(train, target_train, epochs=1000, verbose=False)

print("Metodo entrenado!")

# plt.xlabel("# EPOCA")
# plt.ylabel("# Maginitud de perdidad")
# plt.plot(historial.history["loss"])
# plt.show()


print("prediccion")
resultado = modelo.predict(test)

mse = mean_squared_error(target_test, resultado)

print('Error cuadratico medio: ' + str(mse))
print("=====================================")

x_list = []
x_list2 = []
for i in range(len(target_test)):
    print(f"Posición {i}:")
    print("Predicción:", resultado[i] * (max_open - min_open) + min_open)
    x_list.append(resultado[i] * (max_open - min_open) + min_open)
    print("Objetivo:", target_test[i] * (max_open - min_open) + min_open)
    x_list2.append(target_test[i] * (max_open - min_open) + min_open)
    print("======================")
plt.plot(x_list)
plt.plot(x_list2)
plt.show()
# print("el resultado es: " + str(resultado) + " y el target es " + target_test[0])

# print("variables internas del modelo")
# print(oculta1.get_weights())
# print(oculta2.get_weights())
# print(salida.get_weights)
