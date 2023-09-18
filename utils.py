import json
from datetime import datetime
from keras import layers, models
import matplotlib.pyplot as plt

dias_semana = {
    "monday": 0,
    "tuesday": 1,
    "wednesday": 2,
    "thursday": 3,
    "friday": 4,
}


def getData(path):
    with open(path, 'r') as json_file: data = json.load(json_file)
    result = list(map(lambda x: {"open": x["Open"], "date": x["Date"]}, data))
    return result


def normalize(data, max, min):
    normalized_data = []
    for item in data:
        # Open value
        normalized_open = (item["open"] - min) / (max - min)
        # Day of week
        date_obj = datetime.strptime(item["date"], "%Y-%m-%d")
        dia_semana = date_obj.strftime("%A").lower()
        vector_binario = [0] * 5
        if dia_semana in dias_semana:
            vector_binario[dias_semana[dia_semana]] = 1
        normalized_item = {"open": normalized_open, "day_of_week": vector_binario}
        normalized_data.append(normalized_item)

    return normalized_data


def desnormalizeList(data, max, min):
    # como una lista
    desnormalized_items = []
    for item in data:
        desnormalized_open = item[0] * (max - min) + min
        desnormalized_items.append(desnormalized_open)
    return desnormalized_items


def desnormalize(data, max, min):
    # como una lista
    desnormalized_items = []
    for item in data:
        desnormalized_open = item * (max - min) + min
        desnormalized_items.append(desnormalized_open)
    return desnormalized_items


def genTrainData4DaysBf(data):
    train_labels = []
    train_data = []
    for i, obj in enumerate(data):
        if i >= 4:
            # Obtener los valores "open" de los cuatro días anteriores
            open_values = []
            for j in range(i - 4, i):
                open_values.append(data[j]["open"])
            # Agregar el valor del día de la semana actual en formato binario
            train_data.append(open_values + data[i]["day_of_week"])
            train_labels.append(data[i]["open"])
    trainBorder = int((len(train_data) / 100) * 60)
    validationBorder = int((len(train_data) / 100) * 80)
    return (train_data[:trainBorder], train_labels[:trainBorder]), (
        train_data[trainBorder:validationBorder], train_labels[trainBorder:validationBorder]), (
        train_data[validationBorder:], train_labels[validationBorder:])


def genTrainData9DaysBf(data):
    train_labels = []
    train_data = []
    for i, obj in enumerate(data):
        if i >= 9:
            # Obtener los valores "open" de los cuatro días anteriores
            open_values = []
            for j in range(i - 9, i):
                open_values.append(data[j]["open"])
            # Agregar el valor del día de la semana actual en formato binario
            train_data.append(open_values + data[i]["day_of_week"])
            train_labels.append(data[i]["open"])
    trainBorder = int((len(train_data) / 100) * 60)
    validationBorder = int((len(train_data) / 100) * 80)
    return (train_data[:trainBorder], train_labels[:trainBorder]), (
        train_data[trainBorder:validationBorder], train_labels[trainBorder:validationBorder]), (
        train_data[validationBorder:], train_labels[validationBorder:])


def build_model_regression(input_data_shape):
    model = models.Sequential()
    model.add(layers.Dense(20, activation='softmax', input_shape=[input_data_shape, ]))
    model.add(layers.Dense(40, activation='softmax'))
    model.add(layers.Dense(20, activation='softmax'))
    model.add(layers.Dense(1, activation='sigmoid'))
    model.compile(optimizer='rmsprop', loss='mse', metrics=['mae'])
    return model


def plotResults(resultsList):
    fig, ax = plt.subplots()
    # Configurar el alto de la figura (Valor = 5), ancho de la figura (valor = 10), color ('lightgrey') y título ('Lluvia media mensual')
    fig.suptitle('Predicción de valores Nasdaq', fontsize=20)
    fig.set_facecolor('lightgrey')
    fig.set_figheight(5)
    fig.set_figwidth(10)
    # Etiquetas de los ejes x, y
    fontX = {'family': 'serif', 'color': 'darkblue', 'size': 10}
    fontY = {'family': 'serif', 'color': 'darkred', 'size': 10}
    ax.set_xlabel('Meses', fontX)
    ax.set_ylabel('Valores', fontY)
    # Etiquetas de las reglas
    ax.tick_params(axis='x', labelsize=14, labelrotation=90)
    ax.tick_params(axis='y', labelsize=13)
    # Grilla horizontal, color gris, linea de puntos, espesor 1.
    ax.grid(axis='y', color='grey', linestyle=':', linewidth=1)

    for list in resultsList:
        ax.plot(list, linestyle='--', linewidth=2, marker='o', markersize=5)

    ax.legend(['Valores reales', 'Predicción'], loc='upper center')

    plt.show()
