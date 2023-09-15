import json
from datetime import datetime
import numpy as np
from keras import layers, models, optimizers

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


def genTrainDataFourDaysBf(data):
    train_labels = []
    train_data = []
    for i, obj in enumerate(data):
        if i >= 4:
            # Obtener los valores "open" de los cuatro días anteriores
            open_values = []
            for j in range(i - 4, i): open_values.append(data[j]["open"])
            # Agregar el valor del día de la semana actual en formato binario
            train_data.append(open_values + data[i]["day_of_week"])
            train_labels.append(data[i]["open"])
    trainBorder = len(train_data)//100 * 60
    validationBorder = len(train_data)//100 * 80
    return (train_data[:trainBorder], train_labels[:trainBorder]), (train_data[trainBorder:validationBorder], train_labels[trainBorder:validationBorder]), (
    train_data[validationBorder:], train_labels[validationBorder:])


def build_model_regression(input_data_shape):
    model = models.Sequential()
    model.add(layers.Dense(20, activation='relu', input_shape=[input_data_shape, ]))
    model.add(layers.Dense(40, activation='relu'))
    model.add(layers.Dense(20, activation='relu'))
    model.add(layers.Dense(1))
    model.compile(optimizer='rmsprop', loss='mse', metrics=['mae'])
    return model
