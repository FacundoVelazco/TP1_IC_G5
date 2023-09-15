import json
from datetime import datetime
import numpy as np

dias_semana = {
    "monday": 0,
    "tuesday": 1,
    "wednesday": 2,
    "thursday": 3,
    "friday": 4,
}


def getData():
    with open('nasdaq-index.json', 'r') as json_file: data = json.load(json_file)
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


def desnormalize(data, max, min):
    return []


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

    return (train_data, train_labels), ([], [])
