import matplotlib.pyplot as plt
import keras
import utils
import tradingBot

path = 'nasdaq-index-original.json'

raw_data = utils.getData(path)

max_open = max(item["open"] for item in raw_data)
min_open = min(item["open"] for item in raw_data)

normalized_data = utils.normalize(raw_data, max_open, min_open)

((train_data_4days, train_labels_4days), (validation_data_4days, validation_labels_4days),(test_data_4days, test_labels_4days)) = utils.genTrainData4DaysBf(normalized_data)
model = utils.build_model_regression(len(train_data_4days[0]))

print("Comenzando entrenamiento...")

history = model.fit(train_data_4days, train_labels_4days, epochs=50,
                    validation_data=(validation_data_4days, validation_labels_4days),
                    verbose=False)

print("Metodo entrenado!")

#loss = history.history['loss']
#history_test = model.evaluate(test_data_4days, test_labels_4days)
predicted_values = model.predict(test_data_4days)
predicted_values = utils.desnormalizeList(predicted_values, max_open, min_open)
real_values = utils.desnormalize(test_labels_4days, max_open, min_open)

(simu_balance,simu_stocks) = tradingBot.simulation(real_values,predicted_values)
print(simu_balance,simu_stocks)
utils.plotResults([real_values,predicted_values])