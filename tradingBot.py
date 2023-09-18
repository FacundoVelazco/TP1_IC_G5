def simulation(real_values, predicted_values):
    balance = 10000
    stocks = 0
    for i in range(0, len(real_values) - 1, 1):
        if balance <= 0:
            break
        else:
            if predicted_values[i + 1] > predicted_values[i]:
                stocks = stocks + balance / real_values[i]
                balance = 0
            else:
                balance = balance + stocks * real_values[i]
                stocks = 0
    return balance, stocks*real_values[-1]
