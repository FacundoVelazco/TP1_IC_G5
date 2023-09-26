def simulation(real_values, predicted_values):
    balance = 10000
    stocks = 0
    history_balance = [balance + stocks*real_values[0]]
    for i in range(0, len(real_values) - 1, 1):
        balance = balance + stocks * real_values[i]
        stocks = 0
        if predicted_values[i + 1] > predicted_values[i]:
            stocks = stocks + balance / real_values[i]
            balance = 0
        history_balance.append(balance + stocks*real_values[i])

    return history_balance
