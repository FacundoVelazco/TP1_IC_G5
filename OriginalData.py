import json
from datetime import datetime
import matplotlib.pyplot as plt

with open('nasdaq-index.json', 'r') as json_file: data = json.load(json_file)

result = list(map(lambda x: {"open": x["Open"], "date": x["Date"]}, data))
open_data = []
date_data = []

for item in result:
    open_data.append(item['open'])
    aux_date_obj = datetime.strptime(item["date"], "%Y-%m-%d")
    date_data.append(aux_date_obj)

fig = plt.figure(1, (20, 10))
plt.plot(open_data)
plt.show()
