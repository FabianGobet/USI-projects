import pandas as pd
import matplotlib.pyplot as plt

file_path = './scaling.csv'
data = pd.read_csv(file_path)

data_50 = data[data['tasks'] == 50]
data_100 = data[data['tasks'] == 100]

plt.figure(figsize=(10, 6))
plt.plot(data_50['workers'], data_50['time'], label='Tasks=50', marker='o')
plt.plot(data_100['workers'], data_100['time'], label='Tasks=100', marker='o')

plt.xlabel('Number of Workers')
plt.ylabel('Time')
plt.title('Scaling Analysis: Time vs Number of Workers')
plt.legend()
plt.grid(True)
plt.yticks(data['time'],fontsize=7)

plt.savefig('scaling.png')
plt.show()

def calculate_efficiency(baseline_time, actual_time):
    return (baseline_time / actual_time) * 100

baseline_time_50 = data_50[data_50['workers'] == 2]['time'].iloc[0]
baseline_time_100 = data_100[data_100['workers'] == 2]['time'].iloc[0]

data_50['efficiency'] = data_50['time'].apply(lambda x: calculate_efficiency(baseline_time_50, x))
data_100['efficiency'] = data_100['time'].apply(lambda x: calculate_efficiency(baseline_time_100, x))

plt.figure(figsize=(10, 6))
plt.plot(data_50['workers'], data_50['efficiency'], label='Tasks=50', marker='o')
plt.plot(data_100['workers'], data_100['efficiency'], label='Tasks=100', marker='o')

plt.xlabel('Number of Workers')
plt.ylabel('Efficiency (%)')
plt.title('Scaling Analysis: Efficiency vs Number of Workers')
plt.legend()
plt.grid(True)

plt.savefig('efficiency.png')
plt.show()