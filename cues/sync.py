import random
from typing import List
import matplotlib.pyplot as plt


def get_closest_timestamp_index_in_master(
        master_timestamp: List[float],
        slave_timestamp: float,
        master_timestamp_avg_increment: float,
        master_max_index: int) -> int:
    if slave_timestamp < master_timestamp[0]:
        return 0

    estimated_index = int((slave_timestamp - master_timestamp[0]) / master_timestamp_avg_increment)

    if estimated_index < 0:
        return 0

    while 0 < estimated_index < master_max_index - 1:
        if master_timestamp[estimated_index - 1] <= slave_timestamp <= master_timestamp[estimated_index + 1]:
            return estimated_index
        elif slave_timestamp < master_timestamp[estimated_index]:
            estimated_index -= 1
        elif slave_timestamp > master_timestamp[estimated_index]:
            estimated_index += 1

    closest_point: int = min(range(len(master_timestamp)),
                             key=lambda i: abs(master_timestamp[i] - slave_timestamp))
    return closest_point


def fill(
        start_index: int,
        end_index: int,
        slave_main: List[float],
        slave_main_value_index: int
):
    fill_data = []
    fill_size = (end_index - start_index)
    if fill_size > 0:
        input_data = slave_main[slave_main_value_index]
        channel_data = [input_data] * fill_size
        fill_data = channel_data
    if len(fill_data) == 0 and start_index == end_index:
        fill_data = [slave_main[0]]
    return fill_data


samples = 5000
master_to_slave_ratio = 439
master_to_slave_async = 0.154931
slave_start = -528
slave_end = samples + 312
master_main = [random.random() -1 for _ in range(samples)]
master_timestamp_data = [i*0.1387459 for i in range(samples)]
slave_main = [random.randint(0, 2) for _ in range(slave_start, slave_end, master_to_slave_ratio)]
slave_timestamp_data = [i*master_to_slave_async-15 for i in range(slave_start, slave_end, master_to_slave_ratio)]

master_max_index = len(master_timestamp_data) - 1
master_timestamp_avg_increment = (master_timestamp_data[-1] - master_timestamp_data[0]) / master_max_index

new_slave_data = []
max_slave_index = len(slave_timestamp_data) - 1
last_closest_index = 0
last_slave_index = 0

for slave_timestamp_index, slave_timestamp_value in enumerate(slave_timestamp_data):
    # stop processing if slave timestamp is greater than master timestamp, as we can't be sure if there's more master data incoming to sync
    if slave_timestamp_value > master_timestamp_data[master_max_index]:
        break

    # stop processing if it's the last slave timestamp, as we can't be sure if there's more slave data incoming to sync
    if slave_timestamp_index == max_slave_index:
        break

    closest_point_start = get_closest_timestamp_index_in_master(
        master_timestamp_data,
        slave_timestamp_value,
        master_timestamp_avg_increment,
        master_max_index
    )

    if last_closest_index == closest_point_start and last_closest_index == 0:
        del new_slave_data[0:1]

    value_index = slave_timestamp_index
    if slave_timestamp_index - 1 >= 0:
        value_index = slave_timestamp_index - 1

    new_slave_data.extend(
        fill(
            last_closest_index,
            closest_point_start,
            slave_main,
            value_index
        )
    )
    last_closest_index = closest_point_start
    last_slave_index = slave_timestamp_index

output_timestamp = master_timestamp_data[0:last_closest_index+1]
output_master_main = master_main[0:last_closest_index+1]


plt.plot(slave_timestamp_data, slave_main, label='Raw Marker')
plt.plot(output_timestamp, new_slave_data, label='Synced Marker')
plt.plot(output_timestamp, output_master_main, label='Synced Board')

plt.plot()
plt.legend()
plt.show()