import pandas as pd
import matplotlib.pyplot as plt

# merged = pd.read_csv('./merged.csv')
# master = pd.read_csv('./master.csv')
# slave = pd.read_csv('./slave.csv')
final = pd.read_csv('./out_final.csv')

final['marker'] = (final['marker'] - final['marker'].min()) / (final['marker'].max() - final['marker'].min())
final['x'] = final['x']/(final['x'].min()*-2) + 0.5


plt.plot(final['time_marker'], final['marker']*2, label='Raw Marker')
plt.plot(final['time_board'], final['marker']*2, label='Synced Marker')
plt.plot(final['time_board'], final['x'], label='Synced Board')

# plt.plot(final['time_marker'], final['marker']/1000*22-60, label='Raw Marker')
# plt.plot(final['time_board'], final['marker']/1000*22-60, label='Synced Marker')
# plt.plot(final['time_board'], final['x'], label='Synced Board')
plt.plot()
plt.legend()
plt.show()