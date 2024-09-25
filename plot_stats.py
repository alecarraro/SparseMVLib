import pandas as pd
import matplotlib.pyplot as plt

#Read the CSV file and extract the execution times
df = pd.read_csv('output/stats.csv')
mtxs = df['Matrix Name'].str.split('/').str[-1].str.replace('.mtx', '', regex=False)
time_classic = df['Time Classic']
time_strict = df['Time Strict']
time_relaxed = df['Time Relaxed']

#Plot histogram w/ execution times
plt.figure(figsize=(10, 6))
bar_width = 0.2
index = range(len(mtxs))
plt.bar(index, time_classic, width=bar_width, label='Classic partitioning', color='skyblue')
plt.bar([i + bar_width for i in index], time_strict, width=bar_width, label='Strict NNZ partitioning', color='orange')
plt.bar([i + 2 * bar_width for i in index], time_relaxed, width=bar_width, label='Relaxed NNZ partitioning', color='lightgreen')
plt.title('Execution Time Comparison')
plt.xlabel('Matrix name')
plt.ylabel('Execution Time (seconds)')
plt.xticks([i + bar_width for i in index], mtxs, rotation=90) 
plt.legend()
plt.tight_layout()
plt.savefig("output/stats.png")

