import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('merged_data_upsample.csv')
# Calculate the average of each of the last six columns
df['average'] = df.iloc[:, -4:].mean(axis=1)

# Create a boxplot of these averages
plt.figure(figsize=(12, 8))
plt.boxplot(df['average'])
plt.title('Boxplot of Averaged PACE values')
plt.xlabel('Average PACE score')
plt.show()