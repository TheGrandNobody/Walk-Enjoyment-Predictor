# EDA script 

# Imports
import matplotlib.pyplot as plt
import pandas as pd

# load data
df = pd.read_csv('merged_data_upsample.csv')

# Average PACES scores into 1 variable mood
paces_cols = ['hate-enjoy', 'bored-interested', 'absorbed', 'tiring-energizing', 'depressed-happy', 'bad-good_physically', 'sense_accomplishment']
df['PACES'] = df[paces_cols].mean(axis=1)
print(df['PACES'].describe())

# Make table with missing%, mean, sd, minimum, maximum --------------------------------------------------------------
stats = df.describe().transpose()

# Add the median to the statistics table
stats['median'] = df.median()

# Add percentage of missing values to table
# stats['%missing values'] = df.isnull().mean() * 100
# print(stats['%missing values'])

# Remove ID  and paces questions from stats
stats = stats.drop('ID')
stats = stats.drop(paces_cols)
print(stats)

# Save stats 
stats.to_csv('summary.csv', index=True)

# Calculate percentage of labels for each categorical variable
cat_stats = pd.DataFrame()
col_counts = df['Weather'].value_counts(normalize=True) * 100
cat_stats['Weather'] = col_counts

# Print the percentage table
print(cat_stats)

# Accelerometer timeseries ------------------------------------------------------------------------

# Filter columns containing "acceleration" in their name
acceleration_cols = [col for col in df.columns if 'Acceleration' in col and not 'Linear' in col]

# Convert 'timestamp' column to datetime
df['timestamp'] = pd.to_datetime(df['Tijdstempel'])

# Filter DataFrame for ID 1
df_id1 = df[df['ID'] == 1.0]
print(df_id1)

# Sort DataFrame by 'timestamp' column
df_id1 = df_id1.sort_values('timestamp')

# Define the desired time range
start_time = pd.to_datetime('2023-06-08 18:00:00')
end_time = pd.to_datetime('2023-06-08 21:00:00')

# Create a larger figure with adjusted aspect ratio
fig, ax = plt.subplots(figsize=(12, 6))

# Plot time series for acceleration columns with different line styles per ID
for col in acceleration_cols:
    plt.plot(df_id1['timestamp'], df_id1[col], label=col, )

# Set x-axis limits to the desired time range
plt.xlim(start_time, end_time)

plt.xlabel('Timestamp')
plt.ylabel('Acceleration')
plt.title('Acceleration Time Series')
plt.legend()
plt.savefig('timeseries_acceleration')
