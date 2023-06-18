# EDA script 

# Imports
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from scipy.stats.stats import pearsonr

# Define name for output file
OUT_FILE = 'time_series_10s.png'

# Define title for plot
TITLE = 'Time Series (granularity = 1s)'

# Define the desired time range for plot
START = pd.to_datetime('2023-06-08 18:14:00')
END = pd.to_datetime('2023-06-08 20:30:00')

# load data
df = pd.read_csv('merged_data_1s.csv')

# Average PACES scores into 1 variable mood
paces_cols = ['hate-enjoy', 'notpleasant_pleasant', 'notpleasurable_pleasurable', 'bad-good_feeling']
df['PACES'] = df[paces_cols].sum(axis=1)
print(df['PACES'].describe())

# Rename wind column
# df = df.rename(columns={'Wind (m/s)': 'Wind (Km/h)'})

# Plot weather and mood score -----------------------------------------------------

fig, axes = plt.subplots(nrows=1, ncols=3)

variables = ["Temperature (Celsius)", "Wind (Km/h)", "Humidity (%)"]
titles = ["Temperature", "Wind", "Humidity"]

fig, axes = plt.subplots(nrows=1, ncols=3, sharey=True)

for i in range(len(variables)):
    sns.scatterplot(x=variables[i], y="PACES", data=df, ax=axes[i])
    axes[i].set(xscale="log")
    axes[i].set_title(titles[i])

plt.tight_layout()

# Save figure
plt.savefig("plots/scatter.png")

# Print pearson test for every weather condition
for x in variables:
    print(x, pearsonr(df[x], df['PACES']))

# Plot Sensory data and mood score ------------------------------------------------------------------------------------------

# Make table with missing%, mean, sd, minimum, maximum --------------------------------------------------------------
stats = df.describe().transpose()

# Add the median to the statistics table
stats['median'] = df.median()

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

# Plot timeseries ------------------------------------------------------------------------

# Group columns per category
acceleration_cols = [col for col in df.columns if 'Acceleration' in col and not 'Linear' in col]
mag_cols = [col for col in df.columns if 'Magnetic' in col]
gyr_cols = [col for col in df.columns if 'Gyroscope' in col]
weather_cols = [col for col in df.columns if any(word in col for word in ['Temperature', 'UV', 'Wind', 'Humidity'])]
accel_lin_cols=[col for col in df.columns if 'Acceleration' in col and 'Linear' in col]
loc_cols= [col for col in df.columns if col in ['La' 'Lo','He','V','D','Ho','VA']]
mood = ['PACES']

# Convert 'timestamp' column to datetime
df['timestamp'] = pd.to_datetime(df['datetime'])

# Filter DataFrame for ID 1
df_id1 = df[df['ID'] == 1.0]

# Sort DataFrame by 'timestamp' column
df_id1 = df_id1.sort_values('timestamp')

# Create a larger figure with adjusted aspect ratio
fig, ax = plt.subplots(7,1,figsize=(17,10))

# Put different plot categories in list
category_list=[acceleration_cols, mag_cols, gyr_cols, accel_lin_cols, weather_cols, loc_cols, mood]

# Loop through different plot categories and plot timeseries
for i, category in enumerate(category_list):
    for col in category:
        ax[i].plot(df_id1['timestamp'], df_id1[col], label=col, )

# list of all plots
plots=['Acceleration', 'Magnotmeter', 'Gyroscope', 'Acceleration linear', 'Weather', 'Location (GPS)', 'Mood (PACES)']

# plot different plots
for i, plot in enumerate(plots):
    if i == 0:
        ax[i].set_title(TITLE)

    # plot i
    ax[i].set_ylabel(plot)
    ax[i].legend()
    ax[i].set_xlim(START, END)
    if i != len(plots)-1:
        ax[i].set_xticks([])
    else:
        ax[i].set_xlabel('Date and time')

plt.savefig(f"plots/{OUT_FILE}")

# Cross correlation plots