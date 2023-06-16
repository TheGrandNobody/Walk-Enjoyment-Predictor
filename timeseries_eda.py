# Get information on timeseries

#  Imports
import pandas as pd
import statistics

# Load data
df = pd.read_csv('merged_data_upsample_10s.csv')

# Define maximum amount of minutes that are allowed between walks for them to be considered one walk
LIMIT=20

def separate_walks(df):

    df['Date'] = pd.to_datetime(df['datetime'])
    df = df.sort_values('Date')
    
    # Intialize list for each separate walk
    df_parts = []

    current_part = []
    prev_row = None 

    for index, row in df.iterrows():
        if prev_row is None:
            current_part.append(row)
        else:
            time_diff = row['Date'] - prev_row['Date']
            if time_diff.total_seconds() / 60 > LIMIT or row['ID'] != prev_row["ID"]:
                df_parts.append(pd.DataFrame(current_part))
                current_part = []
            current_part.append(row)
        prev_row = row

    # Add the last part to the list
    df_parts.append(pd.DataFrame(current_part))

    return df_parts

def duration(walks):
    minutes_walk=[]

    for walk in walks:
        print(walk)
        walk['Date'] = pd.to_datetime(walk['Date'])
        minutes_walk.append((walk['Date'].iloc[walk['Date'].shape[0] - 1] -  walk['Date'].iloc[0]).total_seconds()/60)
    return minutes_walk

if __name__ == "__main__":
    walks = separate_walks(df)
    print("Total amount of walks = ", len(walks))

    minutes_walk=duration(walks)
    print("mean, sd, median:", statistics.mean(minutes_walk), statistics.stdev(minutes_walk), statistics.median(minutes_walk))