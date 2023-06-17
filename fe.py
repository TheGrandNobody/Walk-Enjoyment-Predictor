import pandas as pd
import numpy as np

GRANULARITY = 5 # 5 seconds

# Function to calculate distance between two lat/lon points using Haversine formula
def haversine(lat1, lon1, lat2, lon2):
    R = 6371  # Radius of the Earth in kilometers
    phi1 = np.radians(lat1)
    phi2 = np.radians(lat2)
    delta_phi = np.radians(lat2 - lat1)
    delta_lambda = np.radians(lon2 - lon1)
    a = np.sin(delta_phi/2)**2 + np.cos(phi1) * np.cos(phi2) * np.sin(delta_lambda/2)**2
    res = R * (2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a)))
    return np.abs(res)

if __name__ == "__main__":
    # Load your data
    df = pd.read_csv('clean_5s.csv')

    ## GYROSCOPE
    # Calculate mean of gyroscope data
    df['gyro_mean'] = df[['Gyroscope x', 'Gyroscope y', 'Gyroscope z']].mean(axis=1)
    # Calculate median of gyroscope data
    df['gyro_median'] = df[['Gyroscope x', 'Gyroscope y', 'Gyroscope z']].median(axis=1)
    # Calculate magnitude of angular velocity
    df['gyro_magnitude'] = np.sqrt(df['Gyroscope x']**2 + df['Gyroscope y']**2 + df['Gyroscope z']**2)

    ## MAGNETOMETER
    # Calculate mean of magnetometer data
    df['mag_mean'] = df[['Magnetic field x', 'Magnetic field y', 'Magnetic field z']].mean(axis=1)
    # Calculate median of magnetometer data
    df['mag_median'] = df[['Magnetic field x', 'Magnetic field y', 'Magnetic field z']].median(axis=1)
    # Calculate strength of magnetic field
    df['mag_strength'] = np.sqrt(df['Magnetic field x']**2 + df['Magnetic field y']**2 + df['Magnetic field z']**2)

    ## ACCELEROMETER (with gravity)
    # Calculate mean of accelerometer data
    df['acc_mean'] = df[['Acceleration x', 'Acceleration y', 'Acceleration z']].mean(axis=1)
    # Calculate median of accelerometer data
    df['acc_median'] = df[['Acceleration x', 'Acceleration y', 'Acceleration z']].median(axis=1)
    # Calculate the angle of inclination
    df['acc_angle'] = np.arctan(np.sqrt(df['Acceleration x']**2 + df['Acceleration y']**2) / df['Acceleration z'])

    ## ACCELEROMETER (without gravity)
    # Calculate mean of accelerometer data
    df['lacc_mean'] = df[['Linear Acceleration x', 'Linear Acceleration y', 'Linear Acceleration z']].mean(axis=1)
    # Calculate median of accelerometer data
    df['lacc_median'] = df[['Linear Acceleration x', 'Linear Acceleration y', 'Linear Acceleration z']].median(axis=1)
    # Calculate the total acceleration
    df['tot_acc'] = np.sqrt(df['Linear Acceleration x']**2 + df['Linear Acceleration y']**2 + df['Linear Acceleration z']**2)

    ## GPS
    # Shift the DataFrame to get the next coordinates, calculate the distance, and then fill any NaN values with 0
    df['lat_shift'] = df['La'].shift(-1).fillna(df['La'].iloc[-1])
    df['lon_shift'] = df['Lo'].shift(-1).fillna(df['Lo'].iloc[-1])
    df['distance'] = haversine(df['La'], df['Lo'], df['lat_shift'], df['lon_shift'])
    df['distance'].fillna(0, inplace=True)
    # Calculate total cumulative distance traveled
    df['cum_dist'] = df['distance'].cumsum()
    # Calculate total ascent and descent
    df['height_diff'] = df['He'].diff().fillna(df['He'].iloc[0])
    df['ascent'] = np.where(df['height_diff'] > 0, df['height_diff'], 0)
    df['descent'] = np.where(df['height_diff'] < 0, -df['height_diff'], 0)
    df['total_ascent'] = df['ascent'].cumsum()
    df['total_descent'] = df['descent'].cumsum()
    # Calculate changes in direction
    df['direction_diff'] = df['D'].diff().abs()
    df['direction_diff'] = df['direction_diff'].fillna(df['direction_diff'].iloc[1])
    # Calculate acceleration
    df['velocity_diff'] = df['V'].diff().fillna(df['V'].iloc[0])
    df['acceleration'] = df['velocity_diff'] / (GRANULARITY / 3600)  # Assuming time difference is 30 seconds

    # Save the DataFrame to a new CSV file
    df.to_csv('final_5s.csv', index=False)