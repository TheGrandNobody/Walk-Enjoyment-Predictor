from sklearn.ensemble import IsolationForest
import pandas as pd

if __name__ == "__main__":
    
    # Load the data
    data = pd.read_csv("merged_data_upsample.csv")
    data['datetime'] = pd.to_datetime(data['datetime'])
    data['datetime'] = data['datetime'].apply(lambda x: x.timestamp())
    data['UV index'] = data['UV index'].apply(lambda x: x if x.isnumeric() else 5 if 'moderate' else 3)
    data['Weather'] = data['Weather'].apply(lambda x: 0 if x == "Sunny" else 1 if x == "Rainy" else 0)

    # Initialize the IsolationForest model
    clf = IsolationForest(contamination=0.01)

    # Fit the model on the data
    clf.fit(data)

    # Add the prediction to your dataframe
    data['anomaly'] = clf.predict(data)

    # Filter your dataframe to remove the anomalies
    data = data[data['anomaly'] == 1]
    df = data.drop('anomaly', axis=1)

    # Save the dataframe to a csv file
    df.to_csv("clean.csv", index=False)

    
    