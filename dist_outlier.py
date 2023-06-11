from sklearn.ensemble import IsolationForest
import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt

if __name__ == "__main__":
    # Load the data
    data = pd.read_csv("merged_data_upsample.csv")
    data['datetime'] = pd.to_datetime(data['datetime'])
    data['datetime'] = data['datetime'].apply(lambda x: x.timestamp())
    data['UV index'] = data['UV index'].apply(lambda x: x if x.isnumeric() else 5 if 'moderate' else 3)
    data['Weather'] = data['Weather'].apply(lambda x: 0 if x == "Sunny" else 1 if x == "Rainy" else 0)
    # Assuming df is your DataFrame

    # Select only numeric columns
    numeric_cols = data.drop(["Unnamed: 0", "ID", "Weather"], axis = 1).select_dtypes(include=[np.number]).columns.tolist()

    n = len(numeric_cols)
    sqrt_n = int(np.ceil(np.sqrt(n))) # for plotting on a square grid

    fig, axs = plt.subplots(sqrt_n, sqrt_n, figsize=(20, 20))

    for ax, col in zip(axs.flatten(), numeric_cols):
        sns.distplot(data[col], hist = False, kde = True, 
                    kde_kws = {'linewidth': 3}, ax=ax)
        ax.set_title('Distribution of '+col)

    # If there are any subplots you didn't use, you should delete them
    for ax in axs.flatten()[n:]:
        fig.delaxes(ax)

    fig.tight_layout()
    plt.savefig("dist.png")

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

    
    