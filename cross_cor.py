# Test for stationarity

#  Imports
from timeseries_eda import separate_walks
from statsmodels.tsa.stattools import kpss, adfuller
import pandas as pd
from matplotlib.pyplot import xcorr

def stat_result(timeseries, test):
    """Tests timeseries for stationarity and returns a boolean telling you if it is (True) or not (False)

    Args:
        timeseries (list): list with a timeseries. The timeseries must contain a regular time interval.
        test(Str): The test you want to use (kpss for KPSS or adf for Dickey Fuller test). 

    Returns:
        Bool:True = stationary, False = not stationary
    """
    if test == "kpss":
        result = kpss(timeseries, regression='ct')
    else: 
         result = adfuller(timeseries, autolag='AIC')

    if result[1] > 0.05: # Check the p-value
        if test=="kpss":  
            return True
        else: 
            return False
    else:
        if test=="kpss":
            return False
        else:
            return True
    
def crosscorr(datax, datay, lag=0):
    """ Lag-N cross correlation. Source:https://stackoverflow.com/questions/33171413/cross-correlation-time-lag-correlation-with-pandas
    Parameters
    ----------
    lag : int, default 0
    datax, datay : pandas.Series objects of equal length

    Returns
    ----------
    crosscorr : float
    """
    return datax.corr(datay.shift(lag))

if __name__ == "__main__":

    # Load data
    df = pd.read_csv('clean_1s.csv')

    # Average PACES scores into 1 variable mood
    paces_cols = ['hate-enjoy', 'notpleasant_pleasant', 'notpleasurable_pleasurable', 'bad-good_feeling']
    df['PACES'] = df[paces_cols].sum(axis=1)

    # Separate dataset in continuous walks
    walks = separate_walks(df)

    # Define columns for Dickey fuller test
    num_vars=['Acceleration x', 'Acceleration y', 'Acceleration z', 'Linear Acceleration x','Linear Acceleration y', 'Linear Acceleration z', 'Gyroscope x', 'Gyroscope y', 'Gyroscope z', 'La', 'Lo', 'He', 'V', 'D', 'Ho', 'VA', 'Magnetic field x', 'Magnetic field y', 'Magnetic field z']
    
    # Initizalize empty lists to 
    stat = []
    non_stat = []

    walk=walks[1]

    # For every variable 
    for column in walk.columns:
        # Only if variable is numerical
        if column in num_vars:
            # Make list of variables that are stationaty and a list where they are not
            if stat_result(walk[column], test="kpss"):
                stat.append(column)
            else:
                non_stat.append(column)

    # Analyse stationarity
    print("stat", stat,"\n non_stat",non_stat)