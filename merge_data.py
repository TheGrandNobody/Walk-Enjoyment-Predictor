#####################################
# Merge sensory data with label data
#####################################
import os
import datetime
import pandas as pd
from collections import defaultdict

# Sample and duplicate label every interval seconds
UP_SAMPLING_INTERVAL = 30


def load_unify_data(xls):
    """
    Load data from different devices and unify format
    """
    dfs = {}
    # Load neccesary data and unify columns
    dfs["Accelerometer"] = pd.read_excel(xls, "Accelerometer")
    dfs["Accelerometer"].columns = ["Time", "Acceleration x", "Acceleration y", "Acceleration z"]

    try:
        dfs["Linear Accelerometer"] = pd.read_excel(xls, "Linear Accelerometer")
    except:
        # Lucas data (ID=3) have an alternative sheet_name "Linear Acceleration" for "Linear Accelerometer"
        dfs["Linear Accelerometer"] = pd.read_excel(xls, "Linear Acceleration")
    dfs["Linear Accelerometer"].columns = ["Time", "Linear Acceleration x", "Linear Acceleration y", "Linear Acceleration z"]

    dfs["Gyroscope"] = pd.read_excel(xls, "Gyroscope") 
    dfs["Gyroscope"].columns = ["Time", "Gyroscope x", "Gyroscope y", "Gyroscope z"]

    dfs["Location"] = pd.read_excel(xls, "Location")
    dfs["Location"].columns = ["Time", "La", "Lo", "He", "V", "D", "Ho", "VA"]

    dfs["Magnetometer"] = pd.read_excel(xls, "Magnetometer")
    dfs["Magnetometer"].columns = ["Time", "Magnetic field x", "Magnetic field y", "Magnetic field z"]

    return dfs


def calculate_cor_time(x, exp_pairs):
    """
    Calculate corresponding system time for a given pandas series
    """
    for i in range(len(exp_pairs["start_times"])):
        # Find the corresponding interval
        if x >= exp_pairs["start_times"][i]["exp_time"] and x < exp_pairs["end_times"][i]["exp_time"]:
            sst = exp_pairs["start_times"][i]["sys_time"]
            est = exp_pairs["start_times"][i]["exp_time"]
            break
        else:
            continue
    # exp_start_time & sys_start_time
    sst = datetime.datetime.fromtimestamp(sst)
    delta = x-est
    return sst + datetime.timedelta(seconds=delta)


def get_exp_pairs(meta_time):
    """
    Get Start-Pause time pairs for every experiment run
    """
    exp_pairs = {"start_times": [], "end_times": []}
    for index, row in meta_time.iterrows():
        if row["event"] == "START":
            exp_pairs["start_times"].append({"exp_time": row["experiment time"], "sys_time":row["system time"]})
        else:
            exp_pairs["end_times"].append({"exp_time": row["experiment time"], "sys_time":row["system time"]})
    # if num of Start != num of Pause, cut to align them.
    exp_pairs["start_times"] = exp_pairs["start_times"][:min(len(exp_pairs["start_times"]), len(exp_pairs["end_times"]))]
    exp_pairs["end_times"] = exp_pairs["end_times"][:min(len(exp_pairs["start_times"]), len(exp_pairs["end_times"]))]
    return exp_pairs


def calculate_system_time(df, meta_time: pd.DataFrame):
    """
    Experiment time is accumulative, it does not consider the gap between pause and restart,
    we need to add this gap in and calculate the precise system time for every data row.
    """
    exp_pairs = get_exp_pairs(meta_time)
    df = df[(df["Time"]<exp_pairs["end_times"][-1]["exp_time"]) & (df["Time"]>exp_pairs["start_times"][0]["exp_time"])]
    df["datetime"] = df["Time"].apply(lambda x: calculate_cor_time(x, exp_pairs))
    return df


def up_sampling_label(labels, meta_time, interval=60):
    """
    Up-sampling sparse labels
    """
    all_labels = []
    exp_pairs = get_exp_pairs(meta_time)
    for i in range(len(exp_pairs["start_times"])):
        sst = exp_pairs["start_times"][i]["sys_time"]
        pst = exp_pairs["end_times"][i]["sys_time"]
        sst = datetime.datetime.fromtimestamp(sst)
        pst = datetime.datetime.fromtimestamp(pst)
        label = labels[(labels["datetime"] >= sst) & (labels["datetime"] <= pst)].reset_index(drop=True)
        dt = [sst]
        while dt[-1] <= pst:
            dt += [dt[-1]+datetime.timedelta(seconds=interval)]
        expand_labels = pd.concat([label] * len(dt), ignore_index=True)
        expand_labels["datetime"] = dt
        all_labels.append(expand_labels)
    
    return pd.concat(all_labels, ignore_index=True)


def multilabel_merge(dfs, labels, meta_time, up_sampling=True):
    """
    Merge under up-sampling mode
    """
    start_date_time = datetime.datetime.fromtimestamp(int(meta_time["system time"].tolist()[0])) - datetime.timedelta(minutes=1)
    end_date_time = datetime.datetime.fromtimestamp(int(meta_time["system time"].tolist()[-1])) + datetime.timedelta(minutes=1)
    label_dfs = defaultdict(list)
    if up_sampling:
        selected_rows = up_sampling_label(labels, meta_time, interval=UP_SAMPLING_INTERVAL)
    else:
        selected_rows = labels[(labels["datetime"] >= start_date_time) & (labels["datetime"] <= end_date_time)].reset_index(drop=True)
    # For all label rows, find corresponding all variable df and save in a dict
    for sheet_name, var_df in dfs.items():
        var_df = calculate_system_time(var_df, meta_time)
        for dt in selected_rows["datetime"].tolist():
            if up_sampling:
                window = var_df[(var_df["datetime"] >= dt) & \
                                (var_df["datetime"] <= dt+datetime.timedelta(seconds=UP_SAMPLING_INTERVAL))]
            else:
                window = var_df[(var_df["datetime"] >= dt-datetime.timedelta(minutes=5)) & \
                    (var_df["datetime"] <= dt+datetime.timedelta(minutes=5))]
            # TODO two strategies: clean date before merging, or after. Default clean data after merging.
            mean_var = window.mean().to_frame().transpose()
            mean_var = mean_var.drop(columns=["Time", "datetime"])
            label_dfs[str(dt)].append(mean_var)
    var_dfs = []
    # Concat all variables (horizontal)
    for dt, val in label_dfs.items():
        row_merged_df = pd.concat(val, axis=1)
        var_dfs.append(row_merged_df)
    
    # Concat sensory data (all variables) with label data
    var_dfs = [pd.concat(var_dfs, axis=0, ignore_index=True)]
    merged_df = pd.concat(var_dfs+[selected_rows], axis=1).reset_index(drop=True)
    merged_df = merged_df.dropna(axis=0)

    return merged_df


def merge_one_exp(file, google_sheet, up_sampling=True):
    """
    Merge one experiment with corresponding questionaire answer including mood scale and whether.
    Args:
        file (string): the path of a 10 mins data recording using phyphox iPhone version
        google_sheet (Dataframe)
    """
    selected_row = None
    print(file)
    xls = pd.ExcelFile(file)
    dfs = load_unify_data(xls)

    meta_time = pd.read_excel(file, sheet_name="Metadata Time")

    labels = google_sheet
    labels["Date"] = labels["Date"].astype(str)
    labels["Time"] = labels["Time"].astype(str)
    labels["datetime"] = pd.to_datetime(labels["Date"] + " " + labels["Time"])

    # Concat all variable rows
    if up_sampling:
        merged_df = multilabel_merge(dfs, labels, meta_time, up_sampling=up_sampling)
    else:
        # Get time range, add 1 min redundency
        start_date_time = datetime.datetime.fromtimestamp(int(meta_time["system time"].tolist()[0])) - datetime.timedelta(minutes=1)
        end_date_time = datetime.datetime.fromtimestamp(int(meta_time["system time"].tolist()[-1])) + datetime.timedelta(minutes=1)
        var_dfs = []
        # Select corresponding label row
        selected_row = labels[(labels["datetime"] >= start_date_time) & (labels["datetime"] <= end_date_time)].reset_index(drop=True)
        for sheet_name, var_df in dfs.items():
            # TODO two strategies: clean date before merging, or after. Default clean data after merging.
            mean_var = var_df.mean().to_frame().transpose()
            mean_var = mean_var.reset_index(drop=True)
            mean_var = mean_var.drop(columns="Time")
            var_dfs.append(mean_var)
        # Concat sensory data with lable data
        merged_df = pd.concat(var_dfs+[selected_row], axis=1)

    return merged_df


def merge_multiple_exp(file, google_sheet, up_sampling=True):
    """
    This is for merging data recording by Android devices since it record accumulative data.
    Args:
        file (string): the path of a cumulative data recording using phyphox Android version
        google_sheet (Dataframe)
    """
    print(file)
    xls = pd.ExcelFile(file)
    dfs = load_unify_data(xls)

    # Get time range, add 1 min redundency
    meta_time = pd.read_excel(file, sheet_name="Metadata Time")

    labels = google_sheet
    labels["Date"] = labels["Date"].astype(str)
    labels["Time"] = labels["Time"].astype(str)
    labels["datetime"] = pd.to_datetime(labels["Date"] + " " + labels["Time"])

    return multilabel_merge(dfs, labels, meta_time, up_sampling=up_sampling)


if __name__ == "__main__":
    
    google_sheet = pd.read_excel("MFQS walking questionnaire 2.xlsx",
                                  sheet_name="Formulierreacties 1")

    data_dir1 = "data/iPhone"
    files = [os.path.join(data_dir1, file) for file in os.listdir(data_dir1)]
    
    # Merge iPhone's data
    dfs = []
    for file in files:
        dfs.append(merge_one_exp(file, google_sheet, up_sampling=True))
    merged_df1 = pd.concat(dfs, axis=0).reset_index(drop=True)

    data_dir2 = "data/Android"
    files = [os.path.join(data_dir2, file) for file in os.listdir(data_dir2)]

    # Merge Android's data
    dfs = []
    for file in files:
        dfs.append(merge_multiple_exp(file, google_sheet, up_sampling=True))
    merged_df2 = pd.concat(dfs, axis=0).reset_index(drop=True)

    # Merge all data and save
    merged_df_final = pd.concat([merged_df1, merged_df2], axis=0).reset_index(drop=True)

    # Re arrange columns
    merged_df_final = merged_df_final.drop(columns=["Tijdstempel", "Date", "Time"])
    meta_cols = ["ID", "datetime"]
    label_cols = ["hate-enjoy", "bored-interested", "absorbed", "tiring-energizing", "depressed-happy", "bad-good_physically", "sense_accomplishment"]
    whether_cols = ["Weather", "Temperature (Celsius)", "UV index", "Wind (m/s)", "Humidity (%)"]
    new_col = meta_cols + [col for col in merged_df_final.columns.to_list() if col not in meta_cols+whether_cols+label_cols] + whether_cols + label_cols
    merged_df_final = merged_df_final[new_col]
    
    merged_df_final.to_csv("merged_data_upsample.csv")
