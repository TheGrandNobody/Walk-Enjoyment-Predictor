import pandas as pd

# The number of files to process
NUM_FILES = 6
OUT_FILE = "out.csv"

def normalize_time(df, max, min):
    if "Latitude (°)" in df.columns:
        df['Time (s)'] = [(max/min * 0.1) * i for i in range(len(df))]
    else:
        df['Time (s)'] = [0.1 * i for i in range(len(df))]
    return df

if __name__ == "__main__":
    out = None
    for i in range(1, NUM_FILES + 1):
        file = pd.read_excel(f"{i}.xls", sheet_name=[i for i in range(6)], engine="xlrd")
        ma = max([len(file[i]) for i in range(5)])
        mi = min([len(file[i]) for i in range(5)])
        for j in range(3):
            if j == 0:
                dfs = pd.merge(normalize_time(file[j], ma, mi), normalize_time(file[j + 1], ma, mi), on="Time (s)", how="outer")
            else:
                dfs = pd.merge(dfs, normalize_time(file[j + 1], ma, mi), on="Time (s)", how="outer")
        dfs['run'] = i
        out = dfs if out is None else pd.concat([out, dfs])
    out.to_csv(OUT_FILE, index=False)