import pandas as pd
f=pd.read_csv("CovidUaData1.csv")
keep_col = ['Date_reported','New_cases']
new_f = f[keep_col]
new_f.to_csv("CovidUaData.csv", index=False)