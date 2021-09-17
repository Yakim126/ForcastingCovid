#import pandas as pd
#f=pd.read_csv("CovidUaData1.csv")
#keep_col = ['Date_reported','New_cases']
#new_f = f[keep_col]
#new_f.to_csv("CovidUaData.csv", index=False)

with open('../CovidUaData.csv') as s, open('CovidUaData0.csv', 'w') as d:
    for line in s.read().split('\n'):
        d.write(line)
        d.write('.0\n')