import pandas as pd

import sys
sys.path.append('/Users/meruozhu/Downloads/MP_data/MP_codes/MP')


df1 = pd.read_csv("/Users/meruozhu/Downloads/MP_data/MP_codes/MP/Data/BPIC15_1_sorted_new.csv")
df2 = pd.read_csv("/Users/meruozhu/Downloads/MP_data/MP_codes/MP/Data/BPIC15_2_sorted_new.csv")
df3 = pd.read_csv("/Users/meruozhu/Downloads/MP_data/MP_codes/MP/Data/BPIC15_3_sorted_new.csv")
df4 = pd.read_csv("/Users/meruozhu/Downloads/MP_data/MP_codes/MP/Data/BPIC15_4_sorted_new.csv")
df5 = pd.read_csv("/Users/meruozhu/Downloads/MP_data/MP_codes/MP/Data/BPIC15_5_sorted_new.csv")

finaldf = pd.concat([df1, df2, df3, df4, df5])

finaldf.to_csv("/Users/meruozhu/Downloads/MP_data/MP_codes/MP/Data/sudden_drift/BPIC15_ALL.csv")
