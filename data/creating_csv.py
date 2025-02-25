import os
import pandas as pd
import csv
import numpy as np

arr = os.listdir('/models/all_insects')

df = pd.DataFrame([arr]).T
df.columns = ['file_name']
# df['file_name'] = df['file_name'].str.rsplit('.', n=1).str[0]
print(df)

def insect_class(file_name):
    if "butterfly" in file_name:
        return 0
    elif "dragonfly" in file_name:
        return 1
    elif "grasshopper" in file_name:
        return 2
    elif "ladybug" in file_name:
        return 3
    elif "mosquito" in file_name:
        return 4
    else:
        return "unclassified"

df['classification'] = df['file_name'].apply(insect_class)
print(df)

df.to_csv('all_insects.csv', index=False)