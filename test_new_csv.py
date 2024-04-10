import pandas as pd
import numpy as np
import csv

df = pd.read_csv('datasets/Reviews.csv')
print(df.info())
csv_file_array = []

filename = "datasets/amazon-reviews-modified(10000).csv"

m,n = df.shape
print(m,n)

id = np.array(df.iloc[:10000,0])
summary = np.array(df.iloc[:10000,8],dtype=str)
text = np.array(df.iloc[:10000,9],dtype=str)

for i in range(0,10000):
    csv_file_array.append([id[i],summary[i],text[i]])

for i in range(9995,10000):
    print(csv_file_array[i])

fields = ['id', 'summary', 'text'] 

with open(filename, 'w', encoding="utf-8") as csvfile:  
    # creating a csv writer object  
    csvwriter = csv.writer(csvfile)  
        
    # writing the fields  
    csvwriter.writerow(fields)  
        
    # writing the data rows  
    csvwriter.writerows(csv_file_array)
