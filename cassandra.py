import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import csv

df = pd.read_csv('./datasets/output.csv',encoding = "unicode_escape")
print(df.info())
# print(df.head(7))
# print(f"\n {df.iloc[3][1]}")
data = []
print("Enter Student USN")
search_query = input()
search_index = 0
print(search_query)
with open('./datasets/output.csv') as file_obj: 
   reader_obj = csv.reader(file_obj) 
   data = list(reader_obj)
print(data)
#    for row in reader_obj: 
#         data.append([row])
#         print(row)
print(len(data))
for i in range(0,len(data)):
   for j in range(0,len(data)):
      if(data[i][j] == search_query):
         search_index = i
         break
print(search_index)
         
def plotGraph(index):
  group_labels = np.array(["Sem-1","Sem-2","Sem-3","Sem-4","Sem-5"])
  width = 0.25

  Sem1 = float(data[index][3])
  Sem2 = float(data[index][4])
  Sem3 = float(data[index][5])
  Sem4 = float(data[index][6])
  Sem5 = float(data[index][7])
  print(f"\n{Sem1} {Sem2} {Sem3} {Sem4} {Sem5}")
  marks = np.array([Sem1,Sem2,Sem3,Sem4,Sem5])
  sgpa = marks.mean()
  print(f"Sgpa of {data[index][1]} is: ",sgpa)

#   X_axis = np.arange(len(group_labels)) 

#   bar1 = plt.bar(X_axis, Sem1, width, color = 'r', label = 'Precision') 
# #   bar2 = plt.bar(X_axis + width, Sem2, width, color='g', label = 'Recall')
# #   bar3 = plt.bar(X_axis+width*2, Sem3, width, color = 'b', label = 'F-measure') 
  bar1 = plt.bar(group_labels,marks,width=0.4)
  bar2 = plt.bar('SGPA',sgpa,width=0.4)
  plt.xlabel("Semester") 
  plt.ylabel('SGPA') 
  plt.title(f"SGPA per semester of {data[index][1]}") 

#   plt.xticks(X_axis+width,group_labels) 
#   plt.legend()
  plt.show()

plotGraph(search_index)
