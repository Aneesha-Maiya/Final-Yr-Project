import os
print(os.getcwd())
os.chdir('../') 
print(os.getcwd())
print(os.listdir())
os.chdir('Downloads')
print(os.getcwd())
os.chdir(r'BBC_Summ\BBC News Summary\News Articles\sport')
print(os.getcwd())
with open('498.txt', 'r') as reader:
    print(reader.readlines())