import pandas as pd
import textwrap
import nltk 
import string
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from rouge_score import rouge_scorer
import heapq
import numpy as np  
import matplotlib.pyplot as plt  
from rouge import Rouge
import evaluate

df = pd.read_csv('./datasets/news_summary(50-200).csv',encoding = "unicode_escape")

def wrap(x):
  return textwrap.fill(x, width= 100,replace_whitespace = False, fix_sentence_endings = True)

# nltk.download('punkt')
# nltk.download('stopwords')
stopwords = nltk.corpus.stopwords.words('english')
# print(stopwords)

def preprocess(text):
  formatted_text = text.lower()
  indented_text = ''
  final_text = ''
  tokens = []
  # print(len(formatted_text))
  for i in range(0,len(formatted_text)-1):
    if text[i] == '.' and text[i+1] != ' ':
      indented_text = indented_text + str(text[i]) + ' '
    else:
      indented_text = str(indented_text) + str(text[i])
  indented_text = indented_text + '.'
  # print(f"Indented text \n {indented_text}")
  for token in nltk.word_tokenize(indented_text):
    if token not in stopwords and token not in string.punctuation:
      tokens.append(token)
  for words in tokens:
    final_text = final_text + ' ' + words
  return final_text

formatted_text = ''

# print(df.head(10))
# print(df.info()) 
# print(f"Read More: \n {wrap(df.loc[2]['read_more'])}")
# print(f"Text: \n {wrap(df.loc[2]['text'])}")
# print(f"Complete Text \n {wrap(df.loc[2]['ctext'])}")
formatted_text = preprocess(df.loc[2]['ctext'])
# print(f"\n Text after preprocessing: \n {formatted_text}")

word_frequency = nltk.FreqDist(nltk.word_tokenize(formatted_text))
print(f"\n{word_frequency}\n")
# print(f"\n{word_frequency.keys()}")
# print(len(word_frequency.keys()))
print(f"\n{word_frequency.values()}")
print(f"\n{word_frequency.items()}")

highest_frequency = max(word_frequency.values())
# print(f"\n Highest frequency is: {highest_frequency}")

for word in word_frequency.keys():
  word_frequency[word] = (word_frequency[word] / highest_frequency)
# print(f"\n{word_frequency.items()}")

# sentence_list = nltk.sent_tokenize(df.loc[2]['ctext'])
sentence_list = (df.loc[2]['ctext']).split('.')
# print(f"\nSentence Text is: {sentence_list}")
# print(f"\nLength of Sentence Text is: {len(sentence_list)}")
# for i in sentence_list:
#   print(i)

score_sentences = {}
for sentence in sentence_list:
  for word in nltk.word_tokenize(sentence.lower()):
    if sentence not in score_sentences.keys():
      score_sentences[sentence] = word_frequency[word]
    else:
      score_sentences[sentence] += word_frequency[word]
print(f"\n{score_sentences.values()}")
print(f"\n{score_sentences.items()}")

best_sentences = heapq.nlargest(int(len(sentence_list)*0.50), score_sentences, key = score_sentences.get)
# print(f"\n Best sentences are: {best_sentences}")

summary = ' '.join(best_sentences)
# summary = "the cat was found under the bed"
def generateSummary(index,percentage):
  formatted_text = ''
  formatted_text = preprocess(df.loc[index]['ctext'])
  word_frequency = nltk.FreqDist(nltk.word_tokenize(formatted_text))
  highest_frequency = max(word_frequency.values())
  for word in word_frequency.keys():
    word_frequency[word] = (word_frequency[word] / highest_frequency)
  sentence_list = (df.loc[index]['ctext']).split('.')
  score_sentences = {}
  for sentence in sentence_list:
    for word in nltk.word_tokenize(sentence.lower()):
      if sentence not in score_sentences.keys():
        score_sentences[sentence] = word_frequency[word]
      else:
        score_sentences[sentence] += word_frequency[word]
  best_sentences = heapq.nlargest(int(len(sentence_list)*percentage/100), score_sentences, key = score_sentences.get)
  summary = ' '.join(best_sentences)
  return summary

print(f"\nFinal summary: {summary}")

reference_summary = df.loc[2]['text']
# reference_summary = "the cat was under the bed"
print(f"\nReference summary: {reference_summary}")

# rouge = evaluate.load('rouge')
# results = rouge.compute(predictions = [summary], references=[reference_summary])
# print(f"\n rouge by evaluate: {results}")

scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
scores = scorer.score(reference_summary, summary)
# print(f'\nRouge score using rouge_scorer:')
# for key in scores:
#     print(f'\n{key}: {scores[key]}')

numeric_score = []
numeric_score_rougewise = []

for key in scores:
  score_as_token = nltk.word_tokenize(str(scores[key]))
  for i in score_as_token:
    if i not in string.punctuation and i !='Score':
      numeric_score.append(float(i.split('=')[1]))
# print(numeric_score)

for i in range(0,len(numeric_score),3):
  numeric_score_rougewise.append([numeric_score[i],numeric_score[i+1],numeric_score[i+2]])
print(numeric_score_rougewise)

def calculate_Rouge_as_number(index,metric):
  reference_summary = df.loc[index]['text']
  summary = generateSummary(index,60)
  scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
  scores = scorer.score(reference_summary, summary)
  numeric_score = []
  numeric_score_rougewise = []
  for key in scores:
    score_as_token = nltk.word_tokenize(str(scores[key]))
    for i in score_as_token:
      if i not in string.punctuation and i !='Score':
        numeric_score.append(float(i.split('=')[1]))
  for i in range(0,len(numeric_score),3):
    numeric_score_rougewise.append([numeric_score[i],numeric_score[i+1],numeric_score[i+2]])
  if(metric == 'Rouge-1'):
    return numeric_score_rougewise[0]
  elif(metric == 'Rouge-2'):
    return numeric_score_rougewise[1]
  elif(metric == 'Rouge-L'):
    return numeric_score_rougewise[2]
  elif(metric == ''):
    return numeric_score_rougewise

def plotGraph(arr):
  group_labels = ["Rouge-1","Rouge-2","Rouge-L"]
  width = 0.25

  Rouge1 = [arr[0][0],arr[1][0],arr[2][0]]
  Rouge2 = [arr[0][1],arr[1][1],arr[2][1]]
  RougeL = [arr[0][2],arr[1][2],arr[2][2]]
  print(f"\n{Rouge1}\n{Rouge2}\n{RougeL}")

  X_axis = np.arange(len(group_labels)) 

  bar1 = plt.bar(X_axis, Rouge1, width, color = 'r', label = 'Precision') 
  bar2 = plt.bar(X_axis + width, Rouge2, width, color='g', label = 'Recall')
  bar3 = plt.bar(X_axis+width*2, RougeL, width, color = 'b', label = 'F-measure') 

  plt.xlabel("Parameters") 
  plt.ylabel('Scores') 
  plt.title("Rouge Score for Frequency based summarization") 

  plt.xticks(X_axis+width,group_labels) 
  plt.legend()
  plt.show()

plotGraph(numeric_score_rougewise)

print(f"\nTrying getSummary and calculate_Rouge_as_number functions for 5 items in dataset:")
temp1 = []
# temp2 = np.array()
for i in range(0,100):
   rouge_values = calculate_Rouge_as_number(i,'')
   temp1.append(rouge_values)
# print(temp1[0])
# temp2 = np.array([temp1[0],temp1[1],temp1[2],temp1[3],temp1[4]])
temp2 = np.array(temp1)
print(len(temp2))
sum_numeric_score_rougewise = temp2.sum(axis=0)
avg_numeric_score_rougewise = np.mean(temp2,axis=0)
# print(temp)
print(f"\n{sum_numeric_score_rougewise}")
print(f"\n{avg_numeric_score_rougewise}")

plotGraph(avg_numeric_score_rougewise)

# [0.21628084850687224, 0.12434868507198817, 0.14425589344428189]
# [0.7880579053895138, 0.4425428546556421, 0.5161604918666389]
# [0.3242936221162833, 0.18521772488609312, 0.21516997714826694] len = 60% 

# [0.23321744469954614, 0.13071640696614736, 0.15309978414065997]
# [0.7549387037515529, 0.41190203138662446, 0.48926771138405933]
# [0.33964590592962574, 0.18868295129428594, 0.22228222280927049] len = 50%

# [0.26792127557218304, 0.14413826186924536, 0.1757277003810136]
# [0.700527200149833, 0.36665056778130556, 0.44708416668216777]
# [0.3626828716278916, 0.19206872107172437, 0.23496946928732218] len = 40%

# [0.29334432131157956, 0.14837517750251986, 0.19648647168346334]
# [0.6410570874315031, 0.3180334655563443, 0.4100780160548531]
# [0.37299678828780153, 0.18640989906814087, 0.2445857761266697] len = 33%

# [0.3228284362853175, 0.15590081721195234, 0.21741756175014174]
# [0.580237807179467, 0.27152022054413105, 0.3731641860485865]
# [0.3858719215313593, 0.18402139768700942, 0.25469610540587334] len = 25%