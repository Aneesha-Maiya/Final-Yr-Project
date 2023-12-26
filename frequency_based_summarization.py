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

df = pd.read_csv('./datasets/news_summary.csv',encoding = "unicode_escape")

def wrap(x):
  return textwrap.fill(x, width= 100,replace_whitespace = False, fix_sentence_endings = True)

# nltk.download('punkt')
# nltk.download('stopwords')
stopwords = nltk.corpus.stopwords.words('english')
print(stopwords)

def preprocess(text):
  formatted_text = text.lower()
  indented_text = ''
  final_text = ''
  tokens = []
  tokens_without_stopwords = []
  tokens_without_punctuations = []
  print(len(formatted_text))
  for i in range(0,len(formatted_text)-1):
    if text[i] == '.' and text[i+1] != ' ':
      indented_text = indented_text + str(text[i]) + ' '
    else:
      indented_text = str(indented_text) + str(text[i])
  indented_text = indented_text + '.'
  print(f"Indented text \n {indented_text}")
  for token in nltk.word_tokenize(indented_text):
    tokens.append(token)
  print(f"\nTokens are \n {tokens} Length of tokens: {len(tokens)} \n")
  for word in tokens:
    if word not in stopwords:
      tokens_without_stopwords.append(word)
  print(f"\nTokens without stopwords are \n {tokens_without_stopwords} Length of tokens without stopwords: {len(tokens_without_stopwords)} \n")
  for token in tokens_without_stopwords:
    if token not in string.punctuation:
      tokens_without_punctuations.append(token)
  print(f"\nTokens without punctuations are \n {tokens_without_punctuations} Length of tokens without stopwords: {len(tokens_without_punctuations)} \n")
  for words in tokens_without_punctuations:
    final_text = final_text + ' ' + words
  return final_text

formatted_text = ''

print(df.head(10))
print(df.info()) 
print(f"Read More: \n {wrap(df.loc[2]['read_more'])}")
print(f"Text: \n {wrap(df.loc[2]['text'])}")
print(f"Complete Text \n {wrap(df.loc[2]['ctext'])}")
formatted_text = preprocess(df.loc[2]['ctext'])
print(f"\n Text after preprocessing: \n {formatted_text}")

word_frequency = nltk.FreqDist(nltk.word_tokenize(formatted_text))
print(f"\n{word_frequency}\n")
print(f"\n{word_frequency.keys()}")
print(len(word_frequency.keys()))
print(f"\n{word_frequency.values()}")
print(f"\n{word_frequency.items()}")

highest_frequency = max(word_frequency.values())
print(f"\n Highest frequency is: {highest_frequency}")

for word in word_frequency.keys():
  word_frequency[word] = (word_frequency[word] / highest_frequency)
print(f"\n{word_frequency.items()}")

# sentence_list = nltk.sent_tokenize(df.loc[2]['ctext'])
sentence_list = (df.loc[2]['ctext']).split('.')
print(f"\nSentence Text is: {sentence_list}")
print(f"\nLength of Sentence Text is: {len(sentence_list)}")
for i in sentence_list:
  print(i)

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
print(f"\n Best sentences are: {best_sentences}")

summary = ' '.join(best_sentences)
# summary = "the cat was found under the bed"
print(f"\nFinal summary: {summary}")

reference_summary = df.loc[2]['text']
# reference_summary = "the cat was under the bed"
print(f"\nReference summary: {reference_summary}")

rouge = evaluate.load('rouge')
results = rouge.compute(predictions = [summary], references=[reference_summary])
print(f"\n rouge by evaluate: {results}")

scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
scores = scorer.score(reference_summary, summary)
print(f'\nRouge score using rouge_scorer:')
for key in scores:
    print(f'\n{key}: {scores[key]}')

numeric_score = []
numeric_score_rougewise = []

for key in scores:
  score_as_token = nltk.word_tokenize(str(scores[key]))
  for i in score_as_token:
    if i not in string.punctuation and i !='Score':
      numeric_score.append(float(i.split('=')[1]))
print(numeric_score)

for i in range(0,len(numeric_score),3):
  numeric_score_rougewise.append([numeric_score[i],numeric_score[i+1],numeric_score[i+2]])
print(numeric_score_rougewise)

group_labels = ["Rouge-1","Rouge-2","Rouge-L"]
width = 0.25

Rouge1 = numeric_score_rougewise[0]
Rouge2 = numeric_score_rougewise[1]
RougeL = numeric_score_rougewise[2]
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

# reference_summary = reference_summary
# generated_summary = summary

# def evaluate_rouge(hypothesis, reference):
#     rouge = Rouge()
#     scores = rouge.get_scores(hypothesis, reference)
#     return scores
# rouge_scores = evaluate_rouge(summary, reference_summary)

# print(f"\nRouge scores using rouge {rouge_scores}")
# for metric, score in rouge_scores[0].items():
#     print(f"{metric}: {score}")

# count = 0
# for i in nltk.word_tokenize(reference_summary):
#   if i in nltk.word_tokenize(summary) and i not in string.punctuation and i not in stopwords:
#     count = count + 1
#     print(i)

# sum_len = 0
# for i in nltk.word_tokenize(summary):
#   if i not in string.punctuation and i not in stopwords:
#     sum_len = sum_len + 1

# ref_len = 0
# for i in nltk.word_tokenize(reference_summary):
#   if i not in string.punctuation and i not in stopwords:
#     ref_len = ref_len + 1

# print(f"\n Length of summary {sum_len}")
# print(f"\n Length of reference_summary {ref_len}")
# print(f"\n Count is {count}")

# recall = count/ref_len
# precision = count/sum_len
# fscore = 2 * (recall * precision)/(recall + precision)
# print(f"\n Recall: {recall}\n Precision: {precision} \n Fscore: {fscore}")

