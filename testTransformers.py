from transformers import pipeline, BartTokenizer, BartForConditionalGeneration
from youtube_transcript_api import YouTubeTranscriptApi
from datetime import datetime
import pandas as pd
import numpy as np
import nltk 
import string
from rouge_score import rouge_scorer
import matplotlib.pyplot as plt

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
stopwords = nltk.corpus.stopwords.words('english')
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

df = pd.read_csv('./datasets/news_summary(50-200).csv',encoding = "unicode_escape")
# Load pre-trained BART model and tokenizer
start = datetime.now()
tokenizer = BartTokenizer.from_pretrained("facebook/bart-large-cnn")
model = BartForConditionalGeneration.from_pretrained("facebook/bart-large-cnn")
transcript_list = YouTubeTranscriptApi.get_transcript('FXXWHa4CpC8')
transcript = ' '.join([d['text'] for d in transcript_list])

# Function to generate summary for a given chunk of text
def generate_summary(chunk,word_cnt,percentage):
    print(f"Min lenght is {int(word_cnt*percentage)}")
    # Tokenize the input text
    input_ids = tokenizer.encode(chunk, return_tensors="pt", max_length=1024, truncation=True)

    # Generate the summary
    summary_ids = model.generate(input_ids, max_length=160, min_length= int(word_cnt*percentage), 
    length_penalty=2.0, num_beams=4, early_stopping=True)

    # Decode the generated summary
    summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)

    return summary

# Your long input text
# long_text = df.loc[2]['ctext']

# # Set the maximum token limit per chunk
max_token_limit = 1000

# # Split the long text into chunks
# text_chunks = [long_text[i:i + max_token_limit] for i in range(0, len(long_text), max_token_limit)]

# final_summary = []
# # Generate and display summaries for each chunk
# for i, chunk in enumerate(text_chunks, 1):
#     word_count = 0
#     print(f"\n--- Chunk {i} ---\n")
#     print(chunk)
#     for x in chunk:
#         if x == " ":
#             word_count = word_count + 1
#     print(word_count)
#     if(word_count > 100):
#         formatted_text = preprocess(chunk)
#         print(f"Formatted text: \n",formatted_text)
#         summary = generate_summary(chunk)
#         final_summary.append(summary)
#     else:
#         final_summary.append(chunk)
#     print("\nSummary:")
#     print(summary)

# complete_summary = " ".join(final_summary)
# print("Complete Summary:")
# print(complete_summary)
# end = datetime.now()
# td = (end - start).total_seconds()
# print(f"Time taken to generate transcript is: {td} secs ")

# reference_summary = df.loc[2]['text']
# # reference_summary = "the cat was under the bed"
# print(f"\nReference summary: {reference_summary}")

# scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
# scores = scorer.score(reference_summary,complete_summary)
# print(f'\nRouge score using rouge_scorer:')
# for key in scores:
#     print(f'\n{key}: {scores[key]}')

# numeric_score = []
# numeric_score_rougewise = []

# for key in scores:
#   score_as_token = nltk.word_tokenize(str(scores[key]))
#   for i in score_as_token:
#     if i not in string.punctuation and i !='Score':
#       numeric_score.append(float(i.split('=')[1]))
# print(numeric_score)

# for i in range(0,len(numeric_score),3):
#   numeric_score_rougewise.append([numeric_score[i],numeric_score[i+1],numeric_score[i+2]])
# print(numeric_score_rougewise)

def produce_summary(index):
   start = datetime.now()
   complete_summary_len = 0
   reference_summary_len = 0
   long_text_len = 0
   long_text = df.loc[index]['ctext']
   reference_summary = df.loc[index]['text']
   for x in long_text:
      if x == " ":
         long_text_len = long_text_len + 1
   if long_text_len < 500:
      text_chunks = [long_text[i:i + max_token_limit] for i in range(0, len(long_text), max_token_limit)]
      number_of_chunks = len(text_chunks)
      final_summary = []
      for chunk in text_chunks:
          word_count = 0
          for x in chunk:
              if x == " ":
                  word_count = word_count + 1
          if(word_count > 100):
              summary = generate_summary(chunk.lower(),word_count,0.6)
              summary = summary.lower()
              for token in nltk.word_tokenize(summary):
                if token not in string.punctuation:
                  final_summary.append(token+" ")
          else:
              for token in nltk.word_tokenize(chunk):
                if token not in string.punctuation:
                  final_summary.append(token+" ")
      complete_summary = "".join(final_summary)
      for x in complete_summary:
          if x == " ":
            complete_summary_len = complete_summary_len + 1
      for x in reference_summary:
          if x == " ":
            reference_summary_len = reference_summary_len + 1
      end = datetime.now()
      td = (end - start).total_seconds()
      return {
          "long_txt_len": long_text_len,
          "csummary_len" : complete_summary_len,
          "rsummary_len" : reference_summary_len,
          "long_txt": long_text,
          "csummary": complete_summary,
          "rsummary": reference_summary,
          "total_chunks": number_of_chunks,
          "time_taken": td
        }

# print(produce_summary(5))
time_values = []
long_text_len_values = []
csumm_lens = []
rsumm_lens = []

def calculate_Rouge_as_number(index,metric):
  reference_summary = df.loc[index]['text']
  summary = produce_summary(index)
  print(f"Summary generated for {index} item in {summary['time_taken']} seconds")
  time_values.append(summary['time_taken'])
  long_text_len_values.append(summary['long_txt_len'])
  csumm_lens.append(summary['csummary_len'])
  rsumm_lens.append(summary['rsummary_len'])
  scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
  scores = scorer.score(reference_summary, summary['csummary'])
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
  
print(f"\nTrying getSummary and calculate_Rouge_as_number functions for 10 items in dataset:")
temp1 = []
skip = 0
# temp2 = np.array()
for i in range(10,30):
   long_text_length = 0
   long_text = df.loc[i]['ctext']
   for x in long_text:
      if x == " ":
         long_text_length = long_text_length + 1
   if(long_text_length < 500):
    rouge_values = calculate_Rouge_as_number(i,'')
    temp1.append(rouge_values)
   else:
      print(f"Skipped {i} item")
      skip = skip + 1
      pass
for i in range(30,30+skip):
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

print(f"time taken for each summary: {time_values}")
print(f"long_text_length_values: {long_text_len_values}")
print(f"complete summary lengths: {csumm_lens}")
print(f"reference summary lengths: {rsumm_lens}")

def plotTimeVsLen():
   xpoints = np.array(long_text_len_values)
   ypoints = np.array(time_values)
   xpoints = np.sort(xpoints,axis=None)
   ypoints = np.sort(ypoints,axis=None)
   plt.title("Time Taken(generate summary) vs length (input text)")
   plt.xlabel("Length of input text")
   plt.ylabel("Time taken in seconds")
   plt.plot(xpoints, ypoints,marker='o')
   plt.show()
  
plotTimeVsLen()
total_time_taken = 0
for time in time_values:
   total_time_taken = total_time_taken + time
print(f"Total time taken is: {total_time_taken}")