from transformers import pipeline
import textwrap
import numpy as np
import pandas as pd
from pprint import pprint
from youtube_transcript_api import YouTubeTranscriptApi
from datetime import datetime

def wrap(x):
  return textwrap.fill(x, width= 100,replace_whitespace = False, fix_sentence_endings = True)

transcript_list = YouTubeTranscriptApi.get_transcript('o9D0gTqfm6c')
transcript = ' '.join([d['text'] for d in transcript_list])

transcript_len = 0
for i in transcript:
    if i == ' ':
        transcript_len = transcript_len + 1

transcript_split = []
word = ''
for i in range(0,len(transcript)):
    if(transcript[i] != " "):
        word = word + transcript[i]
    else:
        transcript_split.append(word + " ")
        word = ''
wrap_transcript = ''
summary = []
summarizer = pipeline('summarization')
summary_text = ''
final_summary = ''
for i in range(0,(len(transcript_split)//200)*200,200):
    # print("i is: ",i)
    for j in range(i,i+200):
        if(transcript_split[j]):
            wrap_transcript = wrap_transcript + transcript_split[j]
    print("Transcript for "+str(i//200)+")")
    print(wrap(wrap_transcript))
    print()
    summary_txt = summarizer(wrap_transcript)
    final_summary = final_summary + summary_txt[0]['summary_text'] + " "
    wrap_transcript = ''
    print("Summary for "+ str(i//200) +")")
    print(wrap(summary_txt[0]['summary_text']))
    print()
for i in range((len(transcript_split)//200)*200,len(transcript_split)):
    wrap_transcript = wrap_transcript + transcript_split[i]
print(wrap(wrap_transcript))
summary_len = 0
for i in final_summary:
  if i == ' ':
    summmary_len = summary_len + 1
print("Final complete summary is: ")
print(wrap(final_summary))
print("Transcript length: ",transcript_len)
print("Summary length: ",len(final_summary))



# print(wrap(transcript))