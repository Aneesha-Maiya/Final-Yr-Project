from youtube_transcript_api import YouTubeTranscriptApi
from datetime import datetime

transcript_list = YouTubeTranscriptApi.get_transcript('q5qAVmXSecQ')
transcript_as_sentence = []
complete_transcript = ''
for  i in transcript_list:
    print(i['text'])
    if(i['text'] != "[Music]"):
        transcript_as_sentence.append(i['text'])
for i in range (0,len(transcript_as_sentence)):
    complete_transcript = complete_transcript + " " + transcript_as_sentence[i]
print(f"Number of sentences: ",len(transcript_as_sentence))
print("Complete transcript: ")
print(complete_transcript)