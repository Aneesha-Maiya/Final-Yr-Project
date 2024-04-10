from youtube_transcript_api import YouTubeTranscriptApi
from datetime import datetime

start = datetime.now()
video_id = 'oV74Najm6Nc'
transcript_len = 0
transcript1 = YouTubeTranscriptApi.get_transcript(video_id)
transcript_list = YouTubeTranscriptApi.list_transcripts(video_id)
transcript = transcript_list.find_transcript(['en'])
# print(transcript.fetch())
final_transcript = ''
for i in range (0,len(transcript1)):
    # print(transcript1[i]['text'])
    final_transcript = final_transcript + " " + transcript1[i]['text']
for i in range (0,len(transcript1)):
    print(transcript1[i])
for i in final_transcript:
    if i == ' ':
        transcript_len = transcript_len + 1
print("Final Transcript Length:"+str(transcript_len))
print(final_transcript)
print("Final transcript lenght",len(final_transcript))
end = datetime.now()
td = (end - start).total_seconds() * 10**3
print(f"Time taken to generate transcript is: {td} ms")