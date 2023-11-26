
from flask import Flask, request, render_template 
from youtube_transcript_api import YouTubeTranscriptApi
from datetime import datetime
 
# Flask constructor
app = Flask(__name__)   
 
@app.route('/formTest')
def GreetUser():
   return render_template('form.html')

@app.route('/formDisplay', methods =["GET", "POST"])
def result():
   start = datetime.now()
   if request.method == 'POST':
      firstName = request.form['fname']
      lastName = request.form['lname']
    #   video_id = request.form.get('videoID')
      video_id = request.form['videoID']
      fullName = firstName + " " + lastName
      print("Video Id is: "+video_id)
      final_transcript = ''
      transcript = YouTubeTranscriptApi.get_transcript(video_id)
    #   transcript_list = YouTubeTranscriptApi.list_transcripts('video_id')
    #   transcript = transcript_list.find_transcript(['en'])
    #   print(transcript.fetch()) 
      for i in range(0,len(transcript)):
         final_transcript = final_transcript + " " + transcript[i]['text']
      print(final_transcript)
      transcript_len = 0
      for i in final_transcript:
        if i == ' ':
            transcript_len = transcript_len + 1
      end = datetime.now()
      td = (end - start).total_seconds() * 10**3
      print("Time taken to generate transcript is: ",td)
      return render_template('display.html', 
        fullname = fullName, 
        firstname = firstName, 
        lastname = lastName,
        videoId = video_id,
        Transcript = final_transcript,
        transcript_length = transcript_len,
        timeTaken = td
      )
    
if __name__=='__main__':
   app.run(debug=True)