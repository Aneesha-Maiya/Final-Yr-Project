
from flask import Flask, request, render_template 
from youtube_transcript_api import YouTubeTranscriptApi
from transformers import pipeline, BartTokenizer, BartForConditionalGeneration
from datetime import datetime
 
# Flask constructor
app = Flask(__name__)   
 
@app.route('/formTest')
def GreetUser():
   return render_template('form.html')

@app.route('/formDisplay', methods =["GET", "POST"])
def result():
   startT = datetime.now()
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

      endT = datetime.now()
      tdT = (endT - startT).total_seconds() * 10**3
      print("Time taken to generate transcript is: ",tdT)

      startS = datetime.now()
      tokenizer = BartTokenizer.from_pretrained("facebook/bart-large-cnn")
      model = BartForConditionalGeneration.from_pretrained("facebook/bart-large-cnn")

      def generate_summary(chunk):
    # Tokenize the input text
        input_ids = tokenizer.encode(chunk, return_tensors="pt", max_length=1024, truncation=True)

        # Generate the summary
        summary_ids = model.generate(input_ids, max_length=200, min_length=100, length_penalty=2.0, num_beams=4, early_stopping=True)

        # Decode the generated summary
        summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)

        return summary

      long_text = final_transcript
      # Set the maximum token limit per chunk
      max_token_limit = 1000
      # Split the long text into chunks
      final_summary = []

      text_chunks = [long_text[i:i + max_token_limit] for i in range(0, len(long_text), max_token_limit)]

      for i, chunk in enumerate(text_chunks, 1):
        print(f"\n--- Chunk {i} ---\n")
        print(chunk)
        summary = generate_summary(chunk)
        final_summary.append(summary)
        print("\nSummary:")
        print(summary)

      complete_summary = " ".join(final_summary)
      print("Complete Summary:")
      print(complete_summary)

      summary_len = 0
      for i in complete_summary:
        if i == ' ':
            summary_len = summary_len + 1

      endS = datetime.now()
      tdS = (endS - startS).total_seconds()
      print(f"Time taken to generate Summary is: {tdS} secs ")

      return render_template('display.html', 
        fullname = fullName, 
        firstname = firstName, 
        lastname = lastName,
        videoId = video_id,
        Transcript = final_transcript,
        transcript_length = transcript_len,
        Summary = complete_summary,
        summary_length = summary_len,
        timeTakenT = tdT,
        timeTakenS = tdS
      )
    
if __name__=='__main__':
   app.run(debug=True)