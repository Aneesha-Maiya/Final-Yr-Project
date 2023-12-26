from transformers import pipeline, BartTokenizer, BartForConditionalGeneration
from youtube_transcript_api import YouTubeTranscriptApi
from datetime import datetime

# Load pre-trained BART model and tokenizer
start = datetime.now()
tokenizer = BartTokenizer.from_pretrained("facebook/bart-large-cnn")
model = BartForConditionalGeneration.from_pretrained("facebook/bart-large-cnn")
transcript_list = YouTubeTranscriptApi.get_transcript('FXXWHa4CpC8')
transcript = ' '.join([d['text'] for d in transcript_list])

# Function to generate summary for a given chunk of text
def generate_summary(chunk):
    # Tokenize the input text
    input_ids = tokenizer.encode(chunk, return_tensors="pt", max_length=1024, truncation=True)

    # Generate the summary
    summary_ids = model.generate(input_ids, max_length=200, min_length=100, length_penalty=2.0, num_beams=4, early_stopping=True)

    # Decode the generated summary
    summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)

    return summary

# Your long input text
long_text = transcript

# Set the maximum token limit per chunk
max_token_limit = 1000

# Split the long text into chunks
text_chunks = [long_text[i:i + max_token_limit] for i in range(0, len(long_text), max_token_limit)]

final_summary = []
# Generate and display summaries for each chunk
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
end = datetime.now()
td = (end - start).total_seconds()
print(f"Time taken to generate transcript is: {td} secs ")