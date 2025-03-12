import re
import torch
import streamlit as st
from transformers import pipeline
from youtube_transcript_api import YouTubeTranscriptApi

# Set device to CPU
device = "cpu"

# Function to extract video transcript with timestamps
def get_video_transcript_with_timestamps(video_id):
    try:
        transcript = YouTubeTranscriptApi.get_transcript(video_id)
        return transcript
    except Exception as e:
        return f"Error: {str(e)}"

# Function to preprocess text
def preprocess_text(text):
    text = re.sub(r'\[.*?\]', '', text)  # Remove timestamps like [00:00]
    text = re.sub(r'\s+', ' ', text)     # Remove extra spaces
    return text.strip()

# Function to summarize long text and get timestamps
def summarize_long_text_with_timestamps(transcript, max_chunk_length=1024):
    summarizer = pipeline("summarization", model="facebook/bart-large-cnn", device=device)
    
    # Extract text and timestamps
    text = " ".join([line['text'] for line in transcript])
    timestamps = [line['start'] for line in transcript]
    
    # Split text into chunks
    chunks = [text[i:i + max_chunk_length] for i in range(0, len(text), max_chunk_length)]
    summaries = []
    
    for chunk in chunks:
        summary = summarizer(chunk, max_length=50, min_length=10, do_sample=False)
        summaries.append(summary[0]['summary_text'])
    
    # Combine summaries
    combined_summary = " ".join(summaries)
    
    # Map summary sentences back to timestamps
    summary_sentences = combined_summary.split('. ')
    summary_with_timestamps = []
    
    for sentence in summary_sentences:
        # Find the first occurrence of the sentence in the original text
        start_idx = text.find(sentence)
        if start_idx != -1:
            # Find the corresponding timestamp
            for i, line in enumerate(transcript):
                if line['text'] in text[start_idx:start_idx + len(sentence)]:
                    timestamp = line['start']
                    summary_with_timestamps.append((sentence, timestamp))
                    break
    
    return summary_with_timestamps

# Streamlit App
st.title("Video Transcript Summarizer with Timestamps")
st.write("Enter a YouTube Video URL or paste a transcript to generate a summary with timestamps.")

video_url = st.text_input("YouTube Video URL or Transcript")

if video_url:
    if "youtube.com" in video_url or "youtu.be" in video_url:
        # Extract video ID from URL
        if "v=" in video_url:
            video_id = video_url.split("v=")[1].split("&")[0]
        else:
            video_id = video_url.split("/")[-1]
        
        # Get transcript with timestamps
        transcript = get_video_transcript_with_timestamps(video_id)
        if "Error" in transcript:
            st.error(transcript)
        else:
            st.write("### Transcript")
            st.write(" ".join([line['text'] for line in transcript]))
            
            # Preprocess and summarize with timestamps
            summary_with_timestamps = summarize_long_text_with_timestamps(transcript)
            
            st.write("### Summary with Timestamps")
            for sentence, timestamp in summary_with_timestamps:
                st.write(f"{sentence} (Timestamp: {timestamp:.2f}s)")
    else:
        # Treat input as direct transcript
        cleaned_transcript = preprocess_text(video_url)
        # Since we don't have timestamps for direct input, we can't provide them
        summary = summarize_long_text(cleaned_transcript)
        
        st.write("### Summary")
        st.write(summary)
