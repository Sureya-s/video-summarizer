import os
os.environ["STREAMLIT_WATCHDOG"] = "false"
import re
import torch
import streamlit as st
from transformers import pipeline
from youtube_transcript_api import YouTubeTranscriptApi

# Set device to CPU
device = "cpu"

# Function to extract video transcript
def get_video_transcript(video_id):
    try:
        transcript = YouTubeTranscriptApi.get_transcript(video_id)
        text = " ".join([line['text'] for line in transcript])
        return text
    except Exception as e:
        return f"Error: {str(e)}"

# Function to preprocess text
def preprocess_text(text):
    text = re.sub(r'\[.*?\]', '', text)  # Remove timestamps like [00:00]
    text = re.sub(r'\s+', ' ', text)     # Remove extra spaces
    return text.strip()

# Function to summarize long text
def summarize_long_text(text, max_chunk_length=1024):
    summarizer = pipeline("summarization", model="facebook/bart-large-cnn", device=device)
    chunks = [text[i:i + max_chunk_length] for i in range(0, len(text), max_chunk_length)]
    summaries = []
    for chunk in chunks:
        summary = summarizer(chunk, max_length=50, min_length=10, do_sample=False)
        summaries.append(summary[0]['summary_text'])
    return " ".join(summaries)

# Streamlit App
st.title("Video Transcript Summarizer")
st.write("Enter a YouTube Video URL or paste a transcript to generate a summary.")

video_url = st.text_input("YouTube Video URL or Transcript")

if video_url:
    if "youtube.com" in video_url or "youtu.be" in video_url:
        # Extract video ID from URL
        if "v=" in video_url:
            video_id = video_url.split("v=")[1].split("&")[0]
        else:
            video_id = video_url.split("/")[-1]
        
        # Get transcript
        transcript = get_video_transcript(video_id)
        if "Error" in transcript:
            st.error(transcript)
        else:
            st.write("### Transcript")
            st.write(transcript)
            
            # Preprocess and summarize
            cleaned_transcript = preprocess_text(transcript)
            summary = summarize_long_text(cleaned_transcript)
            
            st.write("### Summary")
            st.write(summary)
    else:
        # Treat input as direct transcript
        cleaned_transcript = preprocess_text(video_url)
        summary = summarize_long_text(cleaned_transcript)
        
        st.write("### Summary")
        st.write(summary)
