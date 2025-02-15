import re
import torch
import streamlit as st
from transformers import pipeline
from youtube_transcript_api import YouTubeTranscriptApi
from youtube_transcript_api._errors import TranscriptsDisabled, NoTranscriptFound

# Disable Streamlit file watcher
st.set_option('server.fileWatcherType', 'none')

# Set device to CPU
device = "cpu"

# Function to extract video transcript with improved error handling
def get_video_transcript(video_id):
    try:
        # First try to list available transcripts
        transcript_list = YouTubeTranscriptApi.list_transcripts(video_id)
        
        try:
            # Try getting manually created transcript first
            transcript = transcript_list.find_manually_created_transcript(['en'])
        except:
            try:
                # Try getting auto-generated transcript
                transcript = transcript_list.find_generated_transcript(['en'])
            except:
                # Try getting any available transcript
                transcript = transcript_list.find_transcript(['en'])
        
        # Fetch the transcript
        transcript_data = transcript.fetch()
        text = " ".join([line['text'] for line in transcript_data])
        return text
    
    except TranscriptsDisabled:
        return "Error: Transcripts are disabled for this video."
    except NoTranscriptFound:
        return "Error: No transcript found for this video. The video might not have any captions available."
    except Exception as e:
        return f"Error: Could not retrieve transcript. {str(e)}"

# Function to preprocess text
def preprocess_text(text):
    # Remove timestamps like [00:00]
    text = re.sub(r'\[.*?\]', '', text)
    # Remove extra spaces
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

# Function to summarize long text with improved chunk handling
def summarize_long_text(text, max_chunk_length=500):
    try:
        summarizer = pipeline("summarization", model="facebook/bart-large-cnn", device=device)
        
        # Split text into sentences to avoid cutting words
        sentences = re.split('(?<=[.!?])\s+', text)
        chunks = []
        current_chunk = ""
        
        for sentence in sentences:
            if len(current_chunk) + len(sentence) < max_chunk_length:
                current_chunk += " " + sentence
            else:
                if current_chunk:
                    chunks.append(current_chunk.strip())
                current_chunk = sentence
        
        if current_chunk:
            chunks.append(current_chunk.strip())
        
        summaries = []
        for chunk in chunks:
            if len(chunk) < 50:  # Skip very short chunks
                continue
            summary = summarizer(chunk, max_length=130, min_length=30, do_sample=False)
            summaries.append(summary[0]['summary_text'])
        
        return " ".join(summaries)
    except Exception as e:
        return f"Error in summarization: {str(e)}"

# Function to extract video ID from URL
def extract_video_id(url):
    # Handle various YouTube URL formats
    patterns = [
        r'(?:v=|\/)([0-9A-Za-z_-]{11}).*',  # Standard and shortened
        r'(?:embed\/)([0-9A-Za-z_-]{11})',   # Embed URLs
        r'(?:youtu\.be\/)([0-9A-Za-z_-]{11})'  # youtu.be URLs
    ]
    
    for pattern in patterns:
        match = re.search(pattern, url)
        if match:
            return match.group(1)
    return None

# Streamlit App
st.title("Video Transcript Summarizer")
st.write("Enter a YouTube Video URL or paste a transcript to generate a summary.")

# Add file uploader for transcript files
transcript_file = st.file_uploader("Upload a transcript file (optional)", type=['txt'])
video_url = st.text_input("YouTube Video URL or Transcript")

if transcript_file is not None:
    # Handle uploaded transcript file
    transcript_content = transcript_file.read().decode('utf-8')
    cleaned_transcript = preprocess_text(transcript_content)
    summary = summarize_long_text(cleaned_transcript)
    
    st.write("### Summary")
    st.write(summary)

elif video_url:
    if "youtube.com" in video_url or "youtu.be" in video_url:
        # Extract video ID from URL
        video_id = extract_video_id(video_url)
        
        if video_id:
            with st.spinner('Fetching transcript...'):
                transcript = get_video_transcript(video_id)
                
            if transcript.startswith("Error"):
                st.error(transcript)
            else:
                st.write("### Transcript")
                with st.expander("Show full transcript"):
                    st.write(transcript)
                
                with st.spinner('Generating summary...'):
                    # Preprocess and summarize
                    cleaned_transcript = preprocess_text(transcript)
                    summary = summarize_long_text(cleaned_transcript)
                
                st.write("### Summary")
                st.write(summary)
        else:
            st.error("Invalid YouTube URL. Please check the URL and try again.")
    else:
        # Treat input as direct transcript
        cleaned_transcript = preprocess_text(video_url)
        summary = summarize_long_text(cleaned_transcript)
        
        st.write("### Summary")
        st.write(summary)
