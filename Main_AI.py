import streamlit as st
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
from nltk.tokenize import sent_tokenize
import plotly.express as px
import re
import pandas as pd

# Load CodeBERT tokenizer and model
tokenizer = AutoTokenizer.from_pretrained("YoussefHassan/codeBert-Classifier")
model = AutoModelForSequenceClassification.from_pretrained("YoussefHassan/codeBert-Classifier")

# Advanced Preprocessing
# def preprocess_text(text):
#     """Preprocess the input text by cleaning and normalizing it."""
#     text = re.sub(r'\s+', ' ', text).strip()
#     text = re.sub(r'#.*', '', text)
#     text = re.sub(r'(\"\"\".*?\"\"\"|\'\'\'.*?\'\'\')', '', text, flags=re.DOTALL)
#     return text

def detect_ai_generated_code(text, chunk_size=512):
    tokens = tokenizer.encode(text)
    total_length = len(tokens)
    scores = []
    for i in range(0, total_length, chunk_size):
        chunk = tokenizer.decode(tokens[i:i + chunk_size])
        encoded_input = tokenizer(chunk, return_tensors="pt", padding=True, truncation=True, max_length=chunk_size)
        with torch.no_grad():
            outputs = model(**encoded_input)
            probabilities = torch.nn.functional.softmax(outputs.logits, dim=-1)
            scores.append(probabilities[0][1].item())
    ai_generated_score = sum(scores) / len(scores)
    return ai_generated_score

def analyze_suspicious_parts(text, chunk_size=512):
    """Analyze each line of the text to detect suspicious parts."""
    sentences = sent_tokenize(text)
    suspicious_scores = []
    for sentence in sentences:
        tokens = tokenizer.encode(sentence)
        total_length = len(tokens)
        scores = []
        for i in range(0, total_length, chunk_size):
            chunk = tokenizer.decode(tokens[i:i + chunk_size])
            score = detect_ai_generated_code(chunk)
            scores.append(score)
        average_score = sum(scores) / len(scores)
        suspicious_scores.append((sentence, average_score))
    return suspicious_scores

def plot_suspicious_parts(suspicious_scores):
    """Visualize the suspicious parts of the code."""
    df = pd.DataFrame(suspicious_scores, columns=['Code Segment', 'Suspicious Score'])
    fig = px.bar(
        df,
        x='Suspicious Score', 
        y='Code Segment', 
        orientation='h',
        labels={'Suspicious Score': 'Suspicious Score', 'Code Segment': 'Code Segment'},
        title='Suspicious Code Analysis'
    )
    st.plotly_chart(fig, use_container_width=True)

# Streamlit Page Configuration
st.set_page_config(page_title="AI Code Detector", layout="wide", initial_sidebar_state="expanded")

# Sidebar for Navigation and Settings
st.sidebar.title("Settings")
st.sidebar.subheader("Options")
model_choice = st.sidebar.radio("Choose Model Type", ["CodeBERT (AI Detection)", "Other"])
chunk_size = st.sidebar.slider("Chunk Size", min_value=256, max_value=1024, value=512, step=128)

# Main Title
st.title("CodeBERT AI Code Detector")
st.markdown("""
    <style>
    .css-1q6mrw7 {
        font-size: 1.5em;
        color: #0073e6;
    }
    </style>
    """, unsafe_allow_html=True)

# Text Area for Input Code
text_area = st.text_area("Enter code or text", height=300)

# If there is text input, run analysis on button click
if text_area:
    if st.button("Analyze"):
        with st.spinner("Analyzing... This may take a few seconds."):
           # preprocessed_text = preprocess_text(text_area)

            # Columns for displaying output
            col1, col2 = st.columns([1, 1])

            with col1:
                st.info("Your Input Text")
                st.text_area("Input Code", value=text_area, height=200)

            with col2:
                st.info("AI-Generated Score")
                ai_generated_score = detect_ai_generated_code(text_area, chunk_size)
                st.write(f"AI-Generated Score: {ai_generated_score:.4f}")

                if ai_generated_score > 0.5:
                    st.error("Likely AI-generated content")
                else:
                    st.success("Likely not AI-generated content")

        st.warning("""
            Disclaimer: AI plagiarism detector apps can assist in identifying potential instances of plagiarism; however, 
            it is important to note that their results may not be entirely flawless or completely reliable. 
            Use human judgment alongside AI tools for accurate verification.
        """)

# Footer with some information
st.markdown("""
    <footer style="text-align:center; padding-top:20px;">
        <p><i>Developed by Mohammed Yousef</i></p>
        <p><a href="https://github.com/your-github-link" target="_blank">GitHub</a> | <a href="https://linkedin.com/in/your-linkedin-link" target="_blank">LinkedIn</a></p>
    </footer>
""", unsafe_allow_html=True)
