import streamlit as st
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score

# Load models
graphcodebert_tokenizer = AutoTokenizer.from_pretrained("YoussefHassan/graphcodebert-plagiarism-detector")
graphcodebert_model = AutoModelForSequenceClassification.from_pretrained("YoussefHassan/graphcodebert-plagiarism-detector")
codet5_tokenizer = AutoTokenizer.from_pretrained("YoussefHassan/codet5-multiclass-plagiarism-detector")
codet5_model = AutoModelForSequenceClassification.from_pretrained("YoussefHassan/codet5-multiclass-plagiarism-detector")
unixcoder_tokenizer = AutoTokenizer.from_pretrained("YoussefHassan/unixcoder-multiclass-plagiarism-detector")
unixcoder_model = AutoModelForSequenceClassification.from_pretrained("YoussefHassan/unixcoder-multiclass-plagiarism-detector")

def softmax(logits):
    exp_logits = np.exp(logits - np.max(logits))
    return exp_logits / exp_logits.sum(axis=-1, keepdims=True)

def ensemble_prediction(original_code, submission_code):
    # Tokenize inputs
    inputs = {"input_ids": None, "attention_mask": None}

    # GraphCodeBERT
    graphcodebert_inputs = graphcodebert_tokenizer(
        original_code + " " + submission_code, return_tensors="pt", padding=True, truncation=True
    )
    graphcodebert_logits = graphcodebert_model(**graphcodebert_inputs).logits.detach().numpy()
    graphcodebert_softmax = softmax(graphcodebert_logits)

    # CodeT5
    codet5_inputs = codet5_tokenizer(
        original_code + " " + submission_code, return_tensors="pt", padding=True, truncation=True
    )
    codet5_logits = codet5_model(**codet5_inputs).logits.detach().numpy()
    codet5_softmax = softmax(codet5_logits)

    # UniXcoder
    unixcoder_inputs = unixcoder_tokenizer(
        original_code + " " + submission_code, return_tensors="pt", padding=True, truncation=True
    )
    unixcoder_logits = unixcoder_model(**unixcoder_inputs).logits.detach().numpy()
    unixcoder_softmax = softmax(unixcoder_logits)

    # Ensemble: Sum SoftMax vectors
    combined_softmax = graphcodebert_softmax + codet5_softmax + unixcoder_softmax
    predicted_label = combined_softmax.argmax()
    return predicted_label, combined_softmax

def explain_decision(original_code, submission_code, predicted_label):
    level_explanations = {
        0: "No Plagiarism: The submitted code is completely original and shows no significant similarity to any other source.",
        1: "Level-1 (Comment and Whitespace Modification): The code shows minimal changes, such as modifying or removing comments or adjusting the formatting and spacing, but the actual logic remains unchanged.",
        2: "Level-2 (Identifier Modification): The code has undergone basic changes, such as renaming variables, functions, or classes. However, the underlying logic and structure remain the same.",
        3: "Level-3 (Component Declaration Relocation): The code structure is slightly altered, such as moving variable or component declarations around. The functional logic remains unaffected.",
        4: "Level-4 (Method Structure Change): The code demonstrates moderate changes, such as encapsulating certain statements into methods or reorganizing blocks of logic. However, the primary functionality is still similar to the original.",
        5: "Level-5 (Program Statement Replacement): The code is significantly modified, with certain programmatic elements (like loops or traversal logic) replaced or restructured. The solution's approach, however, is still derived from the original source.",
        6: "Level-6 (Logic Change): The code exhibits substantial transformation, such as replacing an iterative solution with recursion or making major changes to the logic. While the essence may be inspired by the original, the implementation is fundamentally different."
    }

    explanation = f"""
    **Predicted Level**: Level {predicted_label}  
    **Description**: {level_explanations.get(predicted_label, "Unknown level. Please verify the input.")} 
    """
    return explanation

# Streamlit UI
st.set_page_config(page_title="Code Plagiarism Detector", page_icon=":page_facing_up:", layout="wide")
st.title("🛡️ Code Plagiarism Detection System")
st.markdown("### Analyze code snippets for potential plagiarism.")

# Input Section
st.sidebar.header("User Inputs")
original_code = st.sidebar.text_area("Original Code", height=200, placeholder="Paste the original code here...")
submission_code = st.sidebar.text_area("Submission Code", height=200, placeholder="Paste the submission code here...")

if st.sidebar.button("Analyze"):
    if original_code.strip() and submission_code.strip():
        with st.spinner("Analyzing the codes..."):
            predicted_label, combined_softmax = ensemble_prediction(original_code, submission_code)
            explanation = explain_decision(original_code, submission_code, predicted_label)
        st.success("✅ Analysis Complete!")
        st.markdown(f"### 🏷️ Predicted Label: **Level {predicted_label}**")
        st.markdown("### Explanation:")
        st.write(explanation)
    else:
        st.error("Both code snippets are required!")

# Upload and Test Section
st.markdown("---")
st.markdown("### 📊 Bulk Analysis (Upload CSV)")
st.markdown("You can upload a CSV file with columns `OriginalCode` and `SubmissionCode` to analyze multiple cases at once.")
uploaded_file = st.file_uploader("Upload Test Data (CSV)", type=["csv"])


# Footer
st.markdown("---")
st.markdown("""
#### 🚀 About
This app uses fine-tuned models (GraphCodeBERT, CodeT5, and UniXcoder) to detect plagiarism in programming code. The analysis is based on an ensemble approach to improve prediction accuracy.
""")
