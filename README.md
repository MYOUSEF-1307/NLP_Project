# Leveraging Fine-Tuned Ensemble Models for Multi-Class Code Plagiarism and AI Code Detection

Plagiarism detection and the identification of AI-generated code in programming
assignments, both in education and professional settings, are some of the most
considerable challenges. Most current state-of-the-art modern tools have problems
with structural transformations, logical refactorings, and other issues caused by the
complexity introduced into AI code generation. In light of this, a Dual Fine-Tuned
Ensemble model framework is proposed as part of the approach that shows the
most promise of finding a solution.
The code plagiarism detection system combined GraphCodeBERT with CodeT5
and UniXcoder, each fine-tuned to capture different code properties, ranging from
identical cases to strongly transformed variants; the overall accuracy of 88% reached
by aggregating their output outperformed most of the state-of-the-art by far while
being among the very top for the most challenging variations.
We used CodeBERT with few-shot learning and prompt engineering to identify
the code generated by AI. This approach differentiates human-written code from
AI-generated code with a high degree of accuracy at 83%
It deploys the framework using a Streamlit interface, hence giving an extremely
user-friendly interface for decision-based analysis. Experimental results prove that
it enhances code integrity analysis, maintains academic ethics, and deals with the
challenges of software education and practice.
