import streamlit as st
import joblib
import re

svm = joblib.load('svm_model.joblib')
vectorizer = joblib.load('tfidf_vectorizer.joblib')
FINAL_THRESHOLD = -0.173

def clean_text(text):
    text = text.lower()
    text = re.sub(r'#url_\w+#', ' urltoken ', text)
    text = re.sub(r'<[^>]+>', ' ', text)
    text = re.sub(r'[^a-z0-9\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

feature_names = vectorizer.get_feature_names_out()
coefficients = svm.coef_[0]

st.title("AI Fake Job Posting Detector")
st.write("TF-IDF + Linear SVM trained on the EMSCAD dataset (Kaggle). Paste a job posting to check for signs of fraud.")

job_text = st.text_area("Paste job posting text here", height=300)

if st.button("Submit") and job_text.strip():
    cleaned = clean_text(job_text)
    tfidf_vec = vectorizer.transform([cleaned])
    score = svm.decision_function(tfidf_vec)[0]
    flagged = score >= FINAL_THRESHOLD

    if flagged:
        st.error(f"FLAGGED for review (score: {score:.3f})")
    else:
        st.success(f"Looks legitimate (score: {score:.3f})")

    present_indices = tfidf_vec.nonzero()[1]
    contributions = sorted(
        [(feature_names[i], coefficients[i] * tfidf_vec[0, i]) for i in present_indices],
        key=lambda x: x[1], reverse=True
    )
    top_fraud = [f"{w} ({v:.3f})" for w, v in contributions[:5] if v > 0]
    top_legit = [f"{w} ({v:.3f})" for w, v in contributions[-5:] if v < 0]

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Pushing toward FRAUD")
        for item in (top_fraud or ["none found"]):
            st.write(item)
    with col2:
        st.subheader("Pushing toward LEGITIMATE")
        for item in (top_legit or ["none found"]):
            st.write(item)
