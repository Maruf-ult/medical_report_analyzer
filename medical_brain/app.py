"""
MEDICAL BRAIN - App
====================
Run with: streamlit run app.py
"""

import os
import sys
import json
import tempfile
import subprocess
import streamlit as st
from pathlib import Path

# Windows UTF-8 Fix
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

os.environ["PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK"] = "True"

# ── Page Config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Medical Brain",
    page_icon="🧠",
    layout="wide"
)

# ── Load DB & LLM ─────────────────────────────────────────────────────────────
@st.cache_resource
def load_components():
    import chromadb
    from chromadb.utils import embedding_functions
    import ollama

    client = chromadb.PersistentClient(path="vector_db")
    embedder = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name="all-MiniLM-L6-v2"
    )
    collection = client.get_or_create_collection(
        name="medical_reports",
        embedding_function=embedder
    )
    return collection, ollama

# ── OCR via subprocess (avoids Streamlit threading crash) ─────────────────────
def run_ocr_subprocess(file_path: str) -> str:
    script = f"""
import os, json
os.environ['PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK'] = 'True'
from paddleocr import PaddleOCR
ocr = PaddleOCR(lang='en')
results = ocr.predict(r'''{file_path}''')
lines = []
for res in (results or []):
    if isinstance(res, dict) and 'rec_texts' in res:
        lines.extend([t for t in res['rec_texts'] if t])
print(json.dumps(lines))
"""
    result = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True, text=True, timeout=180
    )
    if result.returncode == 0 and result.stdout.strip():
        try:
            lines = json.loads(result.stdout.strip())
            return "\n".join(lines)
        except Exception:
            return ""
    return ""

# ── Extract Text from Uploaded File ──────────────────────────────────────────
def extract_text(uploaded_file) -> str:
    suffix = Path(uploaded_file.name).suffix.lower()

    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(uploaded_file.getvalue())
        tmp_path = tmp.name

    text = ""
    try:
        if suffix == ".txt":
            text = uploaded_file.getvalue().decode("utf-8", errors="replace")

        elif suffix == ".pdf":
            import pdfplumber
            with pdfplumber.open(tmp_path) as pdf:
                text = "\n\n".join(p.extract_text() or "" for p in pdf.pages)
            if len(text.strip()) < 50:
                st.info("Scanned PDF detected, running OCR...")
                text = run_ocr_subprocess(tmp_path)

        elif suffix in [".jpg", ".jpeg", ".png"]:
            text = run_ocr_subprocess(tmp_path)

    except Exception as e:
        st.error(f"Error extracting text: {e}")
    finally:
        try:
            os.unlink(tmp_path)
        except Exception:
            pass

    return text

# ── Search Similar Cases ──────────────────────────────────────────────────────
def search_similar(collection, query: str, n: int = 3) -> list:
    try:
        count = collection.count()
        if count == 0:
            return []
        results = collection.query(
            query_texts=[query],
            n_results=min(n, count)
        )
        return results["documents"][0] if results["documents"] else []
    except Exception:
        return []

# ── AI: Analyze Full Report ───────────────────────────────────────────────────
def analyze_report(ollama, report_text: str, similar_cases: list) -> str:
    prompt = f"""You are a medical report analyzer. Analyze this report concisely.

REPORT:
{report_text[:1500]}

Respond in this exact format:
## Lab Results
- [Test Name]: [Value] - [HIGH/LOW/NORMAL]

## What This Means
[2-3 sentences explaining abnormal results in simple English]

## Summary
[1-2 sentences overall summary]

*Always consult your doctor for medical advice.*"""

    response = ollama.chat(
        model="mistral",
        messages=[{"role": "user", "content": prompt}]
    )
    return response["message"]["content"]

# ── AI: Answer Question ───────────────────────────────────────────────────────
def answer_question(ollama, question: str, report_text: str, similar_cases: list) -> str:
    prompt = f"""Medical assistant. Answer briefly based on this report.

REPORT: {report_text[:800]}

QUESTION: {question}

Give a short, clear answer in 2-3 sentences. Recommend consulting a doctor."""

    response = ollama.chat(
        model="mistral",
        messages=[{"role": "user", "content": prompt}]
    )
    return response["message"]["content"]

# ── Main UI ───────────────────────────────────────────────────────────────────
def main():
    st.title("🧠 Medical Brain")
    st.caption("Upload your medical report for instant AI-powered analysis")

    # Load components
    with st.spinner("Loading AI components..."):
        collection, ollama = load_components()

    # Sidebar
    st.sidebar.title("System Status")
    st.sidebar.success(f"✅ Database: {collection.count()} chunks")
    st.sidebar.info("🤖 Model: Mistral 7B (Local)")
    st.sidebar.warning("⚠️ For informational purposes only. Always consult a doctor.")

    # ── Upload ────────────────────────────────────────────────────────────────
    st.subheader("Step 1: Upload Your Report")
    uploaded = st.file_uploader(
        "Drop your medical report here",
        type=["pdf", "jpg", "jpeg", "png", "txt"],
        help="Supported formats: PDF, JPG, PNG, TXT"
    )

    if uploaded:
        st.success(f"✅ Uploaded: {uploaded.name}")

        with st.spinner("Reading your report..."):
            report_text = extract_text(uploaded)

        if not report_text.strip():
            st.error("Could not extract text. Please try a clearer image or different file.")
            return

        with st.expander("📄 View extracted text"):
            st.text(report_text[:2000])
            if len(report_text) > 2000:
                st.caption(f"...and {len(report_text)-2000} more characters")

        # ── Analyze ───────────────────────────────────────────────────────────
        st.subheader("Step 2: AI Analysis")
        if st.button("🔍 Analyze Report", type="primary", use_container_width=True):

            with st.spinner("Searching similar cases in database..."):
                similar = search_similar(collection, report_text[:500])

            with st.spinner("AI is analyzing your report... (60-90 seconds)"):
                analysis = analyze_report(ollama, report_text, similar)

            st.markdown("### 📋 Analysis Result")
            st.markdown(analysis)

            if similar:
                with st.expander(f"📚 {len(similar)} similar cases used as reference"):
                    for i, case in enumerate(similar, 1):
                        st.markdown(f"**Case {i}:**")
                        st.text(case[:300])
                        st.divider()

            # Save to session state for chat
            st.session_state["report_text"] = report_text
            st.session_state["analyzed"] = True

        # ── Chat ──────────────────────────────────────────────────────────────
        if st.session_state.get("analyzed"):
            st.subheader("Step 3: Ask Questions About Your Report")
            st.caption("Ask anything — about your results, what they mean, or what to do next")

            if "chat_history" not in st.session_state:
                st.session_state.chat_history = []

            # Show chat history
            for msg in st.session_state.chat_history:
                with st.chat_message(msg["role"]):
                    st.markdown(msg["content"])

            # Chat input
            question = st.chat_input("e.g. Why is my hemoglobin low? What should I eat?")
            if question:
                st.session_state.chat_history.append({"role": "user", "content": question})
                with st.chat_message("user"):
                    st.markdown(question)

                with st.chat_message("assistant"):
                    with st.spinner("Thinking..."):
                        similar = search_similar(collection, question)
                        answer = answer_question(
                            ollama,
                            question,
                            st.session_state["report_text"],
                            similar
                        )
                    st.markdown(answer)
                    st.session_state.chat_history.append({
                        "role": "assistant",
                        "content": answer
                    })

if __name__ == "__main__":
    main()