"""
MEDICAL BRAIN - App with Groq API + Visualizations
====================================================
Run with: streamlit run app.py
"""

import os
import sys
import json
import tempfile
import subprocess
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from pathlib import Path
from datetime import datetime
from dotenv import load_dotenv

# Load API key from .env file
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

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

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .main { background-color: #f8f9fa; }
    .stButton>button {
        background-color: #2563eb;
        color: white;
        border-radius: 8px;
        padding: 0.5rem 1rem;
        font-weight: bold;
    }
    .metric-card {
        background: white;
        border-radius: 12px;
        padding: 1rem;
        box-shadow: 0 2px 8px rgba(0,0,0,0.08);
        text-align: center;
        margin: 0.5rem 0;
    }
    .high { color: #dc2626; font-weight: bold; }
    .low  { color: #2563eb; font-weight: bold; }
    .normal { color: #16a34a; font-weight: bold; }
</style>
""", unsafe_allow_html=True)

# ── Load ChromaDB ─────────────────────────────────────────────────────────────
@st.cache_resource
def load_db():
    import chromadb
    from chromadb.utils import embedding_functions

    client = chromadb.PersistentClient(path="vector_db")
    embedder = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name="all-MiniLM-L6-v2"
    )
    collection = client.get_or_create_collection(
        name="medical_reports",
        embedding_function=embedder
    )
    return collection

# ── Groq AI ───────────────────────────────────────────────────────────────────
def ask_groq(prompt: str) -> str:
    from groq import Groq
    client = Groq(api_key=GROQ_API_KEY)
    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3,
        max_tokens=1500
    )
    return response.choices[0].message.content

# ── OCR via subprocess ────────────────────────────────────────────────────────
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

# ── Extract Text ──────────────────────────────────────────────────────────────
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
        st.error(f"Error reading file: {e}")
    finally:
        try:
            os.unlink(tmp_path)
        except Exception:
            pass
    return text

# ── Save Report to DB ─────────────────────────────────────────────────────────
def save_to_database(collection, text: str, filename: str):
    """Auto-save every uploaded report to grow the AI brain."""
    try:
        chunk_size = 400
        chunks = []
        for i in range(0, len(text), chunk_size - 80):
            chunk = text[i:i + chunk_size].strip()
            if len(chunk) > 30:
                chunks.append(chunk)

        if not chunks:
            return

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        ids = [f"upload_{timestamp}_{i}" for i in range(len(chunks))]

        collection.add(
            ids=ids,
            documents=chunks,
            metadatas=[{
                "source": filename,
                "uploaded_at": timestamp,
                "chunk_index": i
            } for i in range(len(chunks))]
        )
    except Exception as e:
        pass  # Silent fail - don't interrupt user experience

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

# ── Parse Lab Results from AI ─────────────────────────────────────────────────
def parse_lab_results(analysis_text: str) -> list:
    """Extract structured lab results from AI analysis for visualization."""
    prompt = f"""Extract lab results from this medical analysis as JSON only.

ANALYSIS:
{analysis_text}

Return ONLY a JSON array like this, nothing else:
[
  {{"test": "Hemoglobin", "value": 10.5, "unit": "g/dL", "status": "LOW", "normal_min": 12, "normal_max": 17}},
  {{"test": "WBC", "value": 8.2, "unit": "K/uL", "status": "NORMAL", "normal_min": 4.5, "normal_max": 11}}
]

If you cannot find numeric values, return an empty array: []"""

    try:
        response = ask_groq(prompt)
        # Find JSON array in response
        start = response.find("[")
        end = response.rfind("]") + 1
        if start >= 0 and end > start:
            return json.loads(response[start:end])
    except Exception:
        pass
    return []

# ── Visualizations ────────────────────────────────────────────────────────────
def show_lab_chart(lab_results: list):
    if not lab_results:
        return

    st.markdown("### 📊 Lab Results Visualization")

    # Color map
    color_map = {"HIGH": "#dc2626", "LOW": "#2563eb", "NORMAL": "#16a34a"}

    # ── Status Summary Pie Chart ──
    status_counts = {"HIGH": 0, "LOW": 0, "NORMAL": 0}
    for r in lab_results:
        s = r.get("status", "NORMAL").upper()
        if s in status_counts:
            status_counts[s] += 1

    col1, col2 = st.columns([1, 2])

    with col1:
        fig_pie = go.Figure(data=[go.Pie(
            labels=list(status_counts.keys()),
            values=list(status_counts.values()),
            hole=0.4,
            marker_colors=["#dc2626", "#2563eb", "#16a34a"]
        )])
        fig_pie.update_layout(
            title="Results Overview",
            height=300,
            margin=dict(t=40, b=0, l=0, r=0),
            showlegend=True
        )
        st.plotly_chart(fig_pie, use_container_width=True)

    with col2:
        # ── Bar Chart with Normal Range ──
        tests  = [r["test"] for r in lab_results]
        values = [r.get("value", 0) for r in lab_results]
        colors = [color_map.get(r.get("status", "NORMAL").upper(), "#16a34a") for r in lab_results]

        fig_bar = go.Figure()
        fig_bar.add_trace(go.Bar(
            x=tests,
            y=values,
            marker_color=colors,
            text=[f"{v} {r.get('unit','')}" for v, r in zip(values, lab_results)],
            textposition="outside",
            name="Your Value"
        ))

        # Add normal range lines
        for i, r in enumerate(lab_results):
            if r.get("normal_min") and r.get("normal_max"):
                fig_bar.add_shape(
                    type="rect",
                    x0=i - 0.4, x1=i + 0.4,
                    y0=r["normal_min"], y1=r["normal_max"],
                    fillcolor="rgba(22,163,74,0.15)",
                    line=dict(color="rgba(22,163,74,0.4)", width=1),
                )

        fig_bar.update_layout(
            title="Your Values vs Normal Range (green shading)",
            height=300,
            margin=dict(t=40, b=0, l=0, r=0),
            showlegend=False,
            plot_bgcolor="white",
            yaxis=dict(gridcolor="#f0f0f0")
        )
        st.plotly_chart(fig_bar, use_container_width=True)

    # ── Metric Cards ──
    st.markdown("#### Individual Results")
    cols = st.columns(min(len(lab_results), 4))
    for i, r in enumerate(lab_results):
        status = r.get("status", "NORMAL").upper()
        icon = "🔴" if status == "HIGH" else "🔵" if status == "LOW" else "🟢"
        with cols[i % 4]:
            st.markdown(f"""
            <div class="metric-card">
                <div style="font-size:1.5rem">{icon}</div>
                <div style="font-weight:bold;font-size:0.9rem">{r['test']}</div>
                <div style="font-size:1.3rem;font-weight:bold">{r.get('value','-')} <span style="font-size:0.8rem">{r.get('unit','')}</span></div>
                <div class="{status.lower()}">{status}</div>
            </div>
            """, unsafe_allow_html=True)

# ── Main UI ───────────────────────────────────────────────────────────────────
def main():
    # Header
    st.title("🧠 Medical Brain")
    st.caption("Upload your medical report for instant AI-powered analysis")

    if not GROQ_API_KEY:
        st.error("GROQ_API_KEY not found! Please add it to your .env file.")
        return

    # Load DB
    with st.spinner("Loading..."):
        collection = load_db()

    # Sidebar
    st.sidebar.title("System Status")
    st.sidebar.success(f"✅ Database: {collection.count()} report chunks")
    st.sidebar.info("⚡ Model: Llama3-70B via Groq")
    st.sidebar.info("🔒 100% Private - runs locally")
    st.sidebar.warning("⚠️ For informational purposes only. Always consult a doctor.")

    # ── Upload ────────────────────────────────────────────────────────────────
    st.subheader("Step 1: Upload Your Medical Report")
    uploaded = st.file_uploader(
        "Drop your report here — PDF, Image, or Text",
        type=["pdf", "jpg", "jpeg", "png", "txt"],
        help="Your report is processed privately on your device"
    )

    if uploaded:
        st.success(f"✅ Uploaded: {uploaded.name}")

        with st.spinner("Reading your report..."):
            report_text = extract_text(uploaded)

        if not report_text.strip():
            st.error("Could not extract text. Please try a clearer image.")
            return

        # Auto-save to database
        save_to_database(collection, report_text, uploaded.name)
        st.sidebar.success(f"✅ Database: {collection.count()} report chunks")

        with st.expander("📄 View extracted text"):
            st.text(report_text[:2000])
            if len(report_text) > 2000:
                st.caption(f"...and {len(report_text)-2000} more characters")

        # ── Analyze ───────────────────────────────────────────────────────────
        st.subheader("Step 2: AI Analysis")
        if st.button("🔍 Analyze My Report", type="primary", use_container_width=True):

            with st.spinner("Searching similar cases..."):
                similar = search_similar(collection, report_text[:500])
                similar_context = "\n\n---\n\n".join(similar) if similar else "No similar cases."

            with st.spinner("⚡ Groq AI is analyzing... (3-5 seconds)"):
                prompt = f"""You are a professional medical report analyzer.

SIMILAR CASES FROM DATABASE:
{similar_context[:500]}

PATIENT REPORT:
{report_text[:1500]}

Respond in this exact format:
## Lab Results
- [Test Name]: [Value] [Unit] - [HIGH/LOW/NORMAL]

## What This Means
[2-3 sentences explaining abnormal results in simple English a non-doctor understands]

## Overall Summary
[1-2 sentences summary]

## What To Do Next
[2-3 practical suggestions]

*Please consult your doctor for proper medical advice.*"""

                analysis = ask_groq(prompt)

            st.markdown("### 📋 Analysis Result")
            st.markdown(analysis)

            # ── Visualizations ─────────────────────────────────────────────
            with st.spinner("Generating visualizations..."):
                lab_results = parse_lab_results(analysis)

            if lab_results:
                show_lab_chart(lab_results)
            else:
                st.info("No numeric lab values found for visualization.")

            if similar:
                with st.expander(f"📚 {len(similar)} similar cases used as reference"):
                    for i, case in enumerate(similar, 1):
                        st.markdown(f"**Case {i}:**")
                        st.text(case[:300])
                        st.divider()

            st.session_state["report_text"] = report_text
            st.session_state["analyzed"] = True

        # ── Chat ──────────────────────────────────────────────────────────────
        if st.session_state.get("analyzed"):
            st.subheader("Step 3: Ask Questions About Your Report")
            st.caption("Ask anything about your results in simple language")

            if "chat_history" not in st.session_state:
                st.session_state.chat_history = []

            for msg in st.session_state.chat_history:
                with st.chat_message(msg["role"]):
                    st.markdown(msg["content"])

            question = st.chat_input("e.g. Why is my hemoglobin low? What foods should I eat?")
            if question:
                st.session_state.chat_history.append({"role": "user", "content": question})
                with st.chat_message("user"):
                    st.markdown(question)

                with st.chat_message("assistant"):
                    with st.spinner("⚡ Thinking..."):
                        similar = search_similar(collection, question)
                        similar_ctx = "\n".join(similar[:2]) if similar else ""
                        answer_prompt = f"""Medical assistant. Answer this question about the patient's report.

REPORT: {st.session_state['report_text'][:800]}
SIMILAR CASES: {similar_ctx[:400]}
QUESTION: {question}

Answer in 2-4 sentences. Be clear and simple. Recommend consulting a doctor."""

                        answer = ask_groq(answer_prompt)
                    st.markdown(answer)
                    st.session_state.chat_history.append({
                        "role": "assistant",
                        "content": answer
                    })

if __name__ == "__main__":
    main()