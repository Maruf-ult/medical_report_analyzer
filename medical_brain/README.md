# 🧠 Medical Brain — Local AI Medical Report Analyzer
## Step-by-Step Setup Guide (Free, Runs 100% Offline)

---

## 📁 Project Folder Structure

```
medical_brain/
│
├── raw_reports/          ← 📂 DROP ALL 2,000 REPORTS HERE
│   ├── report_001.pdf
│   ├── scan_002.jpg
│   └── lab_003.txt
│
├── processed_text/       ← Auto-created: clean .txt files go here
├── logs/                 ← Auto-created: processing logs
│
├── step1_prepare_data.py ← ✅ YOU ARE HERE
├── requirements.txt      ← Python packages list
└── README.md             ← This file
```

---

## 🚀 First-Time Setup (Do This Once)

### 1. Open Terminal in VS Code
Press `` Ctrl+` `` to open the terminal inside VS Code.

### 2. Navigate to the project folder
```bash
cd path/to/medical_brain
```

### 3. Create a virtual environment (keeps your project isolated)
```bash
python -m venv venv
```

### 4. Activate the virtual environment
**Windows:**
```bash
venv\Scripts\activate
```
**Mac/Linux:**
```bash
source venv/bin/activate
```
You should see `(venv)` appear at the start of your terminal line.

### 5. Install all dependencies
```bash
pip install -r requirements.txt
```
⏱ This will take 5–15 minutes. PaddleOCR is large (~1.5GB).

---

## 📂 Step 1: Prepare Your Data

### A. Copy your reports
Put ALL 2,000 reports into the `raw_reports/` folder.
- Supported: `.pdf`, `.jpg`, `.jpeg`, `.png`, `.txt`
- Subfolders are fine — the script scans recursively

### B. Run the preparation script
```bash
python step1_prepare_data.py
```

### C. What it does
| Task | Detail |
|------|--------|
| **PDF → Text** | Extracts text from digital PDFs instantly |
| **Scanned PDF → OCR** | If a PDF has no text layer, runs OCR automatically |
| **JPG/PNG → OCR** | Runs PaddleOCR (free, local, no API key) |
| **De-identification** | Strips names, phones, emails, CNICs |
| **Cleaning** | Removes junk characters, normalizes whitespace |
| **Resume-safe** | If interrupted, re-running skips already done files |

### D. Expected output
```
processed_text/
├── report_001.txt    ← clean, de-identified text
├── scan_002.txt
└── lab_003.txt
```

---

## ❓ Troubleshooting

### "No files found in raw_reports/"
- Make sure your reports are inside the `raw_reports/` folder
- Check the file extension is `.pdf`, `.jpg`, `.png`, or `.txt`

### "PaddleOCR not installed"
```bash
pip install paddlepaddle paddleocr
```
First run will download ~1.5GB of OCR models. Give it time.

### "OCR output looks garbled"
- Make sure images are at least 150 DPI
- For very low-quality scans, results may be imperfect — this is normal

### Script crashed halfway through
No problem! Re-run the script. It automatically skips already-processed files.

---

## ✅ After Step 1 is Complete

Your `processed_text/` folder should contain ~2,000 `.txt` files.

**Next step:** `step2_build_vectordb.py` — this converts all those text files
into a searchable "memory" that the AI can query instantly.

---

## 📊 Estimated Processing Times

| Report Type | Time per File |
|-------------|---------------|
| Plain text (.txt) | < 0.1 sec |
| Digital PDF | 0.5–2 sec |
| Scanned PDF / Image | 3–10 sec |

**For 2,000 image-based reports:** ~2–5 hours total.
**For 2,000 digital PDFs:** ~15–30 minutes total.
**Tip:** Run overnight so it doesn't block your work.
