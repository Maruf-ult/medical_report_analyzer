"""
MEDICAL BRAIN - Step 1: Data Preparation & OCR
================================================
Converts all reports (PDF/JPG/PNG/TXT) to clean text files.

HOW TO RUN:
    python step1_prepare_data.py
"""

import os
import re
import sys
import time
import logging
from pathlib import Path
from datetime import datetime

# Windows UTF-8 Fix
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
if hasattr(sys.stderr, 'reconfigure'):
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')

# Paths
RAW_DIR    = Path("raw_reports")
OUTPUT_DIR = Path("processed_text")
LOG_DIR    = Path("logs")
OUTPUT_DIR.mkdir(exist_ok=True)
LOG_DIR.mkdir(exist_ok=True)

# Logging
log_file = LOG_DIR / f"step1_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(log_file, encoding='utf-8'),
        logging.StreamHandler(sys.stdout),
    ]
)
log = logging.getLogger(__name__)

# Silence PaddleOCR connectivity check
os.environ["PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK"] = "True"

# De-identification patterns
DEIDENT_PATTERNS = [
    (r'\b[A-Z][a-z]+ [A-Z][a-z]+\b',           "[NAME]"),
    (r'\b(\+?\d[\d\s\-().]{7,}\d)\b',           "[PHONE]"),
    (r'\b[\w._%+-]+@[\w.-]+\.[A-Z|a-z]{2,}\b',  "[EMAIL]"),
    (r'\b\d{5}-\d{7}-\d\b',                     "[CNIC]"),
    (r'\bDOB[:\s]+\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b', "[DOB]"),
    (r'\b(MR#?|Patient ID|Pt\.? ID)[:\s]*\w+',  "[PATIENT_ID]"),
]

def deidentify(text):
    for pattern, replacement in DEIDENT_PATTERNS:
        text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)
    return text

def clean_text(text):
    text = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]', '', text)
    text = re.sub(r'\n{3,}', '\n\n', text)
    text = '\n'.join(line.rstrip() for line in text.splitlines())
    return text.strip()

# OCR Singleton - load model once, reuse for all files
_ocr = None

def get_ocr():
    global _ocr
    if _ocr is None:
        log.info("[INIT] Loading PaddleOCR models (first time only)...")
        from paddleocr import PaddleOCR
        _ocr = PaddleOCR(lang='en')
        log.info("[INIT] OCR engine ready.")
    return _ocr

def run_ocr(path):
    try:
        ocr = get_ocr()
        results = ocr.predict(str(path))
        lines = []
        for res in (results or []):
            # PaddleOCR 3.x returns a list of dicts with rec_texts key
            if isinstance(res, dict) and 'rec_texts' in res:
                lines.extend([t for t in res['rec_texts'] if t])
            elif hasattr(res, 'rec_texts'):
                lines.extend([t for t in res.rec_texts if t])
        text = "\n".join(lines)
        if not text.strip():
            log.warning(f"  OCR returned empty for: {path.name}")
        return text
    except Exception as e:
        log.error(f"OCR failed on {path.name}: {e}")
        return f"[OCR_FAILED: {e}]"

def process_txt(path):
    return path.read_text(encoding='utf-8', errors='replace')

def process_pdf(path):
    text = ""
    try:
        import pdfplumber
        with pdfplumber.open(path) as pdf:
            text = "\n\n".join(p.extract_text() or "" for p in pdf.pages)
    except Exception as e:
        log.warning(f"pdfplumber failed: {e}")
    if len(text.strip()) < 50:
        log.info(f"  Scanned PDF, switching to OCR: {path.name}")
        text = run_ocr(path)
    return text

def process_image(path):
    return run_ocr(path)

PROCESSORS = {
    '.txt':  process_txt,
    '.pdf':  process_pdf,
    '.jpg':  process_image,
    '.jpeg': process_image,
    '.png':  process_image,
}

def process_file(path):
    suffix = path.suffix.lower()
    if suffix not in PROCESSORS:
        return False

    output_path = OUTPUT_DIR / (path.stem + ".txt")
    if output_path.exists():
        log.info(f"  [SKIP] Already done: {path.name}")
        return True

    log.info(f"  [PROCESSING] {path.name}")
    try:
        raw_text = PROCESSORS[suffix](path)
    except Exception as e:
        log.error(f"  Failed: {path.name}: {e}")
        return False

    if not raw_text.strip():
        log.warning(f"  [WARN] No text extracted: {path.name}")
        return False

    cleaned = clean_text(raw_text)
    safe = deidentify(cleaned)
    header = (
        f"=== SOURCE: {path.name} ===\n"
        f"=== PROCESSED: {datetime.now().strftime('%Y-%m-%d %H:%M')} ===\n"
        f"=== FORMAT: {suffix.upper()} ===\n\n"
    )
    output_path.write_text(header + safe, encoding='utf-8')
    log.info(f"  [SAVED] {output_path.name} ({len(safe)} chars)")
    return True

def main():
    log.info("=" * 60)
    log.info("  MEDICAL BRAIN - Step 1: Data Preparation")
    log.info("=" * 60)

    all_files = [
        f for f in RAW_DIR.rglob("*")
        if f.is_file() and f.suffix.lower() in PROCESSORS
    ]

    if not all_files:
        log.warning(f"No supported files found in '{RAW_DIR}/'")
        return

    total, success, failed = len(all_files), 0, []
    log.info(f"Found {total} files to process.\n")
    start = time.time()

    for i, fp in enumerate(all_files, 1):
        log.info(f"[{i}/{total}]")
        if process_file(fp):
            success += 1
        else:
            failed.append(fp.name)

    elapsed = time.time() - start
    log.info("\n" + "=" * 60)
    log.info(f"  [OK] {success}/{total} files processed")
    log.info(f"  [TIME] {elapsed/60:.1f} minutes")
    log.info(f"  [OUT] {OUTPUT_DIR.resolve()}")
    if failed:
        log.info(f"  [FAILED] {len(failed)} files:")
        for f in failed[:10]:
            log.info(f"    - {f}")
    log.info("=" * 60)
    log.info("\n[DONE] Step 1 complete! Next: step2_build_vectordb.py\n")

if __name__ == "__main__":
    main()