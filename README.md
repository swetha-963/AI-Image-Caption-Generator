# CaptionAI

A local-first image captioning studio with two model modes:

- **Detailed · SmolVLM-256M** follows a prompt to create grounded paragraph-style descriptions. Its model weights (about 0.5 GB) download on first use and are cached locally.
- **Quick · BLIP** uses the bundled, smaller model for fast, short captions and works without a model download.

## Features

- Short, detailed, or rich-paragraph caption styles.
- Adjustable generation limit and quality-versus-speed setting.
- Single-image preview, editable captions, and TXT downloads.
- Batch captioning for up to 20 images with CSV export.
- Session-only caption history with CSV export and clear-history action.
- Cached repeat generations, CPU/CUDA device selection, and sequential batch processing.
- PNG, JPG, JPEG, and WEBP image support.

## Setup

Use Python 3.9 or newer. The bundled BLIP model should be present in `models/blip-image-captioning-base`.

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python -m streamlit run app.py
```

Select **Detailed · SmolVLM-256M** for paragraph captions. Its first use requires an internet connection to download a pinned Hugging Face model revision; future runs use the local cache and do not send images to a captioning API. Select **Quick · BLIP** to use the bundled model without downloading anything. SmolVLM loads through the standard Transformers implementation without enabling `trust_remote_code`.

Transformers is pinned to 4.51.3 for compatibility with the detailed model.

## Privacy and performance

Images are processed locally by whichever model you choose. Captions are cached in memory for up to one hour. Batch images are processed sequentially to limit memory use. History is session-only and can be exported or cleared in the app.
