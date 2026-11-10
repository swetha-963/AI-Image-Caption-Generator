import csv
import hashlib
import io
from datetime import datetime
from html import escape
from pathlib import Path

import streamlit as st
import torch
from PIL import Image, UnidentifiedImageError
from huggingface_hub.errors import HfHubHTTPError
from transformers import (
    AutoModelForVision2Seq,
    AutoProcessor,
    BlipForConditionalGeneration,
    BlipProcessor,
)


MODEL_DIR = Path(__file__).parent / "models" / "blip-image-captioning-base"
DETAILED_MODEL_ID = "HuggingFaceTB/SmolVLM-256M-Instruct"
DETAILED_MODEL_REVISION = "7e3e67edbbed1bf9888184d9df282b700a323964"
CAPTION_PROMPTS = {
    "Short caption": (
        "Describe this image in one clear sentence. Only mention details that are "
        "directly visible; do not guess."
    ),
    "Detailed paragraph": (
        "Write a detailed caption in 3 to 4 complete sentences. Describe the main "
        "subject, visible appearance, colors, arrangement, and background. Include "
        "readable text only when clear. Do not infer hidden context or guess."
    ),
    "Rich paragraph": (
        "Write a rich, natural 5 to 7 sentence description of this image. Cover the "
        "main subject and its visible features, colors, position and composition, "
        "surrounding objects, and background. Mention text only when it is readable. "
        "Keep every detail grounded in what can actually be seen; do not speculate, "
        "repeat yourself, or add advice."
    ),
}
SUPPORTED_FORMATS = ["png", "jpg", "jpeg", "webp"]
MAX_BATCH_SIZE = 20
CSV_FIELDS = ["image", "caption", "generated_at", "mode"]

st.set_page_config(
    page_title="CaptionAI | Image Caption Generator",
    page_icon="✦",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown(
    """
    <style>
    :root {
        --ink: #f5f4f8;
        --muted: #9a98a8;
        --line: rgba(255, 255, 255, .09);
        --panel: rgba(255, 255, 255, .045);
        --accent: #a78bfa;
        --mint: #72e0c1;
    }

    .stApp {
        color: var(--ink);
        background:
            radial-gradient(ellipse at 12% 0%, rgba(108, 76, 195, .22), transparent 34rem),
            radial-gradient(ellipse at 92% 18%, rgba(27, 145, 151, .12), transparent 32rem),
            #0d0e13;
        font-family: Inter, 'Segoe UI', sans-serif;
    }

    [data-testid="stHeader"] { background: transparent; }
    [data-testid="stToolbar"], #MainMenu, footer { visibility: hidden; }
    .block-container { max-width: 1160px; padding: 1.6rem 2rem 3rem; }

    .topbar {
        display: flex; align-items: center; justify-content: space-between;
        padding: 0 0 3.2rem;
    }
    .brand {
        color: var(--ink); font: 800 1.2rem Inter, 'Segoe UI', sans-serif;
        letter-spacing: -.7px;
    }
    .brand-mark {
        display: inline-grid; place-items: center; width: 30px; height: 30px;
        margin-right: 9px; color: #171321; border-radius: 10px;
        background: linear-gradient(135deg, #c4b5fd, #79dfd0);
    }
    .local-pill {
        display: inline-flex; gap: 8px; align-items: center;
        padding: 8px 13px; border: 1px solid var(--line); border-radius: 30px;
        color: #c6c4ce; background: rgba(255,255,255,.035); font-size: .75rem;
    }
    .local-dot {
        width: 7px; height: 7px; border-radius: 50%; background: var(--mint);
        box-shadow: 0 0 12px rgba(114,224,193,.8);
    }

    .hero { max-width: 760px; margin: 0 auto 2.8rem; text-align: center; }
    .eyebrow {
        display: inline-block; margin-bottom: 1.1rem; padding: 7px 12px;
        color: #c9b9ff; border: 1px solid rgba(167,139,250,.2);
        border-radius: 30px; background: rgba(167,139,250,.08);
        font-size: .68rem; font-weight: 700; letter-spacing: 1.4px;
    }
    .hero h1 {
        margin: 0; color: var(--ink); font: 800 clamp(2.5rem, 6vw, 4.5rem)/1.07 Inter, 'Segoe UI', sans-serif;
        letter-spacing: -3px;
    }
    .hero h1 span {
        color: #b8a2ff;
        background: linear-gradient(90deg, #c4b5fd, #8de1d4);
        -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    }
    .hero p {
        max-width: 550px; margin: 1.15rem auto 0; color: var(--muted);
        font-size: 1rem; line-height: 1.75;
    }

    .section-heading {
        margin-bottom: 1.1rem; color: #f4f2f8;
        font: 700 1.05rem Inter, 'Segoe UI', sans-serif;
    }
    .section-subtitle { margin-top: -.7rem; margin-bottom: 1.2rem; color: var(--muted); font-size: .82rem; }
    [data-testid="stVerticalBlockBorderWrapper"] {
        min-height: 430px; padding: 1.5rem;
        border: 1px solid var(--line); border-radius: 22px;
        background: linear-gradient(145deg, rgba(255,255,255,.055), rgba(255,255,255,.025));
        box-shadow: 0 24px 70px rgba(0,0,0,.2), inset 0 1px rgba(255,255,255,.035);
    }
    [data-testid="stFileUploader"] {
        padding: .35rem; border: 1px dashed rgba(167,139,250,.35);
        border-radius: 15px; background: rgba(167,139,250,.035);
    }
    [data-testid="stFileUploader"]:hover { border-color: rgba(114,224,193,.55); }
    [data-testid="stFileUploaderDropzone"] { background: transparent; }
    [data-testid="stFileUploaderDropzone"] small { color: #858391; }
    [data-testid="stImage"] img { border-radius: 14px; }

    .empty-state {
        min-height: 245px; display: flex; flex-direction: column;
        align-items: center; justify-content: center; text-align: center;
        border: 1px solid rgba(255,255,255,.055); border-radius: 16px;
        background: rgba(0,0,0,.12);
    }
    .empty-icon {
        display: grid; place-items: center; width: 54px; height: 54px;
        margin-bottom: 14px; color: #c2b0ff; border: 1px solid rgba(167,139,250,.18);
        border-radius: 17px; background: rgba(167,139,250,.09); font-size: 1.45rem;
    }
    .empty-title { color: #e4e1eb; font: 600 .95rem 'Manrope', sans-serif; }
    .empty-copy { margin-top: 6px; color: #858391; font-size: .78rem; }
    .file-meta { margin: .7rem 0 0; color: #9f9daa; font-size: .76rem; }

    .stButton > button {
        width: 100%; min-height: 48px; margin-top: .8rem;
        color: #171321; border: 0; border-radius: 12px;
        background: linear-gradient(100deg, #c4b5fd, #8de1d4);
        font: 700 .9rem Inter, 'Segoe UI', sans-serif;
        box-shadow: 0 8px 24px rgba(146,121,220,.2);
        transition: transform .18s ease, box-shadow .18s ease;
    }
    .stButton > button:hover {
        color: #171321; border: 0; transform: translateY(-2px);
        box-shadow: 0 12px 30px rgba(146,121,220,.3);
    }
    .stButton > button:disabled { color: #777382; background: #292832; box-shadow: none; }
    .caption-label {
        margin: 1.1rem 0 .6rem; color: #aaa5b7;
        font-size: .68rem; font-weight: 700; letter-spacing: 1.1px;
        text-transform: uppercase;
    }
    .caption-result {
        padding: 1rem 1.1rem; color: #eeebf5; border: 1px solid rgba(167,139,250,.2);
        border-radius: 14px; background: linear-gradient(120deg, rgba(167,139,250,.10), rgba(114,224,193,.055));
        font: 500 1.04rem/1.65 Inter, 'Segoe UI', sans-serif;
    }
    .stTextArea textarea {
        color: #dedbe6; border-color: var(--line); border-radius: 11px;
        background: rgba(0,0,0,.15);
    }
    .stTextArea label { color: #aaa7b3; font-size: .76rem; }
    [data-testid="stTabs"] button {
        color: #a7a4b2; font-weight: 600;
    }
    [data-testid="stTabs"] button[aria-selected="true"] { color: #d3c6ff; }
    [data-testid="stProgressBar"] > div > div { background: #a78bfa; }
    [data-testid="stDataFrame"] { border: 1px solid var(--line); border-radius: 12px; }
    [data-testid="stDownloadButton"] button {
        width: 100%; color: #e7e2f1; border: 1px solid var(--line);
        border-radius: 10px; background: rgba(255,255,255,.04);
    }
    .feature-row { display: flex; gap: 11px; margin-top: 1.1rem; }
    .feature {
        flex: 1; padding: 13px 15px; border: 1px solid var(--line);
        border-radius: 13px; background: rgba(255,255,255,.025);
    }
    .feature-title { color: #e0dde7; font-size: .78rem; font-weight: 700; }
    .feature-copy { margin-top: 3px; color: #858391; font-size: .69rem; }
    .footer {
        margin-top: 2.5rem; padding-top: 1.1rem; border-top: 1px solid var(--line);
        color: #777581; text-align: center; font-size: .72rem;
    }
    @media (max-width: 700px) {
        .block-container { padding: 1.2rem 1rem 2rem; }
        .topbar { padding-bottom: 2.2rem; }
        .hero { margin-bottom: 2rem; }
        .hero h1 { letter-spacing: -1.7px; }
        [data-testid="stVerticalBlockBorderWrapper"] { min-height: unset; padding: 1.15rem; }
        .feature-row { flex-direction: column; }
    }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource
def load_blip_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    processor = BlipProcessor.from_pretrained(
        str(MODEL_DIR),
        local_files_only=True,
    )
    model = BlipForConditionalGeneration.from_pretrained(
        str(MODEL_DIR),
        local_files_only=True,
    )
    model.to(device)
    model.eval()
    return processor, model, device


@st.cache_resource
def load_detailed_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float16 if device.type == "cuda" else torch.float32
    processor = AutoProcessor.from_pretrained(
        DETAILED_MODEL_ID,
        revision=DETAILED_MODEL_REVISION,
    )
    model = AutoModelForVision2Seq.from_pretrained(
        DETAILED_MODEL_ID,
        revision=DETAILED_MODEL_REVISION,
        torch_dtype=dtype,
        use_safetensors=True,
    )
    model.to(device)
    model.eval()
    return processor, model, device, dtype


@st.cache_data(show_spinner=False, ttl=3600, max_entries=64)
def generate_caption(
    image_data: bytes,
    caption_model: str,
    task: str,
    max_new_tokens: int,
    num_beams: int,
    min_new_tokens: int,
) -> str:
    with Image.open(io.BytesIO(image_data)) as source:
        image = source.convert("RGB")

    if caption_model == "SmolVLM":
        processor, model, device, dtype = load_detailed_model()
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": task},
                ],
            }
        ]
        prompt = processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
        )
        inputs = processor(
            text=prompt,
            images=[image],
            return_tensors="pt",
        ).to(device)
        if device.type == "cuda":
            inputs["pixel_values"] = inputs["pixel_values"].to(dtype=dtype)
        with torch.inference_mode():
            generated_ids = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                num_beams=num_beams,
                min_new_tokens=min_new_tokens,
            )
        caption = processor.batch_decode(
            generated_ids,
            skip_special_tokens=True,
        )[0]
        if "Assistant:" in caption:
            caption = caption.rsplit("Assistant:", maxsplit=1)[-1]
        if not caption.strip():
            raise RuntimeError("The detailed model returned no caption.")
        return caption.strip()

    if caption_model != "BLIP":
        raise ValueError(f"Unsupported caption model: {caption_model}")

    processor, model, device = load_blip_model()
    inputs = processor(images=image, return_tensors="pt").to(device)
    with torch.inference_mode():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            num_beams=num_beams,
        )
    return processor.decode(output_ids[0], skip_special_tokens=True).strip()


def image_digest(image_data: bytes) -> str:
    return hashlib.sha256(image_data).hexdigest()


def add_to_history(image_name: str, caption: str, mode: str) -> None:
    history = st.session_state.setdefault("history", [])
    history.insert(
        0,
        {
            "image": image_name,
            "caption": caption,
            "generated_at": datetime.now().astimezone().strftime("%Y-%m-%d %H:%M"),
            "mode": mode,
        },
    )
    del history[100:]


def csv_content(rows: list[dict[str, str]]) -> str:
    output = io.StringIO(newline="")
    writer = csv.DictWriter(output, fieldnames=CSV_FIELDS, extrasaction="ignore")
    writer.writeheader()
    writer.writerows(rows)
    return output.getvalue()


st.markdown(
    """
    <div class="topbar">
        <div class="brand"><span class="brand-mark">✦</span>CaptionAI</div>
        <div class="local-pill"><span class="local-dot"></span> PRIVATE · RUNS LOCALLY</div>
    </div>
    <div class="hero">
        <div class="eyebrow">YOUR PRIVATE IMAGE CAPTION STUDIO</div>
        <h1>Give your images<br><span>a voice.</span></h1>
        <p>Write natural descriptions, process a whole collection, and keep your results organized — all on your device.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

caption_engine_label = st.radio(
    "Caption engine",
    ["Detailed · SmolVLM-256M", "Quick · BLIP"],
    horizontal=True,
    help="SmolVLM is instruction-tuned for detailed paragraphs. BLIP is the smaller, faster model.",
)
caption_model = "SmolVLM" if caption_engine_label.startswith("Detailed") else "BLIP"

with st.expander("Generation settings", expanded=True):
    settings_left, settings_right = st.columns(2)
    with settings_left:
        detail_level = st.selectbox(
            "Description style",
            list(CAPTION_PROMPTS),
            index=2,
            disabled=caption_model == "BLIP",
            help="Rich paragraph asks the instruction-tuned model for a longer, grounded description.",
        )
        max_new_tokens = st.slider(
            "Maximum generation tokens",
            min_value=32,
            max_value=256,
            value=192,
            step=16,
            help="This is an output ceiling, not a word count. Choose Rich paragraph for the longest model-supported description.",
        )
    with settings_right:
        num_beams = st.slider(
            "Quality vs. speed",
            min_value=1,
            max_value=5,
            value=2,
            help="More beams can improve wording but use more processing time.",
        )

caption_task = CAPTION_PROMPTS[detail_level] if caption_model == "SmolVLM" else ""
minimum_tokens_by_style = {
    "Short caption": 0,
    "Detailed paragraph": 40,
    "Rich paragraph": 64,
}
min_new_tokens = minimum_tokens_by_style[detail_level] if caption_model == "SmolVLM" else 0
if caption_model == "SmolVLM":
    st.info(
        "Detailed mode uses SmolVLM-256M. Its model files (about 0.5 GB) download "
        "once on first use; after that, captions run locally. Use Quick · BLIP for "
        "short captions without the extra model download."
    )

st.caption(
    f"Model: {caption_model} · Device: "
    f"{'CUDA GPU' if torch.cuda.is_available() else 'CPU'} · "
    "Repeat captions are cached for up to one hour."
)

single_tab, batch_tab, history_tab = st.tabs(
    ["✦  Caption an image", "▦  Batch studio", "◷  History"]
)

with single_tab:
    upload_column, caption_column = st.columns(2, gap="large")
    uploaded_file = None
    image = None
    image_data = b""

    with upload_column:
        with st.container(border=True):
            st.markdown('<div class="section-heading">01 &nbsp; Upload an image</div>', unsafe_allow_html=True)
            st.markdown('<div class="section-subtitle">Choose a photo for the AI to describe.</div>', unsafe_allow_html=True)
            uploaded_file = st.file_uploader(
                "Drop an image here or browse files",
                type=SUPPORTED_FORMATS,
                label_visibility="collapsed",
                key="single_upload",
            )
            if uploaded_file is not None:
                image_data = uploaded_file.getvalue()
                try:
                    with Image.open(io.BytesIO(image_data)) as source:
                        image = source.convert("RGB")
                except (UnidentifiedImageError, OSError, Image.DecompressionBombError) as exc:
                    st.error(f"That file could not be opened as an image: {exc}")
                else:
                    st.image(image, width="stretch")
                    st.markdown(
                        f'<div class="file-meta">✓ &nbsp;{escape(uploaded_file.name)} &nbsp;·&nbsp; {image.width} × {image.height} px</div>',
                        unsafe_allow_html=True,
                    )
            else:
                st.markdown(
                    """
                    <div class="empty-state">
                        <div class="empty-icon">↥</div>
                        <div class="empty-title">Your next great caption starts here</div>
                        <div class="empty-copy">PNG, JPG, JPEG or WEBP</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

    with caption_column:
        with st.container(border=True):
            st.markdown('<div class="section-heading">02 &nbsp; Your caption</div>', unsafe_allow_html=True)
            st.markdown('<div class="section-subtitle">A thoughtful description, generated in seconds.</div>', unsafe_allow_html=True)
            image_id = image_digest(image_data) if image_data else None
            settings_id = image_digest(
                f"{caption_model}:{caption_task}:{max_new_tokens}:{num_beams}".encode()
            )
            generate = st.button(
                "✦  Generate caption",
                disabled=image is None,
                width="stretch",
                key="single_generate",
            )
            if generate and image_id:
                try:
                    spinner_text = (
                        "Writing a detailed paragraph..."
                        if caption_model == "SmolVLM"
                        else "Looking closely at your image..."
                    )
                    with st.spinner(spinner_text):
                        caption = generate_caption(
                            image_data,
                            caption_model,
                            caption_task,
                            max_new_tokens,
                            num_beams,
                            min_new_tokens,
                        )
                    if not caption:
                        st.warning("The model returned an empty caption. Try another image or caption style.")
                    else:
                        st.session_state["single_result"] = {
                            "image_id": image_id,
                            "settings_id": settings_id,
                            "image": uploaded_file.name,
                            "caption": caption,
                        }
                        add_to_history(uploaded_file.name, caption, caption_model)
                except (HfHubHTTPError, ImportError, OSError, RuntimeError, ValueError) as exc:
                    st.error(
                        f"Caption generation failed: {exc}. "
                        "Check your internet connection for the first SmolVLM download, "
                        "and confirm the project requirements are installed."
                    )

            result = st.session_state.get("single_result")
            if (
                result
                and result["image_id"] == image_id
                and result["settings_id"] == settings_id
            ):
                st.markdown('<div class="caption-label">Generated caption</div>', unsafe_allow_html=True)
                st.markdown(
                    f'<div class="caption-result">{escape(result["caption"])}</div>',
                    unsafe_allow_html=True,
                )
                edited_caption = st.text_area(
                    "Copy or edit your caption",
                    value=result["caption"],
                    height=82,
                    label_visibility="collapsed",
                    key=f"caption_edit_{image_id}",
                )
                st.download_button(
                    "Download caption (.txt)",
                    data=edited_caption,
                    file_name=f"{Path(result['image']).stem}-caption.txt",
                    mime="text/plain",
                    key=f"single_download_{image_id}",
                )
            else:
                st.markdown(
                    """
                    <div class="empty-state">
                        <div class="empty-icon">✧</div>
                        <div class="empty-title">Your caption will appear here</div>
                        <div class="empty-copy">Upload an image, then generate your caption.</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

with batch_tab:
    st.markdown("### Batch captioning")
    st.markdown(
        "Upload a group of images and create a downloadable caption spreadsheet. "
        f"Up to {MAX_BATCH_SIZE} images per batch.",
    )
    batch_files = st.file_uploader(
        "Choose images for your batch",
        type=SUPPORTED_FORMATS,
        accept_multiple_files=True,
        key="batch_upload",
    )
    batch_fingerprint = tuple(
        (batch_file.name, image_digest(batch_file.getvalue()))
        for batch_file in batch_files
    ) + ((caption_model, caption_task, str(max_new_tokens), str(num_beams)),)
    if len(batch_files) > MAX_BATCH_SIZE:
        st.error(f"Please select no more than {MAX_BATCH_SIZE} images at a time.")
    elif batch_files:
        st.caption(f"{len(batch_files)} image(s) selected. Images are processed one at a time to keep memory use low.")
        if st.button("Generate batch captions", type="primary", key="batch_generate"):
            batch_rows = []
            progress = st.progress(0, text="Preparing images...")
            try:
                if caption_model == "SmolVLM":
                    load_detailed_model()
                else:
                    load_blip_model()
            except (HfHubHTTPError, ImportError, OSError, RuntimeError, ValueError) as exc:
                st.error(f"The caption model could not be loaded: {exc}")
            else:
                for index, batch_file in enumerate(batch_files):
                    image_bytes = batch_file.getvalue()
                    try:
                        with Image.open(io.BytesIO(image_bytes)) as source:
                            source.verify()
                        caption = generate_caption(
                            image_bytes,
                            caption_model,
                            caption_task,
                            max_new_tokens,
                            num_beams,
                            min_new_tokens,
                        )
                    except (
                        ImportError,
                        UnidentifiedImageError,
                        HfHubHTTPError,
                        OSError,
                        RuntimeError,
                        ValueError,
                        Image.DecompressionBombError,
                    ) as exc:
                        st.warning(f"Could not caption {batch_file.name}: {exc}")
                        caption = ""
                    if caption:
                        row = {
                            "image": batch_file.name,
                            "caption": caption,
                            "generated_at": datetime.now().astimezone().strftime("%Y-%m-%d %H:%M"),
                            "mode": f"Batch · {caption_model}",
                        }
                        batch_rows.append(row)
                        add_to_history(batch_file.name, caption, "Batch")
                    else:
                        st.warning(f"The model returned an empty caption for {batch_file.name}.")
                    progress.progress(
                        (index + 1) / len(batch_files),
                        text=f"Processed {index + 1} of {len(batch_files)} images",
                    )
                st.session_state["batch_results"] = batch_rows
                st.session_state["batch_fingerprint"] = batch_fingerprint

    batch_results = st.session_state.get("batch_results", [])
    if st.session_state.get("batch_fingerprint") != batch_fingerprint:
        batch_results = []
    if batch_results:
        st.markdown("#### Batch results")
        st.dataframe(
            [{"Image": row["image"], "Caption": row["caption"]} for row in batch_results],
            width="stretch",
            hide_index=True,
        )
        st.download_button(
            "Download results as CSV",
            data=csv_content(batch_results),
            file_name="image-captions.csv",
            mime="text/csv",
            key="batch_download",
        )

with history_tab:
    st.markdown("### Recent captions")
    history = st.session_state.get("history", [])
    if history:
        history_action, history_summary = st.columns([1, 3])
        with history_action:
            if st.button("Clear history", key="clear_history"):
                st.session_state["history"] = []
                st.rerun()
        with history_summary:
            st.caption(f"{len(history)} of 100 recent captions · kept only for this session")
        st.dataframe(
            [
                {
                    "Image": row["image"],
                    "Caption": row["caption"],
                    "Generated": row["generated_at"],
                    "Mode": row["mode"],
                }
                for row in history
            ],
            width="stretch",
            hide_index=True,
        )
        st.download_button(
            "Export history as CSV",
            data=csv_content(history),
            file_name="caption-history.csv",
            mime="text/csv",
            key="history_download",
        )
    else:
        st.info("Your generated captions will show up here. History is private to this browser session.")

st.markdown(
    """
    <div class="feature-row">
        <div class="feature"><div class="feature-title">⌂ &nbsp; Private by design</div><div class="feature-copy">Your images stay on your device.</div></div>
        <div class="feature"><div class="feature-title">✦ &nbsp; Powered by BLIP</div><div class="feature-copy">Natural captions from a vision model.</div></div>
        <div class="feature"><div class="feature-title">◉ &nbsp; No API required</div><div class="feature-copy">Works offline with the local model.</div></div>
    </div>
    <div class="footer">CaptionAI &nbsp;·&nbsp; Local image captioning, made effortless</div>
    """,
    unsafe_allow_html=True,
)
