import streamlit as st
from PIL import Image
import torch
from transformers import BlipProcessor, BlipForConditionalGeneration

MODEL_DIR = "models/blip-image-captioning-base"

st.set_page_config(
    page_title="Offline BLIP Image Captioning",
    page_icon="🖼️"
)

st.title("🖼️ Offline BLIP Image Captioning")
st.write("Upload an image and generate a caption using a local BLIP model.")


@st.cache_resource
def load_model():
    processor = BlipProcessor.from_pretrained(
        MODEL_DIR,
        local_files_only=True
    )

    model = BlipForConditionalGeneration.from_pretrained(
        MODEL_DIR,
        local_files_only=True
    )

    model.eval()

    return processor, model


processor, model = load_model()


uploaded_file = st.file_uploader(
    "Upload an image",
    type=["png", "jpg", "jpeg", "webp"]
)


if uploaded_file:

    image = Image.open(uploaded_file).convert("RGB")

    st.image(
        image,
        caption="Uploaded image",
        use_container_width=True
    )

    if st.button("Generate Caption"):

        with st.spinner("Generating caption..."):

            inputs = processor(
                images=image,
                return_tensors="pt"
            )

            with torch.no_grad():

                output_ids = model.generate(
                    **inputs,
                    max_new_tokens=40
                )

            caption = processor.decode(
                output_ids[0],
                skip_special_tokens=True
            )

        st.success("Caption generated!")

        st.text_area(
            "Copy caption",
            value=caption,
            height=100
        )

else:

    st.info("Upload an image to get started.")












# import streamlit as st
# from PIL import Image
# import torch
# from transformers import BlipProcessor, BlipForConditionalGeneration



# MODEL_DIR = "models/blip-image-captioning-base"

# st.set_page_config(
#     page_title="CaptionAI",
#     page_icon="✦",
#     layout="wide",
#     initial_sidebar_state="collapsed",
# )



# st.markdown(
#     """
#     <style>

#     /* ---------- GLOBAL ---------- */

#     @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Space+Grotesk:wght@400;500;600;700&display=swap');

#     html, body, [class*="css"] {
#         font-family: 'DM Sans', sans-serif;
#     }

#     .stApp {
#         background:
#             radial-gradient(circle at 10% 10%, rgba(124, 58, 237, 0.18), transparent 30%),
#             radial-gradient(circle at 90% 20%, rgba(236, 72, 153, 0.14), transparent 30%),
#             radial-gradient(circle at 50% 100%, rgba(6, 182, 212, 0.10), transparent 35%),
#             #07070b;
#         color: #f5f5f7;
#     }

#     /* ---------- HIDE STREAMLIT UI ---------- */

#     #MainMenu {
#         visibility: hidden;
#     }

#     footer {
#         visibility: hidden;
#     }

#     header {
#         background: transparent !important;
#     }

#     [data-testid="stToolbar"] {
#         display: none;
#     }

#     /* ---------- MAIN CONTAINER ---------- */

#     .block-container {
#         max-width: 1180px;
#         padding-top: 2rem;
#         padding-bottom: 4rem;
#     }

#     /* ---------- NAVBAR ---------- */

#     .navbar {
#         display: flex;
#         justify-content: space-between;
#         align-items: center;
#         padding: 8px 0 35px 0;
#     }

#     .brand {
#         font-family: 'Space Grotesk', sans-serif;
#         font-size: 24px;
#         font-weight: 700;
#         letter-spacing: -1px;
#     }

#     .brand-gradient {
#         background: linear-gradient(
#             90deg,
#             #a78bfa,
#             #ec4899,
#             #22d3ee
#         );
#         -webkit-background-clip: text;
#         -webkit-text-fill-color: transparent;
#     }

#     .status {
#         display: inline-flex;
#         align-items: center;
#         gap: 8px;
#         padding: 8px 14px;
#         border: 1px solid rgba(255,255,255,0.10);
#         background: rgba(255,255,255,0.04);
#         border-radius: 999px;
#         font-size: 12px;
#         color: #b7b7c3;
#     }

#     .status-dot {
#         width: 7px;
#         height: 7px;
#         border-radius: 50%;
#         background: #34d399;
#         box-shadow: 0 0 12px rgba(52,211,153,0.8);
#     }

#     /* ---------- HERO ---------- */

#     .hero {
#         text-align: center;
#         padding: 30px 10px 55px;
#     }

#     .eyebrow {
#         display: inline-block;
#         padding: 7px 13px;
#         border-radius: 999px;
#         background: rgba(167,139,250,0.10);
#         border: 1px solid rgba(167,139,250,0.20);
#         color: #c4b5fd;
#         font-size: 12px;
#         font-weight: 600;
#         letter-spacing: 0.5px;
#         margin-bottom: 22px;
#     }

#     .hero h1 {
#         font-family: 'Space Grotesk', sans-serif;
#         font-size: clamp(48px, 7vw, 82px);
#         line-height: 0.98;
#         letter-spacing: -4px;
#         margin: 0;
#         font-weight: 700;
#     }

#     .hero-gradient {
#         background: linear-gradient(
#             90deg,
#             #ffffff 10%,
#             #c4b5fd 38%,
#             #f9a8d4 65%,
#             #67e8f9 90%
#         );
#         -webkit-background-clip: text;
#         -webkit-text-fill-color: transparent;
#     }

#     .hero p {
#         max-width: 620px;
#         margin: 25px auto 0;
#         color: #9b9ba8;
#         font-size: 17px;
#         line-height: 1.7;
#     }

#     /* ---------- GLASS CARD ---------- */

#     .glass-card {
#         background: rgba(255,255,255,0.045);
#         border: 1px solid rgba(255,255,255,0.09);
#         border-radius: 28px;
#         padding: 28px;
#         box-shadow:
#             0 25px 80px rgba(0,0,0,0.35),
#             inset 0 1px 0 rgba(255,255,255,0.05);
#         backdrop-filter: blur(20px);
#     }

#     /* ---------- UPLOAD ---------- */

#     [data-testid="stFileUploader"] {
#         background: rgba(255,255,255,0.025);
#         border: 1px dashed rgba(167,139,250,0.40);
#         border-radius: 20px;
#         padding: 15px;
#         transition: 0.25s ease;
#     }

#     [data-testid="stFileUploader"]:hover {
#         border-color: rgba(236,72,153,0.65);
#         background: rgba(167,139,250,0.045);
#     }

#     [data-testid="stFileUploaderDropzone"] {
#         background: transparent !important;
#     }

#     /* ---------- BUTTON ---------- */

#     .stButton > button {
#         width: 100%;
#         border: none !important;
#         border-radius: 15px;
#         padding: 15px 20px;
#         margin-top: 12px;
#         font-family: 'DM Sans', sans-serif;
#         font-weight: 700;
#         font-size: 15px;
#         color: white;
#         background: linear-gradient(
#             100deg,
#             #7c3aed,
#             #db2777,
#             #0891b2
#         );
#         box-shadow:
#             0 10px 35px rgba(124,58,237,0.25);
#         transition: all 0.25s ease;
#     }

#     .stButton > button:hover {
#         transform: translateY(-2px);
#         box-shadow:
#             0 15px 45px rgba(236,72,153,0.30);
#     }

#     /* ---------- IMAGE ---------- */

#     [data-testid="stImage"] {
#         border-radius: 20px;
#         overflow: hidden;
#     }

#     /* ---------- RESULT ---------- */

#     .result-label {
#         color: #a78bfa;
#         font-size: 12px;
#         font-weight: 700;
#         letter-spacing: 1.5px;
#         text-transform: uppercase;
#         margin-bottom: 12px;
#     }

#     .caption-box {
#         padding: 22px;
#         border-radius: 18px;
#         background:
#             linear-gradient(
#                 135deg,
#                 rgba(124,58,237,0.12),
#                 rgba(236,72,153,0.08)
#             );
#         border: 1px solid rgba(167,139,250,0.18);
#         color: #f5f5f7;
#         font-size: 18px;
#         line-height: 1.65;
#         margin-bottom: 15px;
#     }

#     /* ---------- INFO ---------- */

#     .mini-card {
#         text-align: center;
#         padding: 20px 10px;
#         border-radius: 18px;
#         background: rgba(255,255,255,0.025);
#         border: 1px solid rgba(255,255,255,0.07);
#     }

#     .mini-icon {
#         font-size: 23px;
#         margin-bottom: 7px;
#     }

#     .mini-title {
#         font-size: 13px;
#         font-weight: 600;
#         color: #dddde5;
#     }

#     .mini-text {
#         font-size: 11px;
#         color: #777784;
#         margin-top: 3px;
#     }

#     /* ---------- FOOTER ---------- */

#     .footer {
#         text-align: center;
#         margin-top: 60px;
#         padding-top: 25px;
#         border-top: 1px solid rgba(255,255,255,0.06);
#         color: #62626d;
#         font-size: 12px;
#     }

#     /* ---------- MOBILE ---------- */

#     @media (max-width: 700px) {

#         .block-container {
#             padding-left: 18px;
#             padding-right: 18px;
#         }

#         .hero h1 {
#             font-size: 48px;
#             letter-spacing: -2px;
#         }

#         .hero p {
#             font-size: 15px;
#         }

#         .glass-card {
#             padding: 18px;
#             border-radius: 20px;
#         }

#         .navbar {
#             padding-bottom: 15px;
#         }

#         .status {
#             font-size: 10px;
#         }
#     }

#     </style>
#     """,
#     unsafe_allow_html=True,
# )



# st.markdown(
#     """
#     <div class="navbar">
#         <div class="brand">
#             ✦ <span class="brand-gradient">CaptionAI</span>
#         </div>

#         <div class="status">
#             <span class="status-dot"></span>
#             LOCAL AI • BLIP
#         </div>
#     </div>
#     """,
#     unsafe_allow_html=True,
# )




# st.markdown(
#     """
#     <div class="hero">

#         <div class="eyebrow">
#             ✦ SEE IT. UNDERSTAND IT. DESCRIBE IT.
#         </div>

#         <h1>
#             <span class="hero-gradient">Turn pixels</span><br>
#             into words.
#         </h1>

#         <p>
#             Upload any image and let AI transform what it sees
#             into a natural, meaningful caption — privately,
#             directly on your device.
#         </p>

#     </div>
#     """,
#     unsafe_allow_html=True,
# )




# @st.cache_resource
# def load_model():

#     processor = BlipProcessor.from_pretrained(
#         MODEL_DIR,
#         local_files_only=True
#     )

#     model = BlipForConditionalGeneration.from_pretrained(
#         MODEL_DIR,
#         local_files_only=True
#     )

#     model.eval()

#     return processor, model


# try:

#     processor, model = load_model()

# except Exception as e:

#     st.error("Unable to load the local BLIP model.")

#     st.code(str(e))

#     st.stop()




# left, right = st.columns([1, 1], gap="large")




# with left:

#     st.markdown(
#         '<div class="glass-card">',
#         unsafe_allow_html=True
#     )

#     st.markdown(
#         """
#         <div style="margin-bottom:18px;">
#             <div style="
#                 font-family:'Space Grotesk';
#                 font-size:22px;
#                 font-weight:700;
#             ">
#                 Upload your image
#             </div>

#             <div style="
#                 color:#777784;
#                 font-size:13px;
#                 margin-top:5px;
#             ">
#                 Give the AI something to look at.
#             </div>
#         </div>
#         """,
#         unsafe_allow_html=True
#     )

#     uploaded_file = st.file_uploader(
#         "Drop your image here",
#         type=["png", "jpg", "jpeg", "webp"],
#         label_visibility="visible"
#     )

#     if uploaded_file:

#         image = Image.open(uploaded_file).convert("RGB")

#         st.image(
#             image,
#             use_container_width=True
#         )

#         st.markdown(
#             f"""
#             <div style="
#                 color:#777784;
#                 font-size:12px;
#                 margin-top:10px;
#             ">
#                 ✓ {uploaded_file.name}
#             </div>
#             """,
#             unsafe_allow_html=True
#         )

#     st.markdown("</div>", unsafe_allow_html=True)




# with right:

#     st.markdown(
#         '<div class="glass-card">',
#         unsafe_allow_html=True
#     )

#     st.markdown(
#         """
#         <div style="margin-bottom:18px;">
#             <div style="
#                 font-family:'Space Grotesk';
#                 font-size:22px;
#                 font-weight:700;
#             ">
#                 AI Caption
#             </div>

#             <div style="
#                 color:#777784;
#                 font-size:13px;
#                 margin-top:5px;
#             ">
#                 Your image, translated into words.
#             </div>
#         </div>
#         """,
#         unsafe_allow_html=True
#     )

#     if uploaded_file:

#         generate = st.button(
#             "✦  Generate Caption",
#             use_container_width=True
#         )

#         if generate:

#             with st.spinner("AI is looking at your image..."):

#                 inputs = processor(
#                     images=image,
#                     return_tensors="pt"
#                 )

#                 with torch.no_grad():

#                     output_ids = model.generate(
#                         **inputs,
#                         max_new_tokens=40
#                     )

#                 caption = processor.decode(
#                     output_ids[0],
#                     skip_special_tokens=True
#                 )

#             st.markdown(
#                 '<div class="result-label">✦ Generated caption</div>',
#                 unsafe_allow_html=True
#             )

#             st.markdown(
#                 f'<div class="caption-box">"{caption}"</div>',
#                 unsafe_allow_html=True
#             )

#             st.text_area(
#                 "Copy caption",
#                 value=caption,
#                 height=90,
#                 label_visibility="collapsed"
#             )

#         else:

#             st.markdown(
#                 """
#                 <div style="
#                     min-height:280px;
#                     display:flex;
#                     align-items:center;
#                     justify-content:center;
#                     text-align:center;
#                     color:#686874;
#                 ">
#                     <div>
#                         <div style="font-size:45px;margin-bottom:12px;">
#                             ✦
#                         </div>

#                         <div style="
#                             font-family:'Space Grotesk';
#                             font-size:17px;
#                             color:#aaaab5;
#                         ">
#                             Your caption will appear here
#                         </div>

#                         <div style="
#                             font-size:12px;
#                             margin-top:7px;
#                         ">
#                             Ready when you are.
#                         </div>
#                     </div>
#                 </div>
#                 """,
#                 unsafe_allow_html=True
#             )

#     else:

#         st.markdown(
#             """
#             <div style="
#                 min-height:280px;
#                 display:flex;
#                 align-items:center;
#                 justify-content:center;
#                 text-align:center;
#                 color:#686874;
#             ">
#                 <div>
#                     <div style="font-size:45px;margin-bottom:12px;">
#                         🖼️
#                     </div>

#                     <div style="
#                         font-family:'Space Grotesk';
#                         font-size:17px;
#                         color:#aaaab5;
#                     ">
#                         Nothing here yet
#                     </div>

#                     <div style="
#                         font-size:12px;
#                         margin-top:7px;
#                     ">
#                         Upload an image to get started.
#                     </div>
#                 </div>
#             </div>
#             """,
#             unsafe_allow_html=True
#         )

#     st.markdown("</div>", unsafe_allow_html=True)




# st.markdown("<br>", unsafe_allow_html=True)

# col1, col2, col3 = st.columns(3)

# with col1:
#     st.markdown(
#         """
#         <div class="mini-card">
#             <div class="mini-icon">🔒</div>
#             <div class="mini-title">Private</div>
#             <div class="mini-text">Runs locally on your device</div>
#         </div>
#         """,
#         unsafe_allow_html=True
#     )

# with col2:
#     st.markdown(
#         """
#         <div class="mini-card">
#             <div class="mini-icon">⚡</div>
#             <div class="mini-title">Fast</div>
#             <div class="mini-text">Powered by BLIP AI</div>
#         </div>
#         """,
#         unsafe_allow_html=True
#     )

# with col3:
#     st.markdown(
#         """
#         <div class="mini-card">
#             <div class="mini-icon">☁️</div>
#             <div class="mini-title">Offline</div>
#             <div class="mini-text">No API required</div>
#         </div>
#         """,
#         unsafe_allow_html=True
#     )




# st.markdown(
#     """
#     <div class="footer">
#         ✦ CaptionAI &nbsp;·&nbsp; Powered by BLIP &nbsp;·&nbsp; Runs locally
#     </div>
#     """,
#     unsafe_allow_html=True
# )