import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image


# =========================================================
# PAGE CONFIG
# =========================================================

st.set_page_config(
    page_title="LeafGuard AI",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="expanded"
)


# =========================================================
# SESSION STATE
# =========================================================

if "dark_mode" not in st.session_state:
    st.session_state.dark_mode = False

if "prediction_done" not in st.session_state:
    st.session_state.prediction_done = False

if "disease" not in st.session_state:
    st.session_state.disease = ""

if "confidence" not in st.session_state:
    st.session_state.confidence = 0.0


# =========================================================
# THEME COLORS
# =========================================================

if st.session_state.dark_mode:

    background = "#07100A"
    card = "#101A13"

    text = "#F8FFF9"
    heading = "#FFFFFF"
    muted = "#D1DDD4"
    secondary = "#BFCBC2"

    green = "#22C55E"
    green_dark = "#15803D"

    light_green = "#12351D"
    border = "#25452E"

    upload_button_bg = "#101A13"
    upload_button_text = "#F8FFF9"
    upload_button_border = "#25452E"

else:

    background = "#F4FAF5"
    card = "#FFFFFF"

    text = "#111827"
    heading = "#0B1F12"
    muted = "#374151"
    secondary = "#1F2937"

    green = "#166534"
    green_dark = "#14532D"

    light_green = "#DCFCE7"
    border = "#B7D8BE"

    upload_button_bg = "#DCFCE7"
    upload_button_text = "#14532D"
    upload_button_border = "#B7D8BE"


# =========================================================
# CSS
# =========================================================

st.markdown(
    f"""
    <style>

    /* =====================================================
       GLOBAL BACKGROUND
       ===================================================== */

    .stApp {{
        background-color: {background} !important;
        color: {text} !important;
    }}

    [data-testid="stAppViewContainer"] {{
        background-color: {background} !important;
    }}

    [data-testid="stHeader"] {{
        background-color: {background} !important;
    }}

    .main {{
        background-color: {background} !important;
    }}

    .block-container {{
        background-color: {background} !important;
        padding-top: 2.5rem !important;
        padding-bottom: 3rem !important;
        max-width: 1200px !important;
    }}


    /* =====================================================
       ALL NORMAL TEXT
       ===================================================== */

    p {{
        color: {text} !important;
    }}

    label {{
        color: {text} !important;
    }}

    .stMarkdown {{
        color: {text} !important;
    }}

    .stCaption {{
        color: {muted} !important;
    }}


    /* =====================================================
       HEADINGS
       ===================================================== */

    h1, h2, h3, h4, h5, h6 {{
        color: {heading} !important;
    }}


    /* =====================================================
       SIDEBAR
       ===================================================== */

    section[data-testid="stSidebar"] {{
        background-color: {card} !important;
        border-right: 1px solid {border};
    }}

    section[data-testid="stSidebar"] * {{
        color: {text} !important;
    }}

    section[data-testid="stSidebar"] .stCaption {{
        color: {muted} !important;
    }}


    /* =====================================================
       HERO
       ===================================================== */

    .smart-badge {{
        display: inline-block;
        background-color: {light_green};
        color: {green} !important;
        padding: 11px 22px;
        border-radius: 40px;
        font-size: 18px;
        font-weight: 800;
        letter-spacing: 0.5px;
        margin-bottom: 15px;
        border: 1px solid {border};
    }}

    .main-title {{
        color: {heading} !important;
        font-size: 46px;
        font-weight: 850;
        line-height: 1.15;
        margin-top: 0;
        margin-bottom: 10px;
    }}

    .main-title span {{
        color: {green} !important;
    }}

    .main-subtitle {{
        color: {muted} !important;
        font-size: 17px;
        line-height: 1.7;
        max-width: 720px;
        margin-bottom: 25px;
    }}


    /* =====================================================
       UPLOAD
       ===================================================== */

    .upload-heading {{
        color: {heading} !important;
        font-size: 22px;
        font-weight: 800;
        margin-top: 25px;
        margin-bottom: 5px;
    }}

    .upload-description {{
        color: {muted} !important;
        font-size: 14px;
        margin-bottom: 12px;
    }}

    [data-testid="stFileUploader"] {{
        background-color: {card} !important;
        border: 2px dashed {border} !important;
        border-radius: 20px !important;
        padding: 18px !important;
    }}

    [data-testid="stFileUploader"] * {{
        color: {text} !important;
    }}

    [data-testid="stFileUploader"]:hover {{
        border-color: {green} !important;
    }}


    /* =====================================================
       BUTTON
       ===================================================== */

    .stButton > button {{
        width: 100%;
        min-height: 48px;
        border-radius: 12px;
        border: 1px solid {green};
        background-color: {green};
        color: #FFFFFF !important;
        font-size: 15px;
        font-weight: 800;
    }}

    .stButton > button:hover {{
        background-color: {green_dark};
        border-color: {green_dark};
        color: #FFFFFF !important;
    }}


    /* =====================================================
       ALERTS
       ===================================================== */

    [data-testid="stAlert"] p {{
        color: {text} !important;
    }}

    [data-testid="stAlert"] {{
        color: {text} !important;
    }}


    /* =====================================================
       METRIC
       ===================================================== */

    [data-testid="stMetricLabel"] {{
        color: {muted} !important;
    }}

    [data-testid="stMetricValue"] {{
        color: {heading} !important;
        font-weight: 800 !important;
    }}

    [data-testid="stMetricDelta"] {{
        color: {text} !important;
    }}


    /* =====================================================
       IMAGE CAPTION
       ===================================================== */

    [data-testid="stImageCaption"] {{
        color: {muted} !important;
    }}


    /* =====================================================
       FILE UPLOADER
       ===================================================== */

    [data-testid="stFileUploaderDropzone"] {{
        background-color: {card} !important;
    }}

    [data-testid="stFileUploaderDropzone"] * {{
        color: {text} !important;
    }}

    [data-testid="stFileUploader"] button {{
        background-color: {upload_button_bg} !important;
        color: {upload_button_text} !important;
        border: 1px solid {upload_button_border} !important;
        border-radius: 8px !important;
        font-weight: 700 !important;
    }}

    [data-testid="stFileUploader"] button:hover {{
        background-color: {upload_button_bg} !important;
        color: {upload_button_text} !important;
        border-color: {upload_button_border} !important;
    }}


    /* =====================================================
       FOOTER
       ===================================================== */

    .footer {{
        color: {muted} !important;
        font-size: 12px;
        text-align: center;
    }}

    </style>
    """,
    unsafe_allow_html=True
)


# =========================================================
# LOAD MODELS
# =========================================================

MODEL_PATH = "plant_disease_model.keras"

VALIDATOR_MODEL_PATH = "plant_validator.keras"


@st.cache_resource
def load_model():

    return tf.keras.models.load_model(
        MODEL_PATH
    )


@st.cache_resource
def load_validator_model():

    return tf.keras.models.load_model(
        VALIDATOR_MODEL_PATH
    )


# =========================================================
# LOAD DISEASE MODEL
# =========================================================

try:

    model = load_model()

    model_error = None

except Exception as e:

    model = None

    model_error = str(e)


# =========================================================
# LOAD VALIDATOR MODEL
# =========================================================

try:

    validator_model = load_validator_model()

    validator_error = None

except Exception as e:

    validator_model = None

    validator_error = str(e)


# =========================================================
# CLASS NAMES
# =========================================================

CLASS_NAMES = [

    "Apple___Apple_scab",
    "Apple___Black_rot",
    "Apple___Cedar_apple_rust",
    "Apple___healthy",

    "Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot",
    "Corn_(maize)___Common_rust_",
    "Corn_(maize)___healthy",
    "Corn_(maize)___Northern_Leaf_Blight",

    "Grape___Black_rot",
    "Grape___Esca_(Black_Measles)",
    "Grape___Leaf_blight_(Isariopsis_Leaf_Spot)",
    "Grape___healthy",

    "Potato___Early_blight",
    "Potato___Late_blight",
    "Potato___healthy",

    "Tomato___Bacterial_spot",
    "Tomato___Early_blight",
    "Tomato___Late_blight",
    "Tomato___Leaf_Mold",
    "Tomato___Septoria_leaf_spot",
    "Tomato___Spider_mites Two-spotted_spider_mite",
    "Tomato___Target_Spot",
    "Tomato___Tomato_Yellow_Leaf_Curl_Virus",
    "Tomato___Tomato_mosaic_virus",
    "Tomato___healthy"
]


# =========================================================
# GUIDANCE
# =========================================================

GUIDANCE = {

    "Apple___Apple_scab": {

        "management": [
            "Remove badly affected leaves.",
            "Keep good air circulation around the plant.",
            "Monitor new leaves regularly."
        ],

        "prevention": [
            "Avoid unnecessary moisture on leaves.",
            "Keep enough space between plants.",
            "Remove fallen infected leaves."
        ]
    },


    "Apple___healthy": {

        "management": [
            "The leaf appears healthy.",
            "Continue regular plant monitoring.",
            "Maintain suitable watering and nutrition."
        ],

        "prevention": [
            "Keep the growing area clean.",
            "Provide good sunlight and air circulation.",
            "Check leaves regularly."
        ]
    },


    "Tomato___Early_blight": {

        "management": [
            "Remove severely affected leaves.",
            "Keep the area around the plant clean.",
            "Avoid unnecessary wetting of leaves."
        ],

        "prevention": [
            "Maintain good spacing between plants.",
            "Water near the soil instead of the leaves.",
            "Monitor lower leaves regularly."
        ]
    },


    "Tomato___Late_blight": {

        "management": [
            "Remove severely affected plant material.",
            "Keep infected material away from healthy plants.",
            "Monitor nearby plants."
        ],

        "prevention": [
            "Improve air circulation.",
            "Avoid prolonged leaf wetness.",
            "Inspect plants regularly."
        ]
    },


    "Tomato___healthy": {

        "management": [
            "The leaf appears healthy.",
            "Continue regular monitoring.",
            "Maintain suitable growing conditions."
        ],

        "prevention": [
            "Keep plants clean and well spaced.",
            "Provide adequate sunlight.",
            "Check plants regularly."
        ]
    }
}


# =========================================================
# SIDEBAR
# =========================================================

with st.sidebar:

    st.markdown("# 🌿 LeafGuard AI")

    st.caption(
        "Plant Health Assistant"
    )

    st.divider()

    st.subheader(
        "🏠 Navigation"
    )

    st.success(
        "📷 Check My Plant"
    )

    st.divider()

    st.subheader(
        "📌 How It Works"
    )

    st.markdown(
        "**① 📷 Take a Photo**"
    )

    st.caption(
        "Take a clear photo of your plant leaf."
    )

    st.markdown(
        "**② ⬆️ Upload**"
    )

    st.caption(
        "Upload your photo here."
    )

    st.markdown(
        "**③ 🧠 Check**"
    )

    st.caption(
        "AI checks the leaf and gives a result."
    )

    st.divider()

    st.subheader(
        "💡 Photo Tips"
    )

    st.markdown(
        """
        ☀️ Use good lighting

        🍃 Keep the leaf visible

        📸 Avoid blurry photos

        🌱 Prefer one clear leaf
        """
    )

    st.divider()

    st.subheader(
        "🎨 Appearance"
    )

    if st.session_state.dark_mode:

        if st.button(
            "☀️ Light Mode"
        ):

            st.session_state.dark_mode = False

            st.rerun()

    else:

        if st.button(
            "🌙 Dark Mode"
        ):

            st.session_state.dark_mode = True

            st.rerun()


# =========================================================
# HERO
# =========================================================

st.markdown(
    """
    <div class="smart-badge">
        🌱 SMART PLANT HEALTH CHECK
    </div>

    <div class="main-title">
        Keep Your <span>Plants Healthy</span> 🌿
    </div>

    <div class="main-subtitle">
        Upload a clear photo of your plant leaf and let AI
        help identify possible plant diseases.
    </div>
    """,
    unsafe_allow_html=True
)


# =========================================================
# UPLOAD
# =========================================================

st.markdown(
    '<div class="upload-heading">'
    '📷 Upload Your Leaf Photo'
    '</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="upload-description">'
    'Choose a clear photo of one plant leaf • '
    'JPG • JPEG • PNG'
    '</div>',
    unsafe_allow_html=True
)

uploaded_file = st.file_uploader(
    "Upload leaf photo",
    type=[
        "jpg",
        "jpeg",
        "png"
    ],
    label_visibility="collapsed"
)


# =========================================================
# BEFORE UPLOAD
# =========================================================

if uploaded_file is None:

    st.session_state.prediction_done = False

    st.session_state.disease = ""

    st.session_state.confidence = 0.0

    st.info(
        "📸 Upload a clear plant leaf photo to get started."
    )

    st.write("")

    col1, col2, col3 = st.columns(3)

    with col1:

        st.info(
            "🌿 **Easy to Use**\n\n"
            "Simple photo upload"
        )

    with col2:

        st.info(
            "🧠 **AI Powered**\n\n"
            "CNN-based analysis"
        )

    with col3:

        st.info(
            "📊 **Clear Result**\n\n"
            "Simple health guidance"
        )
        # =========================================================
# AFTER UPLOAD
# =========================================================

else:

    image = Image.open(
        uploaded_file
    ).convert("RGB")

    st.divider()

    st.subheader(
        "🖼️ Your Leaf Photo"
    )

    col1, col2 = st.columns(
        [1.1, 0.9]
    )

    with col1:

        st.image(
            image,
            caption="Uploaded Leaf",
            use_container_width=True
        )

    with col2:

        st.success(
            "✅ Image Ready"
        )

        st.write(
            "Your photo has been uploaded successfully."
        )

        st.write(
            "Click below to check your plant health."
        )

        check_button = st.button(
            "🔍 Check My Plant",
            use_container_width=True
        )


    # =====================================================
    # PREDICTION
    # =====================================================

    if check_button:

        if model is None:

            st.error(
                "❌ Model could not be loaded."
            )

            st.code(
                model_error
            )

        elif validator_model is None:

            st.error(
                "❌ Plant validator model could not be loaded."
            )

            st.code(
                validator_error
            )

        else:

            with st.spinner(
                "🧠 Checking your plant..."
            ):

                try:

                    # =================================================
                    # DISEASE MODEL INPUT SIZE
                    # =================================================

                    input_height = model.input_shape[1]
                    input_width = model.input_shape[2]


                    # =================================================
                    # PREPARE DISEASE MODEL IMAGE
                    # =================================================

                    img = image.resize(
                        (
                            input_width,
                            input_height
                        )
                    )

                    img_array = np.array(
                        img
                    ).astype("float32")

                    img_array = (
                        img_array / 255.0
                    )

                    img_array = np.expand_dims(
                        img_array,
                        axis=0
                    )


                    # =================================================
                    # PLANT / NON-PLANT VALIDATION
                    # =================================================

                    validator_img = image.resize(
                        (128, 128)
                    )

                    validator_array = np.array(
                        validator_img
                    ).astype("float32")

                    validator_array = np.expand_dims(
                        validator_array,
                        axis=0
                    )


                    # =================================================
                    # VALIDATOR PREDICTION
                    # =================================================

                    validator_prediction = (
                        validator_model.predict(
                            validator_array,
                            verbose=0
                        )
                    )

                    validator_index = int(
                        np.argmax(
                            validator_prediction[0]
                        )
                    )

                    validator_confidence = float(
                        validator_prediction[0][
                            validator_index
                        ]
                    ) * 100


                    # =================================================
                    # 0 = NON-PLANT
                    # 1 = PLANT
                    # =================================================

                    if validator_index == 0:

                        st.session_state.prediction_done = False
                        st.session_state.disease = ""
                        st.session_state.confidence = 0.0

                        st.error(
                            "❌ This does not appear to be "
                            "a plant leaf."
                        )

                        st.info(
                            "🌿 Please upload a clear photo "
                            "of a single plant leaf."
                        )


                    # =================================================
                    # PLANT DETECTED
                    # =================================================

                    else:

                        prediction = model.predict(
                            img_array,
                            verbose=0
                        )

                        predicted_index = int(
                            np.argmax(
                                prediction[0]
                            )
                        )

                        confidence = float(
                            prediction[0][
                                predicted_index
                            ]
                        ) * 100


                        # =================================================
                        # CONFIDENCE CHECK
                        # =================================================

                        if confidence < 60:

                            st.session_state.prediction_done = False

                            st.warning(
                                "⚠️ Unable to identify this plant leaf.\n\n"
                                "Please upload a clear photo of a single "
                                "plant leaf from a supported crop."
                            )

                        else:

                            if predicted_index < len(
                                CLASS_NAMES
                            ):

                                disease = CLASS_NAMES[
                                    predicted_index
                                ]

                            else:

                                disease = "Unknown"


                            st.session_state.disease = (
                                disease
                            )

                            st.session_state.confidence = (
                                confidence
                            )

                            st.session_state.prediction_done = (
                                True
                            )

                            st.rerun()


                except Exception as e:

                    st.error(
                        "❌ Error while analyzing the image."
                    )

                    st.exception(e)


# =========================================================
# RESULTS
# =========================================================

if st.session_state.prediction_done:

    disease = (
        st.session_state.disease
    )

    confidence = (
        st.session_state.confidence
    )


    readable_disease = (
        disease
        .replace(
            "___",
            " - "
        )
        .replace(
            "_",
            " "
        )
    )


    st.divider()

    st.header(
        "🌿 Plant Health Result"
    )

    st.write(
        "Your plant leaf has been analyzed successfully."
    )

    st.write("")


    # =====================================================
    # POSSIBLE PROBLEM
    # =====================================================

    col1, col2 = st.columns(2)

    with col1:

        st.subheader(
            "🦠 Possible Problem"
        )

        st.success(
            readable_disease
        )


    # =====================================================
    # CONFIDENCE
    # =====================================================

    with col2:

        st.subheader(
            "📊 Confidence"
        )

        st.metric(
            "Prediction Confidence",
            f"{confidence:.2f}%"
        )

        st.progress(
            min(
                confidence / 100,
                1.0
            )
        )


    # =====================================================
    # GUIDANCE
    # =====================================================

    guidance = GUIDANCE.get(
        disease,
        {
            "management": [
                "Monitor the plant regularly.",
                "Remove severely affected plant material if needed.",
                "Consider consulting a local agriculture expert."
            ],

            "prevention": [
                "Maintain good plant hygiene.",
                "Provide suitable spacing and airflow.",
                "Monitor plants regularly for early symptoms."
            ]
        }
    )


    # =====================================================
    # MANAGEMENT
    # =====================================================

    st.subheader(
        "💚 What You Can Do"
    )

    st.info(
        "These are general management suggestions. "
        "For serious crop problems, consult a local agriculture expert."
    )

    for item in guidance[
        "management"
    ]:

        st.write(
            f"✓ {item}"
        )


    # =====================================================
    # PREVENTION
    # =====================================================

    st.subheader(
        "🛡️ How to Prevent It"
    )

    for item in guidance[
        "prevention"
    ]:

        st.write(
            f"✓ {item}"
        )


    # =====================================================
    # DISCLAIMER
    # =====================================================

    st.warning(
        "🌱 AI results are a preliminary identification "
        "and should not replace professional agricultural advice."
    )


# =========================================================
# FOOTER
# =========================================================

st.divider()

st.caption(
    "🌿 LeafGuard AI • CNN-Based Plant Disease Detection • "
    "Built for smarter and healthier farming"
)