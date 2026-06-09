import streamlit as st
import tensorflow as tf
import tensorflow_hub as hub
import numpy as np
import pandas as pd

# ==========================================================
# PAGE CONFIGURATION
# ==========================================================

st.set_page_config(
    page_title="SkimLit | Scientific Abstract Classifier",
    page_icon="🧠",
    layout="wide"
)

# ==========================================================
# LABELS & COLORS
# ==========================================================

LABELS = [
    "BACKGROUND",
    "OBJECTIVE",
    "METHODS",
    "RESULTS",
    "CONCLUSIONS"
]

COLORS = {
    "BACKGROUND": "#3B82F6",
    "OBJECTIVE": "#8B5CF6",
    "METHODS": "#F59E0B",
    "RESULTS": "#10B981",
    "CONCLUSIONS": "#EF4444"
}

# ==========================================================
# MODEL LOADING
# ==========================================================

@st.cache_resource
def load_model():
    model = tf.keras.models.load_model(
        "skimlit_model1.keras",
        custom_objects={"KerasLayer": hub.KerasLayer},
        compile=False
    )
    return model

model = load_model()

# ==========================================================
# HEADER
# ==========================================================

st.title("🧠 SkimLit AI: Medical Abstract Assistant")
st.markdown("---")
st.markdown(
    """
    Automatically classify each sentence of a scientific abstract into its
    rhetorical section using a deep learning model powered by the
    **Deep Learning**.
    """
)

# ==========================================================
# SIDEBAR
# ==========================================================

with st.sidebar:

    st.image(
        "images/skimlit_logo.png",
        width=135
    )
    
    st.header("**About**")

    st.markdown(
        """
        **SkimLit** is a deep learning application that identifies the
        structural role of individual sentences within Medical abstracts.

        ### Supported Sections
        - Background
        - Objective
        - Methods
        - Results
        - Conclusions
        """
    )

# ==========================================================
# INPUT
# ==========================================================

st.subheader("📄 Add Abstract Here...")

example_text = """Type 2 diabetes is a major public health concern associated with significant morbidity and mortality.
This study evaluated the effectiveness of a mobile health intervention for glycemic control in adults with type 2 diabetes.
A randomized controlled trial was conducted involving 500 participants across five clinical centers.
Participants in the intervention group received personalized lifestyle recommendations through a mobile application for 12 months.
The intervention group demonstrated significantly lower HbA1c levels compared with the control group at follow-up.
Mobile health interventions may improve long-term diabetes management and clinical outcomes.
"""

abstract_text = st.text_area(
    "Enter a scientific abstract (one sentence per line)",
    value=example_text,
    height=200,
    help="Each line should contain a single sentence from the abstract."
)

# ==========================================================
# PREPROCESSING
# ==========================================================

def preprocess(text):
    return [
        line.strip()
        for line in text.split("\n")
        if line.strip()
    ]

# ==========================================================
# PREDICTION
# ==========================================================

def predict(text):

    sentences = preprocess(text)

    if len(sentences) == 0:
        return None

    probabilities = model.predict(
        tf.constant(sentences),
        verbose=0
    )

    predicted_classes = np.argmax(probabilities, axis=1)

    results = []

    for sentence, pred, conf in zip(
        sentences,
        predicted_classes,
        probabilities.max(axis=1)
    ):
        results.append({
            "Sentence": sentence,
            "Section": LABELS[pred],
            "Confidence": round(float(conf), 4)
        })

    return pd.DataFrame(results)

# ==========================================================
# ANALYSIS BUTTON
# ==========================================================

if st.button("🔍 Analyze Abstract", use_container_width=True):

    with st.spinner("Running classification model..."):

        df = predict(abstract_text)

    if df is None:
        st.warning(
            "No text detected. Please enter at least one sentence from a scientific abstract."
        )
        st.stop()

    st.success(
        f"Classification completed successfully. {len(df)} sentence(s) analyzed."
    )

    # ======================================================
    # METRICS
    # ======================================================

    total_sentences = len(df)
    avg_confidence = df["Confidence"].mean()

    col1, col2 = st.columns(2)

    with col1:
        st.metric(
            label="Sentences Analyzed",
            value=total_sentences
        )

    with col2:
        st.metric(
            label="Average Confidence",
            value=f"{avg_confidence:.1%}"
        )

    # ======================================================
    # DISTRIBUTION
    # ======================================================

    st.subheader("📊 Section Distribution")

    st.caption(
        "Distribution of predicted rhetorical sections across the abstract."
    )

    counts = df["Section"].value_counts()

    st.bar_chart(counts)

    # ======================================================
    # RESULTS TABLE
    # ======================================================

    st.subheader("📋 Classification Results")

    st.caption(
        "Sentence-level predictions with associated confidence scores."
    )

    st.dataframe(
        df,
        use_container_width=True,
        hide_index=True
    )

    # ======================================================
    # SENTENCE CARDS
    # ======================================================

    st.subheader("🔬 Detailed Structured Abstract")


    for label in LABELS:

        section_df = df[df["Section"] == label]

        if len(section_df) == 0:
            continue

        color = COLORS[label]

        st.markdown(
            f"""
            <h4 style="
                color:{color};
                margin-top:25px;
                margin-bottom:10px;
                border-bottom:2px solid {color};
                padding-bottom:6px;
            ">
                {label}
            </h4>
            """,
            unsafe_allow_html=True
        )

        for sentence in section_df["Sentence"]:
            st.markdown(
                f"""
                <div style="
                    padding:8px 0;
                    font-size:16px;
                    line-height:1.7;
                ">
                    {sentence}
                </div>
                """,
                unsafe_allow_html=True
            )