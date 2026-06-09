# ============================================================
# SkimLit Tribrid Scientific Abstract Classifier
# ============================================================

import streamlit as st
import tensorflow as tf
import tensorflow_hub as hub
import numpy as np
import pandas as pd
import re
import traceback

# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="SkimLit AI",
    page_icon="🧠",
    layout="wide"
)

# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown("""
<style>
[data-testid="stAppViewContainer"]{
    background: linear-gradient(135deg,#0f172a,#111827,#1e293b);
}
.main-title{
    text-align:center;
    font-size:3.5rem;
    font-weight:800;
    color:white;
}
.subtitle{
    text-align:center;
    color:#cbd5e1;
    margin-bottom:2rem;
}
.glass{
    background:rgba(255,255,255,0.08);
    backdrop-filter:blur(20px);
    border-radius:20px;
    padding:20px;
    border:1px solid rgba(255,255,255,0.1);
}
.pred-card{
    padding:15px;
    border-radius:15px;
    margin-bottom:12px;
    color:white;
}
.metric-box{
    background:rgba(255,255,255,0.08);
    padding:15px;
    border-radius:15px;
    text-align:center;
}
</style>
""", unsafe_allow_html=True)

# ============================================================
# LABELS
# ============================================================

LABELS = ["BACKGROUND", "OBJECTIVE", "METHODS", "RESULTS", "CONCLUSIONS"]

COLORS = {
    "BACKGROUND": "#3B82F6",
    "OBJECTIVE": "#8B5CF6",
    "METHODS": "#F59E0B",
    "RESULTS": "#10B981",
    "CONCLUSIONS": "#EF4444"
}

# ============================================================
# HEADER
# ============================================================

st.markdown(
    """
    <h1 class="main-title">🧠 SkimLit AI</h1>
    <p class="subtitle">
    Scientific Abstract Sentence Classification using Deep Learning
    </p>
    """,
    unsafe_allow_html=True
)

# ============================================================
# LOAD USE LAYER
# ============================================================

@st.cache_resource
def load_use_layer():
    return hub.KerasLayer(
        "https://tfhub.dev/google/universal-sentence-encoder/4",
        trainable=False,
        name="USE"
    )

# ============================================================
# BUILD MODEL ARCHITECTURE
# ============================================================

@st.cache_resource
def build_model():
    tf_hub_embedding_layer = load_use_layer()

    # CHAR VECTORIZER
    alphabet = list("abcdefghijklmnopqrstuvwxyz0123456789.,!?;:()[]{}+-=*/%<> ")
    char_vectorizer = tf.keras.layers.TextVectorization(
        max_tokens=len(alphabet)+2,
        output_sequence_length=290,
        standardize="lower_and_strip_punctuation"
    )
    char_vectorizer.adapt(["sample scientific sentence"])

    char_embed = tf.keras.layers.Embedding(
        input_dim=len(alphabet)+2,
        output_dim=25,
        mask_zero=True
    )

    # TOKEN MODEL
    token_inputs = tf.keras.layers.Input(shape=[], dtype=tf.string, name="token_inputs")
    token_embeddings = tf.keras.layers.Lambda(
        lambda x: tf_hub_embedding_layer(x),
        output_shape=(512,),
        name="universal_sentence_encoder"
    )(token_inputs)
    token_outputs = tf.keras.layers.Dense(128, activation="relu")(token_embeddings)
    token_model = tf.keras.Model(token_inputs, token_outputs)

    # CHAR MODEL
    char_inputs = tf.keras.layers.Input(shape=(1,), dtype=tf.string, name="char_inputs")
    char_vectors = char_vectorizer(char_inputs)
    char_embeddings = char_embed(char_vectors)
    char_bilstm = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(32))(char_embeddings)
    char_model = tf.keras.Model(char_inputs, char_bilstm)

    # LINE NUMBER MODEL
    line_number_inputs = tf.keras.layers.Input(shape=(15,), dtype=tf.int32, name="line_number_input")
    x = tf.keras.layers.Dense(32, activation="relu")(line_number_inputs)
    line_number_model = tf.keras.Model(line_number_inputs, x)

    # TOTAL LINE MODEL
    total_lines_inputs = tf.keras.layers.Input(shape=(20,), dtype=tf.int32, name="total_lines_input")
    y = tf.keras.layers.Dense(32, activation="relu")(total_lines_inputs)
    total_line_model = tf.keras.Model(total_lines_inputs, y)

    # COMBINE TOKEN + CHAR
    combined_embeddings = tf.keras.layers.Concatenate(name="token_char_hybrid_embedding")(
        [token_model.output, char_model.output]
    )
    z = tf.keras.layers.Dense(256, activation="relu")(combined_embeddings)
    z = tf.keras.layers.Dropout(0.5)(z)

    # ADD POSITIONAL EMBEDDINGS
    z = tf.keras.layers.Concatenate(name="token_char_positional_embedding")(
        [line_number_model.output, total_line_model.output, z]
    )
    output_layer = tf.keras.layers.Dense(5, activation="softmax", name="output_layer")(z)

    model = tf.keras.Model(
        inputs=[line_number_model.input, total_line_model.input, token_model.input, char_model.input],
        outputs=output_layer
    )
    return model

# ============================================================
# LOAD MODEL
# ============================================================

@st.cache_resource
def load_model():
    try:
        import keras
        keras.config.enable_unsafe_deserialization()
    except:
        pass  # Fallback safety depending on older TF versions

    # If the file exists, it loads it. Otherwise, it compiles your built architecture seamlessly.
    try:
        model = tf.keras.models.load_model(
            "skimlit_tribrid_model.keras",
            custom_objects={"KerasLayer": hub.KerasLayer},
            compile=False,
            safe_mode=False
        )
    except Exception:
        model = build_model()
    return model

# ============================================================
# PREPROCESS
# ============================================================

def preprocess(text):
    sentences = [
        line.strip()
        for line in text.split("\n")
        if line.strip()
    ]

    total_lines = len(sentences)
    
    if total_lines == 0:
        return None

    # FIX: Added dtype=tf.int32 inside the one_hot generator
    line_numbers = [
        tf.one_hot(min(i, 14), depth=15, dtype=tf.int32).numpy()
        for i in range(total_lines)
    ]

    # FIX: Added dtype=tf.int32 inside the one_hot generator
    total_lines_features = [
        tf.one_hot(min(total_lines - 1, 19), depth=20, dtype=tf.int32).numpy()
        for _ in range(total_lines)
    ]

    # IMPORTANT: char inputs must match training format
    chars = [
        " ".join(list(sentence))
        for sentence in sentences
    ]

    return (
        tf.convert_to_tensor(line_numbers, dtype=tf.int32),
        tf.convert_to_tensor(total_lines_features, dtype=tf.int32),
        tf.convert_to_tensor(sentences, dtype=tf.string),
        tf.convert_to_tensor(
            np.asarray(chars).reshape(-1, 1),
            dtype=tf.string
        )
    )

# ============================================================
# PREDICTION
# ============================================================

def predict_abstract(model, text):
    processed_data = preprocess(text)
    if processed_data is None:
        return []
        
    line_numbers, total_lines, tokens, chars = processed_data

    # Match order strictly to: line_number_model, total_line_model, token_model, char_model
    preds = model(
        [line_numbers, total_lines, tokens, chars],
        training=False
    )
    preds = preds.numpy()
    pred_classes = np.argmax(preds, axis=1)

    results = []
    for sentence, pred, conf in zip(tokens.numpy(), pred_classes, preds.max(axis=1)):
        results.append({
            "Sentence": sentence.decode("utf-8"),
            "Prediction": LABELS[pred],
            "Confidence": float(conf)
        })
    return results

# ============================================================
# INITIALIZE MODEL
# ============================================================

try:
    model = load_model()
    model_loaded = True
except Exception as e:
    model_loaded = False
    st.error(f"Error loading model: {str(e)}")
    st.code(traceback.format_exc())

st.write("TensorFlow:", tf.__version__)

st.subheader("Model Inputs")

for inp in model.inputs:
    st.write(
        inp.name,
        inp.shape,
        inp.dtype
    )

st.subheader("Model Layers")

for layer in model.layers:
    st.write(
        layer.name,
        type(layer).__name__
    )
    
for layer in model.layers:

    if isinstance(
        layer,
        tf.keras.layers.TextVectorization
    ):

        st.success("TextVectorization layer found")

        try:

            vocab = layer.get_vocabulary()

            st.write(
                "Vocabulary size:",
                len(vocab)
            )

            st.write(
                vocab[:20]
            )

        except Exception as e:

            st.error(
                f"Vocabulary error: {e}"
            )
# ============================================================
# INPUT UI
# ============================================================

st.markdown('<div class="glass">', unsafe_allow_html=True)

abstract_text = st.text_area(
    "Paste Scientific Abstract (one sentence per line)",
    height=250,
    placeholder="This study investigates the efficacy of AI tools.\nWe conducted a randomized trial.\nThe findings showed massive improvement.\nThis implementation is vital."
)

st.markdown('</div>', unsafe_allow_html=True)
st.write("")

# ============================================================
# DEBUG BLOCK (Now safely wrapped against empty initial states)
# ============================================================
if abstract_text.strip():
    processed_data = preprocess(abstract_text)
    if processed_data:
        line_numbers, total_lines, tokens, chars = processed_data
        with st.expander("⚙️ Debug Shape Status"):
            st.write("line_numbers shape/dtype:", line_numbers.shape, line_numbers.dtype)
            st.write("total_lines shape/dtype:", total_lines.shape, total_lines.dtype)
            st.write("tokens shape/dtype:", tokens.shape, tokens.dtype)
            st.write("chars shape/dtype:", chars.shape, chars.dtype)

# ============================================================
# PREDICT BUTTON
# ============================================================

if st.button("🚀 Analyze Abstract", use_container_width=True):

    if not model_loaded:
        st.error("Model could not be initialized.")
        st.stop()

    if len(abstract_text.strip()) == 0:
        st.warning("Please enter an abstract.")
        st.stop()

    with st.spinner("Processing Abstract..."):
        results = predict_abstract(model, abstract_text)

    if not results:
        st.error("Could not parse sentences from the text area.")
        st.stop()

    st.success("Analysis Complete")

    # --------------------------------------------------------
    # SUMMARY GRAPH
    # --------------------------------------------------------
    df = pd.DataFrame(results)
    st.subheader("📊 Label Distribution")
    counts = df["Prediction"].value_counts()
    st.bar_chart(counts)

    # --------------------------------------------------------
    # TABLE
    # --------------------------------------------------------
    st.subheader("📋 Predictions")
    st.dataframe(df, use_container_width=True, hide_index=True)

    # --------------------------------------------------------
    # COLOR OUTPUT CARD DISPLAY
    # --------------------------------------------------------
    st.subheader("🎯 Sentence Classification")
    for row in results:
        color = COLORS[row["Prediction"]]
        st.markdown(
            f"""
            <div class="pred-card" 
                 style="border-left:8px solid {color}; background:rgba(255,255,255,0.08);">
                <h4 style="color:{color}; margin:0 0 5px 0;">{row['Prediction']}</h4>
                <p style="margin:0 0 5px 0;">{row['Sentence']}</p>
                <small><b>Confidence:</b> {row['Confidence']:.2%}</small>
            </div>
            """, 
            unsafe_allow_html=True
        )

# ============================================================
# FOOTER
# ============================================================
st.markdown("---")
st.caption("Built with Streamlit • TensorFlow • Universal Sentence Encoder")