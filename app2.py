import streamlit as st
import librosa
import numpy as np
import tensorflow as tf
import pickle
import whisper
import tempfile
import string
import nltk
from nltk.corpus import stopwords
import gdown
import zipfile
import os



def download_models():

    if not os.path.exists("fusion_model.keras"):

        url = "https://drive.google.com/uc?id=1TRliBT-QJiz_fh36OyclFYOPy485WnOC"

        gdown.download(url, "archive.zip", quiet=False)

        with zipfile.ZipFile("archive.zip", "r") as zip_ref:
            zip_ref.extractall()

        os.remove("archive.zip")

download_models()

stop_words = {
'a','about','above','after','again','against','all','am','an','and',
'any','are','as','at','be','because','been','before','being','below',
'between','both','but','by','could','did','do','does','doing','down',
'during','each','few','for','from','further','had','has','have','having',
'he','her','here','hers','herself','him','himself','his','how','i','if',
'in','into','is','it','its','itself','just','me','more','most','my',
'myself','no','nor','not','now','of','off','on','once','only','or',
'other','our','ours','ourselves','out','over','own','same','she','should',
'so','some','such','than','that','the','their','theirs','them','themselves',
'then','there','these','they','this','those','through','to','too','under',
'until','up','very','was','we','were','what','when','where','which','while',
'who','whom','why','with','you','your','yours','yourself','yourselves'
}

st.title("🚨 AI Emergency Call Analyzer")

st.write("Analyze emergency calls using AI")

# ----------------------------
# Load Models
# ----------------------------

@st.cache_resource
def load_models():

    text_model = tf.keras.models.load_model("text_model.keras")
    emotion_model = tf.keras.models.load_model("ravdees-2.keras")
    sound_model = tf.keras.models.load_model("background_model.keras")
    fusion_model = tf.keras.models.load_model("fusion_model.keras")

    return text_model, emotion_model, sound_model, fusion_model

with open("vectorizer_text.pkl","rb") as f:
    vectorizer = pickle.load(f)
with open("label_encoder_fusion.pkl", "rb") as f:
    urgency_encoder = pickle.load(f)
with open("label_encoder_text.pkl",'rb') as f:
    text_encoder = pickle.load(f)
# Feature models
text_feature_model = tf.keras.Model(
    inputs=text_model.inputs,
    outputs=text_model.layers[-2].output
)

emotion_feature_model = tf.keras.Model(
    inputs=emotion_model.inputs,
    outputs=emotion_model.layers[-2].output
)

sound_feature_model = tf.keras.Model(
    inputs=sound_model.inputs,
    outputs=sound_model.layers[-2].output
)

whisper_model = whisper.load_model("small")

def preprocess_text(text):

    # lower case
    text = text.lower()

    # remove punctuation
    text = text.translate(str.maketrans('', '', string.punctuation))

    # remove numbers
    text = ''.join([i for i in text if not i.isdigit()])

    # remove emojis / non-ascii
    text = text.encode('ascii', 'ignore').decode()

    # remove stopwords
    words = text.split()
    words = [w for w in words if w not in stop_words]

    text = " ".join(words)

    return text

# ----------------------------
# Function: Process Audio
# ----------------------------


def analyze_audio(audio_path):

    audio, sr = librosa.load(audio_path, sr=16000)

    # Speech to Text
    result = whisper_model.transcribe(
        audio_path,
        language="hi",
        task="translate"
    )
    text = result["text"]


    st.write("### 📝 Transcribed Text")
    st.write(text)
    text = preprocess_text(text)
    st.write("### Clean Text Used For Model")
    st.write(text)
    # TEXT FEATURES
    text_vec = vectorizer.transform([text]).toarray()
    text_features = text_feature_model.predict(text_vec)[0]
    text_pred = text_model.predict(text_vec)



    category = text_encoder.inverse_transform([text_pred.argmax()])
    # EMOTION FEATURES
    mel1 = librosa.feature.melspectrogram(
        y=audio,
        sr=sr,
        n_mels=128
    )
    mel1 = librosa.power_to_db(mel1)
    mel1 = librosa.util.fix_length(mel1, size=200, axis=1)
    mel1 = mel1.reshape(1, 128, 200, 1)

    emotion_features = emotion_feature_model.predict(mel1)[0]

    # SOUND FEATURES
    mel = librosa.feature.melspectrogram(y=audio, sr=sr, n_mels=128)
    mel = librosa.power_to_db(mel)
    mel = librosa.util.fix_length(mel, size=174, axis=1)
    mel = mel.reshape(1,128,174,1)

    sound_features = sound_feature_model.predict(mel)[0]

    # FUSION
    fusion_input = np.concatenate(
        [
            text_features * 2.0,
            emotion_features * 0.7,
            sound_features * 0.7
        ]
    ).reshape(1,-1)

    pred = fusion_model.predict(fusion_input)
    print(pred)
    pred_class = np.argmax(pred)

    urgency = urgency_encoder.inverse_transform([pred_class])[0]

    st.write("## 🚨 Predicted Urgency")
    st.success(urgency)
    st.write("### 🚨 Detected Incident Type")
    st.warning(category)
    for i, label in enumerate(urgency_encoder.classes_):
        st.write(label, float(pred[0][i]))



# ----------------------------
# AUDIO OPTIONS
# ----------------------------

option = st.radio(
    "Choose Input Method",
    ["Upload Audio File", "Record Audio"]
)

# ----------------------------
# Upload Audio
# ----------------------------

if option == "Upload Audio File":

    uploaded_file = st.file_uploader(
        "Upload Emergency Call",
        type=["wav","mp3"]
    )

    if uploaded_file:

        st.audio(uploaded_file)

        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
            tmp.write(uploaded_file.read())
            audio_path = tmp.name

        analyze_audio(audio_path)


# ----------------------------
# Record Audio
# ----------------------------

if option == "Record Audio":

    audio_record = st.audio_input("Record Emergency Call")

    if audio_record:

        st.audio(audio_record)

        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
            tmp.write(audio_record.read())
            audio_path = tmp.name

        analyze_audio(audio_path)