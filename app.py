import streamlit as st
import pickle
import pandas as pd
from joblib import load
import torch
from bertopic import BERTopic



@st.cache_resource
def load_classification_model():
    tfidf = load(open("models/vectorizer_tfidf.pkl", "rb"))
    clf = load(open("models/model_logisticregression.pkl", "rb"))
    return tfidf, clf

@st.cache_resource
def load_topic_model():
    from bertopic import BERTopic
    model = BERTopic.load("models\Bertopic_model-20251120T112845Z-1-001\Bertopic_model")
    return model

tfidf, classifier = load_classification_model()
topic_model = load_topic_model()


# --------------------------
# 2) TITRE DU DASHBOARD
# --------------------------
st.title("📊 Dashboard d'analyse des avis clients")
st.write("Classification + Extraction de thématiques")


# --------------------------
# 3) BARRE LATERALE (OPTIONS)
# --------------------------

st.sidebar.header("🛠️ Options")

mode = st.sidebar.selectbox(
    "Choisir une analyse",
    ["Classification", "Thématiques"]
)

user_text = st.sidebar.text_area("✍️ Entrez un avis à analyser")

btn = st.sidebar.button("Analyser")


# --------------------------
# 4) ANALYSE : CLASSIFICATION
# --------------------------

if btn and user_text and mode == "Classification":
    st.subheader("🔍 Résultat de la classification")

    # Transformer le texte avec TF-IDF
    vec = tfidf.transform([user_text])

    # Prédiction
    pred = classifier.predict(vec)[0]
    proba = classifier.predict_proba(vec).max()

    # Affichage
    st.write(f"**Classe prédite :** {pred}")
    st.write(f"**Confiance :** {proba:.2f}")


# --------------------------
# 5) ANALYSE : THEMES (BERTopic)
# --------------------------

if btn and user_text and mode == "Thématiques":
    st.subheader("🧠 Thème dominant")

    topic, probas = topic_model.transform([user_text])

    st.write(f"**Thème prédit :** {topic[0]}")
    st.write(f"**Probabilité :** {probas[0].max():.2f}")

    # Mots-clés du thème
    st.subheader("🔑 Mots-clés du thème")
    keywords = topic_model.get_topic(topic[0])
    st.write(keywords)

