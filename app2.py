import streamlit as st
import pickle
import pandas as pd
from joblib import load
import torch
from bertopic import BERTopic
import plotly.express as px
from wordcloud import WordCloud
import matplotlib.pyplot as plt

# --------------------------
# 1) CHARGEMENT DES MODELES
# --------------------------
#Chargement du modle de régression Logistique et du vectoriseur TF-IDF
@st.cache_resource
def load_classification_model():
    tfidf = load(open("models/vectorizer_tfidf.pkl", "rb"))
    clf = load(open("models/model_logisticregression.pkl", "rb"))
    return tfidf, clf

#Chargemet du modle BERTopic
@st.cache_resource
def load_topic_model():
    model = BERTopic.load("models\\Bertopic_model-20251120T112845Z-1-001\\Bertopic_model")
    return model

tfidf, classifier = load_classification_model()
topic_model = load_topic_model()

# --------------------------
# 2) TITRE DU DASHBOARD
# --------------------------
st.markdown("<h1 style='text-align: center; color: #4CAF50;'>📊 Dashboard d'analyse des avis clients</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center;'>Classification + Extraction de thématiques</p>", unsafe_allow_html=True)

# --------------------------
# 3) BARRE LATERALE
# --------------------------
st.sidebar.header("🛠️ Options")
mode = st.sidebar.selectbox("Choisir une analyse", ["Classification", "Thématiques"])
user_text = st.sidebar.text_area("✍️ Entrez un avis à analyser")
btn = st.sidebar.button("Analyser")

# --------------------------
# 4) ANALYSE : CLASSIFICATION
# --------------------------
if btn and user_text and mode == "Classification":
    st.markdown("<h2 style='color: #2196F3;'>🔍 Résultat de la classification</h2>", unsafe_allow_html=True)
    
    vec = tfidf.transform([user_text])
    pred = classifier.predict(vec)[0]
    proba = classifier.predict_proba(vec).max()
    
    # Colonnes pour une mise en page sympa
    col1, col2 = st.columns(2)
    with col1:
        st.markdown(f"<p style='background-color:#e3f2fd; padding:10px; border-radius:5px;'><b>Classe prédite :</b> {pred}</p>", unsafe_allow_html=True)
    with col2:
        st.markdown(f"<p style='background-color:#e3f2fd; padding:10px; border-radius:5px;'><b>Confiance :</b> {proba:.2f}</p>", unsafe_allow_html=True)
    
    # Graphique interactif avec toutes les probabilités
    prob_df = pd.DataFrame({
        "Classe": classifier.classes_,
        "Probabilité": classifier.predict_proba(vec)[0]
    })
    fig = px.bar(prob_df, x="Classe", y="Probabilité", color="Probabilité", text="Probabilité", color_continuous_scale="Blues")
    st.plotly_chart(fig, use_container_width=True)

# --------------------------
# 5) ANALYSE : THEMES (BERTopic)
# --------------------------
if btn and user_text and mode == "Thématiques":
    st.markdown("<h2 style='color: #FF9800;'>🧠 Thème dominant</h2>", unsafe_allow_html=True)
    
    topic, probas = topic_model.transform([user_text])
    st.markdown(f"<p style='background-color:#fff3e0; padding:10px; border-radius:5px;'><b>Thème prédit :</b> {topic[0]}</p>", unsafe_allow_html=True)
    st.markdown(f"<p style='background-color:#fff3e0; padding:10px; border-radius:5px;'><b>Probabilité :</b> {probas[0].max():.2f}</p>", unsafe_allow_html=True)
    
    # Top mots-clés
    st.subheader("🔑 Mots-clés du thème")
    keywords = topic_model.get_topic(topic[0])
    keywords_df = pd.DataFrame(keywords, columns=["Mot", "Score"])
    
    # Graphique interactif des mots-clés
    fig2 = px.bar(keywords_df, x="Mot", y="Score", text="Score", color="Score", color_continuous_scale="Oranges")
    st.plotly_chart(fig2, use_container_width=True)
    
    # Nuage de mots
    st.subheader("☁️ Nuage de mots")
    wordcloud = WordCloud(width=800, height=400, background_color="white").generate_from_frequencies(dict(keywords))
    plt.figure(figsize=(10,5))
    plt.imshow(wordcloud, interpolation="bilinear")
    plt.axis("off")
    st.pyplot(plt)

st.sidebar.subheader("📂 Analyse de fichiers")
uploaded_file = st.sidebar.file_uploader("Charger un fichier CSV ou TXT", type=["csv","txt"])

# Bouton pour lancer l'analyse du fichier
analyze_file_btn = st.sidebar.button("Analyser le fichier")

def get_theme_name(topic_id, model, top_n=3):
    """Renvoie un nom de thème lisible à partir des top mots-clés"""
    if topic_id == -1:  # thème outlier/noise
        return "Autres / Inconnu"
    keywords = model.get_topic(topic_id)
    top_words = [w for w, _ in keywords[:top_n]]  # prendre les top_n mots
    return ", ".join(top_words)

if uploaded_file is not None:
    st.sidebar.success("Fichier chargé avec succès !")
    
    # Lecture du fichier
    if uploaded_file.type == "text/csv":
        df = pd.read_csv(uploaded_file)
        comments = df["comment"].tolist() if "comment" in df.columns else df.iloc[:,0].tolist()
    elif uploaded_file.type == "text/plain":
        df = pd.read_csv(uploaded_file, header=None, names=["text"])
        comments = df["text"].tolist()
    
    st.sidebar.write(f"Nombre de lignes dans le fichier : {len(df)}")
    
    # Nettoyage pour éviter les erreurs
    comments = [str(c) for c in comments if str(c).strip() != ""]
    
    # Lancer l'analyse seulement si l'utilisateur clique sur le bouton
    if analyze_file_btn:
        if mode == "Classification":
            vecs = tfidf.transform(comments)
            preds = classifier.predict(vecs)
            probs = classifier.predict_proba(vecs).max(axis=1)
            
            df_results = pd.DataFrame({
                "Commentaire": comments,
                "Classe": preds,
                "Confiance": probs
            })
            
            st.subheader("📊 Résultats de la classification")
            st.dataframe(df_results.head(20))
            
            st.download_button(
                "📥 Télécharger les résultats CSV",
                df_results.to_csv(index=False),
                file_name="resultats_classification.csv"
            )
        
        elif mode == "Thématiques":
            topics, topic_probs = topic_model.transform(comments)
            
            # Conversion des IDs en noms lisibles
            theme_names = [get_theme_name(t, topic_model) for t in topics]
            
            df_topics = pd.DataFrame({
                "Commentaire": comments,
                "Thème": theme_names,
                "Probabilité": [p.max() for p in topic_probs]
            })
            
            st.subheader("🧠 Résultats des thèmes")
            st.dataframe(df_topics.head(20))
            
            # Top thèmes
            top_themes = pd.Series(theme_names).value_counts().reset_index()
            top_themes.columns = ["Thème", "Nombre de commentaires"]
            st.bar_chart(top_themes.set_index("Thème"))
            
            st.download_button(
                "📥 Télécharger les résultats CSV",
                df_topics.to_csv(index=False),
                file_name="resultats_themes.csv"
            )


