# generate_report.py
import pandas as pd
import torch
from transformers import BertTokenizer, BertForSequenceClassification
import matplotlib.pyplot as plt
from wordcloud import WordCloud, STOPWORDS
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image, PageBreak
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet
from sklearn.feature_extraction.text import CountVectorizer
from pathlib import Path
import numpy as np
import re
from config import DATA_PATH, MODEL_DIR, OUTPUT_DIR
from utils import limpiar_texto, extraer_bigrams, frases_para_resumen

# ------------------------------
# Preparar directorios y datos
# ------------------------------
Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)
df = pd.read_excel(DATA_PATH)
model_path = Path(MODEL_DIR) / "fase2" if (Path(MODEL_DIR) / "fase2").exists() else Path(MODEL_DIR)
tokenizer = BertTokenizer.from_pretrained(str(model_path))
model = BertForSequenceClassification.from_pretrained(str(model_path))
model.eval()
restaurantes = df["restaurant"].unique().tolist()

doc = SimpleDocTemplate(f"{OUTPUT_DIR}/reporte_final_siguenza.pdf", pagesize=A4)
styles = getSampleStyleSheet()
story = []

stop_words = set(STOPWORDS)
stop_words.update(["restaurante","sigüenza","comida","plato","bueno","buen","bien"])

# ------------------------------
# Función de predicción
# ------------------------------
def predict_sentiment(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=256)
    with torch.no_grad():
        outputs = model(**inputs)
        probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
    pred = torch.argmax(probs, dim=1).item()
    confidence = probs[0][pred].item()
    return pred, confidence

# ------------------------------
# Generar report por restaurante
# ------------------------------
for restaurant_name in restaurantes:
    df_rest = df[df["restaurant"]==restaurant_name].copy()
    if df_rest.empty: 
        continue

    # Predicciones
    results = df_rest["text"].apply(predict_sentiment)
    df_rest["pred"] = results.apply(lambda x: x[0])
    df_rest["confianza"] = results.apply(lambda x: x[1])
    df_rest["sentiment"] = df_rest["pred"].map({0:"no positiva",1:"positiva"})

    total = len(df_rest)
    positivas = sum(df_rest["sentiment"]=="positiva")
    no_positivas = sum(df_rest["sentiment"]=="no positiva")

    # Pie chart
    plt.figure()
    plt.pie([positivas,no_positivas], labels=["Positivas","No positivas"], autopct="%1.1f%%", colors=["green","red"])
    pie_path = Path(OUTPUT_DIR) / f"pie_{restaurant_name}.png"
    plt.savefig(pie_path)
    plt.close()

    # Wordclouds
    positives_text = " ".join(df_rest[df_rest["sentiment"]=="positiva"]["text"].apply(limpiar_texto))
    pos_path = None
    if positives_text.strip():
        wc_pos = WordCloud(width=800, height=400, background_color="white").generate(positives_text)
        pos_path = Path(OUTPUT_DIR) / f"wc_pos_{restaurant_name}.png"
        wc_pos.to_file(pos_path)

    negatives_text = " ".join(df_rest[df_rest["sentiment"]=="no positiva"]["text"].apply(limpiar_texto))
    neg_path = None
    if negatives_text.strip():
        wc_neg = WordCloud(width=800, height=400, background_color="white").generate(negatives_text)
        neg_path = Path(OUTPUT_DIR) / f"wc_neg_{restaurant_name}.png"
        wc_neg.to_file(neg_path)

    # Frases clave
    top_pos_bigrams = extraer_bigrams(df_rest[df_rest["sentiment"]=="positiva"]["text"].apply(limpiar_texto))
    top_neg_bigrams = extraer_bigrams(df_rest[df_rest["sentiment"]=="no positiva"]["text"].apply(limpiar_texto))
    fortalezas = frases_para_resumen(top_pos_bigrams)
    problemas = frases_para_resumen(top_neg_bigrams)

    # -----------------------------
    # Sección PDF
    # -----------------------------
    story.append(Paragraph(f"------ {restaurant_name} ------", styles["Title"]))
    story.append(Spacer(1,12))
    story.append(Paragraph(f"Total reseñas analizadas: {total}", styles["Normal"]))
    story.append(Paragraph(f"Positivas: {positivas}", styles["Normal"]))
    story.append(Paragraph(f"No positivas: {no_positivas}", styles["Normal"]))
    story.append(Spacer(1,12))

    story.append(Paragraph("Distribución de Sentimientos:", styles["Heading2"]))
    story.append(Image(str(pie_path), width=400, height=300))
    story.append(Spacer(1,12))

    if pos_path:
        story.append(Paragraph("Palabras más repetidas en elogios:", styles["Heading2"]))
        story.append(Image(str(pos_path), width=400, height=200))
        story.append(Spacer(1,12))

    if neg_path:
        story.append(Paragraph("Palabras más repetidas en críticas:", styles["Heading2"]))
        story.append(Image(str(neg_path), width=400, height=200))
        story.append(Spacer(1,12))

    story.append(PageBreak())

# Generar PDF
doc.build(story)
print(f"PDF generado correctamente: {OUTPUT_DIR}/reporte_final_siguenza.pdf ✅")
