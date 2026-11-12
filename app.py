import gradio as gr
from transformers import pipeline

# Carrega o modelo do Hugging Face
classifier = pipeline("sentiment-analysis", model="aluc0de/bertimbau-olist-sentiment")

def predict_sentiment(text):
    if not text.strip():
        return "Por favor, digite um texto.", ""
    
    result = classifier(text)[0]
    # Mapeia as labels numéricas
    label_map = {"LABEL_0": "Negative", "LABEL_1": "Neutral", "LABEL_2": "Positive"}
    sentiment = label_map.get(result['label'], result['label'])
    confidence = f"{result['score']:.2%}"
    
    return sentiment, confidence

# Cria a interface
iface = gr.Interface(
    fn=predict_sentiment,
    inputs=gr.Textbox(lines=3, placeholder="Digite um comentário em português..."),
    outputs=[gr.Textbox(label="Sentimento"), gr.Textbox(label="Confiança")],
    title="Classificador de Sentimentos - Olist",
    description="Analise se um comentário de e-commerce é Positivo, Neutro ou Negativo."
)

# Gera um link temporário
iface.launch(share=True)