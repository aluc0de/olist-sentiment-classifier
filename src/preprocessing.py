import re
import pandas as pd

def create_sentiment(score):
    # Mapeia a nota de 1 a 5 para as categorias de sentimento.
    if score >= 4:
        return 'Positive'
    elif score == 3:
        return 'Neutral'
    else:
        return 'Negative'

def clean_text(text):
    # Limpa o texto removendo URLs, pontuação e caracteres especiais.
    text = str(text).lower()
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    text = re.sub(r'[^a-zà-ÿ\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def load_and_prepare_data(filepath):
    # Carrega o CSV do Olist, limpa e retorna o DataFrame processado.
    df = pd.read_csv(filepath)
    df_clean = df.dropna(subset=['review_comment_message']).copy()
    df_clean['sentiment'] = df_clean['review_score'].apply(create_sentiment)
    df_clean['clean_text'] = df_clean['review_comment_message'].apply(clean_text)
    df_clean = df_clean.drop_duplicates(subset=['clean_text']).copy()
    return df_clean