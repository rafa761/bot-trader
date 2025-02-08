# sentiment_analysis.py

import nltk
import numpy as np
import requests
from nltk.sentiment.vader import SentimentIntensityAnalyzer

from config import NEWS_API_KEY, SENTIMENT_ANALYSIS_ENABLED
from logger import logger

# Verifica se o 'vader_lexicon' está instalado. Se não estiver, faz o download.
try:
    nltk.data.find('vader_lexicon')
except LookupError:
    nltk.download('vader_lexicon')


def get_market_sentiment(query='bitcoin') -> float:
    """
    Busca notícias via NewsAPI.org e calcula o sentimento médio (entre -1 e 1).
    Retorna 0.5 caso haja falhas ou se a análise estiver desativada.
    """
    if not SENTIMENT_ANALYSIS_ENABLED or not NEWS_API_KEY:
        # Se não estiver habilitado ou não houver chave, retornamos neutro (0.5)
        return 0.5

    try:
        url = f"https://newsapi.org/v2/everything?q={query}&apiKey={NEWS_API_KEY}"
        response = requests.get(url).json()

        articles = response.get('articles', [])
        if not articles:
            return 0.5  # Se não houver artigos, retornamos neutro

        # Extrair títulos (ou descrições) e calcular sentimento
        sia = SentimentIntensityAnalyzer()
        titles = [article['title'] for article in articles if 'title' in article]
        sentiment_scores = [sia.polarity_scores(title)['compound'] for title in titles]

        # Retorna média dos compound scores
        return float(np.mean(sentiment_scores)) if sentiment_scores else 0.5
    except Exception as e:
        logger.error(f"Erro ao obter sentimento do mercado: {e}", exc_info=True)
        return 0.5
