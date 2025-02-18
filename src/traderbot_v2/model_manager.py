# model_manager.py

"""
Este módulo gerencia o carregamento, treinamento e inferência (predição) dos
modelos de Take-Profit (TP) e Stop-Loss (SL).
"""

import sys
from pathlib import Path

import joblib
import pandas as pd
from sklearn.pipeline import Pipeline

from constants import FEATURE_COLUMNS
from logger import logger

models_dir = Path("train_data")


class ModelManager:
    """
    Classe responsável por carregar e treinar os modelos, bem como gerar
    previsões de TP e SL.
    """

    def __init__(self, model_tp_path: str = models_dir / "model_tp.pkl",
                 model_sl_path: str = models_dir / "model_sl.pkl"):
        """
        Construtor que tenta carregar os modelos TP e SL de arquivos .pkl.
        Caso não existam, o programa é encerrado.

        :param model_tp_path: Caminho do arquivo pkl do modelo TP
        :param model_sl_path: Caminho do arquivo pkl do modelo SL
        """
        logger.info("Iniciando classe ModelManager")
        self.pipeline_tp: Pipeline | None = None
        self.pipeline_sl: Pipeline | None = None

        try:
            self.pipeline_tp = joblib.load(model_tp_path)
            self.pipeline_sl = joblib.load(model_sl_path)
            logger.info("Modelos TP e SL carregados com sucesso.")
            logger.info(f"Pipeline TP: {self.pipeline_tp} e Pipeline SL: {self.pipeline_sl}")
        except FileNotFoundError:
            logger.error("Modelos TP e SL não encontrados. Encerrando aplicação.")
            sys.exit(1)
        except Exception as e:
            logger.error(f"Erro ao carregar modelos: {e}", exc_info=True)
            sys.exit(1)

    def train_models(
            self,
            features: pd.DataFrame,
            target_tp: pd.Series,
            target_sl: pd.Series,
    ) -> None:
        """
        Realiza o re-treinamento (online) dos modelos de TP e SL.

        :param features: DataFrame com as variáveis preditoras (indicadores)
        :param target_tp: Série com valores de variação % futura para TP
        :param target_sl: Série com valores de variação % futura para SL
        """
        if self.pipeline_tp is not None and self.pipeline_sl is not None:
            try:
                self.pipeline_tp.fit(features, target_tp)
                self.pipeline_sl.fit(features, target_sl)
                logger.info("Modelos TP/SL re-treinados com sucesso.")
            except Exception as e:
                logger.error(f"Erro ao treinar modelos: {e}", exc_info=True)
        else:
            logger.warning("Pipelines TP ou SL não inicializados, não é possível treinar.")

    def predict_tp_sl(self, df_eval: pd.DataFrame) -> tuple[float, float]:
        """
        Gera previsões de variação percentual para TP e SL com base no DataFrame fornecido.

        :param df_eval: DataFrame com colunas [sma_short, sma_long, rsi, macd, boll_hband, boll_lband, atr]
        :return: (predicted_tp_pct, predicted_sl_pct)
        """
        required_features = FEATURE_COLUMNS
        missing = set(required_features) - set(df_eval.columns)
        if missing:
            logger.error(f"Features faltando: {missing}")
            return 0.0, 0.0

        if self.pipeline_tp is not None and self.pipeline_sl is not None:
            try:
                predicted_tp_pct = self.pipeline_tp.predict(df_eval)[0]
                predicted_sl_pct = self.pipeline_sl.predict(df_eval)[0]
                return predicted_tp_pct, predicted_sl_pct
            except Exception as e:
                logger.error(f"Erro ao executar predição TP/SL: {e}", exc_info=True)
                return 0.0, 0.0

        logger.warning("Pipelines não disponíveis para predição.")

        return 0.0, 0.0
