# services/model_retrainer.py

import asyncio
import threading
import time
from collections.abc import Callable
from datetime import datetime, timedelta

import numpy as np
import pandas as pd

from core.constants import FEATURE_COLUMNS, TRAINED_MODELS_DIR, TRAINED_MODELS_BACKUP_DIR, TRAINED_MODELS_TEMP_DIR
from core.logger import logger
from models.lstm.model import LSTMModel
from models.lstm.schemas import LSTMConfig, LSTMTrainingConfig
from models.lstm.trainer import LSTMTrainer
from models.managers.model_manager import ModelManager
from repositories.data_handler import LabelCreator
from repositories.data_preprocessor import DataPreprocessor


class ModelRetrainer:
    """
    Classe responsável pelo retreinamento periódico de modelos LSTM.

    Executa em uma thread separada para não interromper as operações de trading,
    permitindo que os modelos sejam atualizados com novos dados de mercado conforme
    o tempo passa.
    """

    def __init__(
            self,
            tp_model: LSTMModel,
            sl_model: LSTMModel,
            get_data_callback: Callable[[], pd.DataFrame],
            signal_generator_ref: Callable = None,
            retraining_interval_hours: int = 24,
            min_data_points: int = 1000,
            performance_threshold: float = 0.15
    ):
        """
        Inicializa o sistema de retreinamento.

        Args:
            tp_model: Modelo LSTM para previsão de take profit
            sl_model: Modelo LSTM para previsão de stop loss
            get_data_callback: Função de callback para obter dados históricos atualizados
            signal_generator_ref: Função para obter referência atualizada ao signal_generator
            retraining_interval_hours: Intervalo mínimo entre retreinamentos (em horas)
            min_data_points: Número mínimo de pontos de dados necessários para retreinar
            performance_threshold: Limiar de erro acima do qual o retreinamento é forçado
        """
        self.tp_model = tp_model
        self.sl_model = sl_model
        self.get_data_callback = get_data_callback
        self.signal_generator_ref = signal_generator_ref
        self.retraining_interval_hours = retraining_interval_hours
        self.min_data_points = min_data_points
        self.performance_threshold = performance_threshold

        # Controle interno
        self.last_retraining_time = datetime.now() - timedelta(hours=retraining_interval_hours - 1)
        self.retraining_in_progress = False
        self.retraining_thread = None
        self.recent_prediction_errors = []
        self.max_error_history = 100
        self.lock = threading.Lock()
        self.models_updated = threading.Event()

        # Caminhos para os modelos
        self.tp_model_path = TRAINED_MODELS_DIR / f"{tp_model.config.model_name}.keras"
        self.sl_model_path = TRAINED_MODELS_DIR / f"{sl_model.config.model_name}.keras"

        # Contagem de ciclos para verificação de retreinamento
        self.cycles_since_last_check = 0
        self.check_interval_cycles = 12  # Verificar frequentemente (60s com sleep de 5s)

        # Métricas para avaliar necessidade de retreinamento
        self.tp_errors = []
        self.sl_errors = []
        self.error_history_size = 50

        logger.info("Sistema de retreinamento de modelos inicializado")

    def start(self):
        """Inicia o thread de monitoramento para retreinamento."""
        if self.retraining_thread is None or not self.retraining_thread.is_alive():
            self.retraining_thread = threading.Thread(
                target=self._monitoring_loop,
                daemon=True
            )
            self.retraining_thread.start()
            logger.info("Thread de monitoramento para retreinamento iniciada")

    def _monitoring_loop(self):
        """Loop principal de monitoramento executado em uma thread separada."""
        logger.info("Loop de monitoramento de retreinamento iniciado")
        while True:
            try:
                # Verificar condições para retreinamento
                if self._should_retrain():
                    logger.info("Critérios para retreinamento atendidos - Iniciando processo")
                    # Executar retreinamento em uma thread separada para não bloquear o monitoramento
                    retrain_thread = threading.Thread(
                        target=self._perform_retraining
                    )
                    retrain_thread.start()

                    # Aguardar conclusão do retreinamento antes de continuar monitoramento
                    retrain_thread.join()
                else:
                    # Log menos verboso para evitar spam
                    self.cycles_since_last_check += 1
                    if self.cycles_since_last_check >= self.check_interval_cycles:
                        logger.debug("Verificação de retreinamento: critérios não atendidos")
                        self.cycles_since_last_check = 0

                # Dormir entre verificações para não consumir recursos
                time.sleep(5)  # Verificar a cada 5 segundos

            except Exception as e:
                logger.error(f"Erro no loop de monitoramento de retreinamento: {e}", exc_info=True)
                time.sleep(60)  # Tempo maior em caso de erro para evitar ciclo de erros

    def _should_retrain(self) -> bool:
        """
        Determina se os modelos devem ser retreinados com base em múltiplos critérios.

        Returns:
            bool: True se o retreinamento deve ser realizado, False caso contrário
        """
        # Evitar verificações desnecessárias a cada ciclo
        self.cycles_since_last_check += 1
        if self.cycles_since_last_check < self.check_interval_cycles:
            return False

        self.cycles_since_last_check = 0

        # Não retreinar se já estiver em andamento
        if self.retraining_in_progress:
            return False

        current_time = datetime.now()

        # Critério 1: Tempo desde o último retreinamento
        time_criterion = (
                current_time - self.last_retraining_time
                > timedelta(hours=self.retraining_interval_hours)
        )

        # Critério 2: Quantidade de dados disponíveis
        data = self.get_data_callback()
        data_criterion = len(data) >= self.min_data_points

        # Critério 3: Desempenho recente do modelo
        performance_criterion = self._evaluate_model_performance()

        # Log dos critérios para debug
        logger.info(
            f"Avaliação para retreinamento: "
            f"Tempo({time_criterion}), "
            f"Dados({data_criterion}: {len(data) if data is not None else 0}), "
            f"Desempenho({performance_criterion})"
        )

        # Precisamos atender ao critério de tempo E dados, OU ao critério de desempenho
        should_retrain = (time_criterion and data_criterion) or performance_criterion

        if should_retrain:
            logger.info("Critérios para retreinamento atendidos. Preparando processo.")

        return should_retrain

    def _evaluate_model_performance(self) -> bool:
        """
        Avalia se o desempenho recente do modelo justifica retreinamento.

        Returns:
            bool: True se o desempenho estiver abaixo do limiar aceitável
        """
        if not self.recent_prediction_errors:
            return False

        # Calcular erro médio recente
        mean_error = np.mean(self.recent_prediction_errors)

        # Se houver pelo menos 20 registros de erros e o erro médio for alto, retreinar
        if len(self.recent_prediction_errors) >= 20 and mean_error > self.performance_threshold:
            logger.warning(
                f"Desempenho do modelo abaixo do aceitável: erro médio={mean_error:.4f}, "
                f"threshold={self.performance_threshold}"
            )
            return True

        return False

    def record_prediction_error(self, predicted: float, actual: float, error_type: str = "combined"):
        """
        Registra o erro entre uma previsão e o valor real observado.
        Esta função deve ser chamada pelo bot quando os resultados reais dos trades forem conhecidos.

        Args:
            predicted: Valor percentual previsto pelo modelo
            actual: Valor percentual real observado no mercado
            error_type: Tipo de erro - "tp", "sl" ou "combined"
        """
        if predicted is None or actual is None:
            return

        with self.lock:
            # Calcular erro relativo absoluto
            error = abs(predicted - actual) / (abs(actual) if abs(actual) > 0.1 else 0.1)

            # Armazenar separadamente erros de TP e SL para análise específica
            if error_type == "tp":
                self.tp_errors.append(error)
                if len(self.tp_errors) > self.error_history_size:
                    self.tp_errors.pop(0)
                logger.debug(f"Erro de TP registrado: {error:.4f} (previsto={predicted:.2f}%, real={actual:.2f}%)")
            elif error_type == "sl":
                self.sl_errors.append(error)
                if len(self.sl_errors) > self.error_history_size:
                    self.sl_errors.pop(0)
                logger.debug(f"Erro de SL registrado: {error:.4f} (previsto={predicted:.2f}%, real={actual:.2f}%)")

            # Adicionar ao histórico de erros combinados
            self.recent_prediction_errors.append(error)

            # Manter o tamanho do histórico limitado
            if len(self.recent_prediction_errors) > self.max_error_history:
                self.recent_prediction_errors.pop(0)

            # Log detalhado para monitoramento
            if len(self.recent_prediction_errors) % 10 == 0:  # A cada 10 erros, mostrar média
                avg_error = np.mean(self.recent_prediction_errors)
                logger.info(
                    f"Média dos últimos {len(self.recent_prediction_errors)} erros de previsão: {avg_error:.4f}"
                )

    def _apply_transfer_learning(self, new_model: LSTMModel, base_model: LSTMModel,
                                 training_phase: str = "initial") -> bool:
        """
        Aplica transfer learning de forma avançada com congelamento seletivo de camadas e
        técnicas de fine-tuning para otimizar o treinamento incremental.

        Args:
            new_model: Modelo novo que receberá os pesos
            base_model: Modelo base de onde os pesos serão transferidos
            training_phase: Fase de treinamento ('initial', 'fine_tuning')

        Returns:
            bool: True se o transfer learning foi aplicado com sucesso
        """
        try:
            if not hasattr(base_model, 'model') or base_model.model is None:
                logger.warning("Modelo base não possui atributo 'model' ou é None")
                return False

            if not hasattr(new_model, 'model') or new_model.model is None:
                logger.warning("Modelo novo não possui atributo 'model' ou é None")
                return False

            # Verificar compatibilidade de arquitetura
            base_weights = base_model.model.get_weights()
            new_weights = new_model.model.get_weights()

            if len(base_weights) != len(new_weights):
                logger.warning(
                    f"Incompatibilidade na estrutura dos modelos. "
                    f"Base: {len(base_weights)} camadas, Novo: {len(new_weights)} camadas"
                )
                return False

            # Verificar forma dos tensores de pesos para cada camada
            compatible = True
            for i, (base_w, new_w) in enumerate(zip(base_weights, new_weights)):
                if base_w.shape != new_w.shape:
                    logger.warning(
                        f"Incompatibilidade na camada {i}: "
                        f"Base: {base_w.shape}, Novo: {new_w.shape}"
                    )
                    compatible = False
                    break

            if not compatible:
                logger.warning("Arquiteturas incompatíveis. Transfer learning não aplicado.")
                return False

            # Aplicar pesos do modelo base ao novo modelo
            logger.info("Transferindo pesos do modelo base para o novo modelo...")
            new_model.model.set_weights(base_weights)

            # Congelamento seletivo de camadas com base na fase de treinamento
            if training_phase == "initial":
                # Fase inicial: congelar camadas LSTM (preservar conhecimento extraído)
                layer_count = 0
                for layer in new_model.model.layers:
                    if 'lstm' in layer.name.lower():
                        layer.trainable = False
                        layer_count += 1
                        logger.info(f"Congelada camada LSTM: {layer.name}")

                logger.info(f"Total de {layer_count} camadas LSTM congeladas para fase inicial")

            elif training_phase == "fine_tuning":
                # Fase de fine-tuning: descongelar todas as camadas para ajuste fino
                for layer in new_model.model.layers:
                    layer.trainable = True

                logger.info(f"Todas as camadas descongeladas para fase de fine-tuning")

            # Ajustar taxa de aprendizado baseado na fase
            if hasattr(new_model.model, 'optimizer'):
                import tensorflow as tf

                if training_phase == "initial":
                    # Taxa reduzida para phase inicial (apenas ajustar camadas finais)
                    lr = base_model.config.learning_rate * 0.4
                else:
                    # Taxa muito reduzida para fine-tuning (ajuste suave de todas as camadas)
                    lr = base_model.config.learning_rate * 0.1

                # Aplicar nova taxa de aprendizado
                if isinstance(new_model.model.optimizer, tf.keras.optimizers.Adam):
                    tf.keras.backend.set_value(new_model.model.optimizer.learning_rate, lr)
                    logger.info(f"Taxa de aprendizado ajustada para {lr}")

            # Armazenar métricas do modelo base para comparação posterior
            if not hasattr(new_model, 'base_metrics'):
                new_model.base_metrics = {}
                if hasattr(base_model, 'last_metrics'):
                    new_model.base_metrics = base_model.last_metrics.copy()

            logger.info("Transfer learning aplicado com sucesso!")
            return True

        except Exception as e:
            logger.error(f"Erro ao aplicar transfer learning: {e}", exc_info=True)
            return False

    def _perform_retraining(self):
        """Executa o processo completo de retreinamento dos modelos."""
        if self.retraining_in_progress:
            logger.warning("Tentativa de retreinamento enquanto já em andamento. Ignorando.")
            return

        try:
            with self.lock:
                self.retraining_in_progress = True
                # Limpar flag de modelos atualizados
                self.models_updated.clear()

            logger.info("Iniciando processo de retreinamento...")

            # 1. Obter dados atualizados
            df = self.get_data_callback()
            if df is None or df.empty:
                logger.error("Não foi possível obter dados para retreinamento")
                return

            if df is not None and not df.empty:
                # Usar apenas os últimos N dados mais recentes para retreinamento incremental
                max_samples = 1000
                if len(df) > max_samples:
                    logger.info(f"Limitando a {max_samples} amostras mais recentes para retreinamento incremental")
                    df = df.tail(max_samples)

            # 2. Criar labels para treinamento (TP e SL)
            df = LabelCreator.create_labels(df)

            if df.empty:
                logger.error("Falha ao criar labels para retreinamento")
                return

            # 3. Preprocessar os dados
            preprocessor = DataPreprocessor(
                feature_columns=FEATURE_COLUMNS,
                outlier_method='iqr',
                scaling_method='robust'
            )
            preprocessor.fit(df)
            df_processed = preprocessor.process_dataframe(df)

            # Log de estatísticas dos dados
            logger.info(f"Dados para retreinamento: {len(df_processed)} registros após preprocessamento")
            if 'take_profit_pct' in df_processed.columns and 'stop_loss_pct' in df_processed.columns:
                tp_stats = df_processed['take_profit_pct'].describe()
                sl_stats = df_processed['stop_loss_pct'].describe()
                logger.info(
                    f"Estatísticas TP: média={tp_stats['mean']:.2f}%, min={tp_stats['min']:.2f}%, max={tp_stats['max']:.2f}%")
                logger.info(
                    f"Estatísticas SL: média={sl_stats['mean']:.2f}%, min={sl_stats['min']:.2f}%, max={sl_stats['max']:.2f}%")

            # 4. Retreinar para cada target (TP e SL)
            targets = ['take_profit_pct', 'stop_loss_pct']
            new_models = {}

            for target in targets:
                logger.info(f"Retreinando modelo para {target}...")

                # Configurar modelo baseado no existente
                if target == 'take_profit_pct':
                    base_model = self.tp_model
                else:
                    base_model = self.sl_model

                # Criar cópia da configuração e ajustar para retreinamento
                retraining_config = self._create_retraining_config(base_model.config)

                # Criar novo modelo para retreinamento
                model = LSTMModel(retraining_config)

                # Configurar trainer
                training_config = LSTMTrainingConfig(
                    validation_split=0.15,
                    early_stopping_patience=5,
                    reduce_lr_patience=2,
                    reduce_lr_factor=0.6,
                    use_early_stopping=True,
                    min_delta=0.0008,
                    test_size=0.2,
                    random_state=42,
                    shuffle=False
                )
                trainer = LSTMTrainer(model, training_config)

                # Criar gerenciador para executar pipeline de treinamento
                manager = ModelManager(model, trainer, retraining_config)

                # Inicializar novo modelo com pesos do anterior (transfer learning)
                try:
                    # Fase 1: Treinamento inicial com camadas LSTM congeladas
                    transfer_success = self._apply_transfer_learning(
                        model,
                        base_model,
                        training_phase="initial"
                    )

                    if transfer_success:
                        # Treinamento inicial (apenas camadas finais são treinadas)
                        initial_metrics = manager.execute_full_pipeline(
                            data=df_processed,
                            feature_columns=FEATURE_COLUMNS,
                            target_column=target,
                            save_path=TRAINED_MODELS_TEMP_DIR,
                            epochs=10  # Menos épocas para primeira fase
                        )

                        logger.info(f"Fase inicial concluída com métricas: {initial_metrics}")

                        # Fase 2: Fine-tuning com todas as camadas descongeladas
                        self._apply_transfer_learning(
                            model,
                            base_model,
                            training_phase="fine_tuning"
                        )

                        # Treinamento completo para ajuste fino
                        metrics = manager.execute_full_pipeline(
                            data=df_processed,
                            feature_columns=FEATURE_COLUMNS,
                            target_column=target,
                            save_path=TRAINED_MODELS_TEMP_DIR
                        )

                        logger.info(f"Fase de fine-tuning concluída com métricas: {metrics}")
                    else:
                        # Se transfer learning falhar, treinamene normal
                        logger.warning("Transfer learning falhou, executando treinamento normal")
                        metrics = manager.execute_full_pipeline(
                            data=df_processed,
                            feature_columns=FEATURE_COLUMNS,
                            target_column=target,
                            save_path=TRAINED_MODELS_TEMP_DIR
                        )
                except Exception as e:
                    logger.error(f"Erro durante processo de transfer learning: {e}", exc_info=True)
                    # Treinamento normal como fallback
                    metrics = manager.execute_full_pipeline(
                        data=df_processed,
                        feature_columns=FEATURE_COLUMNS,
                        target_column=target,
                        save_path=TRAINED_MODELS_TEMP_DIR
                    )

            # 5. Substituir modelos antigos pelos novos
            self._replace_models(new_models)

            # Atualizar tempo do último retreinamento
            self.last_retraining_time = datetime.now()

            # Limpar histórico de erros após retreinamento
            self.recent_prediction_errors = []
            self.tp_errors = []
            self.sl_errors = []

            # Sinalizar que os modelos foram atualizados
            self.models_updated.set()

            logger.info("Processo de retreinamento concluído com sucesso.")

        except Exception as e:
            logger.error(f"Erro durante processo de retreinamento: {e}", exc_info=True)
        finally:
            with self.lock:
                self.retraining_in_progress = False

    def _create_retraining_config(self, base_config: LSTMConfig) -> LSTMConfig:
        """
        Cria uma nova configuração para retreinamento baseada na configuração existente.

        Args:
            base_config: Configuração do modelo base

        Returns:
            LSTMConfig: Nova configuração otimizada para retreinamento
        """
        # Criar cópia da configuração
        version_parts = base_config.version.split('.')
        new_version = f"{version_parts[0]}.{version_parts[1]}.{int(version_parts[2]) + 1}"

        # Configuração otimizada para retreinamento mais rápido
        retraining_config = LSTMConfig(
            model_name=base_config.model_name,
            version=new_version,
            description=f"{base_config.description} (Retreinado em {datetime.now().strftime('%Y-%m-%d')})",
            sequence_length=base_config.sequence_length,
            lstm_units=base_config.lstm_units,
            dense_units=base_config.dense_units,
            dropout_rate=base_config.dropout_rate,
            learning_rate=base_config.learning_rate * 0.8,  # Taxa de aprendizagem reduzida
            batch_size=base_config.batch_size * 2,  # Batch maior para retreinamento mais rápido
            epochs=25  # Menos épocas para retreinamento incremental
        )

        return retraining_config

    def _replace_models(self, new_models: dict):
        """
        Substitui os modelos antigos pelos novos retreinados.

        Args:
            new_models: Dicionário com novos modelos e suas métricas
        """
        should_update_tp = True
        should_update_sl = True

        with self.lock:
            try:
                # Flag para rastrear se algum modelo foi realmente atualizado
                models_updated = False

                # Substituir modelo de TP se disponível
                if 'take_profit_pct' in new_models:
                    tp_info = new_models['take_profit_pct']

                    if 'test_loss' in tp_info['metrics'] and hasattr(self.tp_model, 'last_metrics'):
                        if tp_info['metrics']['test_loss'] >= self.tp_model.last_metrics['test_loss'] * 0.95:
                            logger.warning(
                                f"Novo modelo TP não apresenta melhoria significativa. "
                                f"Loss atual: {self.tp_model.last_metrics['test_loss']}, "
                                f"novo: {tp_info['metrics']['test_loss']}. Mantendo modelo atual."
                            )
                            should_update_tp = False
                        else:
                            self.tp_model.last_metrics = tp_info['metrics']

                    if should_update_tp:
                        # Fazer backup do modelo antigo com timestamp
                        timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
                        backup_path = TRAINED_MODELS_BACKUP_DIR / f"{self.tp_model.config.model_name}_{timestamp}.keras"

                        if self.tp_model_path.exists():
                            try:
                                import shutil
                                shutil.copy2(self.tp_model_path, backup_path)
                                logger.info(f"Backup do modelo TP criado em {backup_path}")
                            except Exception as e:
                                logger.error(f"Erro ao criar backup do modelo TP: {e}")

                        # Mover novo modelo para localização padrão
                        try:
                            # Se o arquivo de destino já existir, removê-lo primeiro
                            if self.tp_model_path.exists():
                                self.tp_model_path.unlink()

                            # Copiar o novo modelo para o local correto
                            import shutil
                            shutil.copy2(tp_info['temp_path'], self.tp_model_path)

                            # Atualizar referência ao modelo
                            self.tp_model = tp_info['model']
                            models_updated = True

                            logger.info(f"Modelo TP atualizado para versão {self.tp_model.config.version}")
                        except Exception as e:
                            logger.error(f"Erro ao substituir modelo TP: {e}", exc_info=True)

                # Substituir modelo de SL se disponível
                if 'stop_loss_pct' in new_models:
                    sl_info = new_models['stop_loss_pct']

                    if 'test_loss' in sl_info['metrics'] and hasattr(self.sl_model, 'last_metrics'):
                        if sl_info['metrics']['test_loss'] >= self.sl_model.last_metrics['test_loss'] * 0.95:
                            logger.warning(
                                f"Novo modelo TP não apresenta melhoria significativa. "
                                f"Loss atual: {self.sl_model.last_metrics['test_loss']}, "
                                f"novo: {sl_info['metrics']['test_loss']}. Mantendo modelo atual."
                            )
                            should_update_sl = False  # alterar flag em vez de retornar
                        else:
                            self.sl_model.last_metrics = tp_info['metrics']

                    if should_update_sl:
                        # Fazer backup do modelo antigo com timestamp
                        timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
                        backup_path = TRAINED_MODELS_BACKUP_DIR / f"{self.sl_model.config.model_name}_{timestamp}.keras"

                        if self.sl_model_path.exists():
                            try:
                                import shutil
                                shutil.copy2(self.sl_model_path, backup_path)
                                logger.info(f"Backup do modelo SL criado em {backup_path}")
                            except Exception as e:
                                logger.error(f"Erro ao criar backup do modelo SL: {e}")

                        # Mover novo modelo para localização padrão
                        try:
                            # Se o arquivo de destino já existir, removê-lo primeiro
                            if self.sl_model_path.exists():
                                self.sl_model_path.unlink()

                            # Copiar o novo modelo para o local correto
                            import shutil
                            shutil.copy2(sl_info['temp_path'], self.sl_model_path)

                            # Atualizar referência ao modelo
                            self.sl_model = sl_info['model']
                            models_updated = True

                            logger.info(f"Modelo SL atualizado para versão {self.sl_model.config.version}")
                        except Exception as e:
                            logger.error(f"Erro ao substituir modelo SL: {e}", exc_info=True)

                # Atualizar o signal generator do bot, se fornecido
                if models_updated and self.signal_generator_ref:
                    try:
                        signal_generator = self.signal_generator_ref()
                        if signal_generator:
                            # Atualizar as referências dos modelos
                            signal_generator.tp_model = self.tp_model
                            signal_generator.sl_model = self.sl_model
                            logger.info("Referências do signal_generator atualizadas com os novos modelos")
                    except Exception as e:
                        logger.error(f"Erro ao atualizar signal_generator: {e}", exc_info=True)

                # Remover arquivos temporários
                for model_info in new_models.values():
                    if 'temp_path' in model_info and model_info['temp_path'].exists():
                        try:
                            model_info['temp_path'].unlink()
                        except Exception as e:
                            logger.error(f"Erro ao remover arquivo temporário: {e}")

            except Exception as e:
                logger.error(f"Erro ao substituir modelos: {e}", exc_info=True)

    async def wait_for_retraining(self, timeout_seconds: int = 1800):
        """
        Aguarda a conclusão de um retreinamento em andamento.

        Args:
            timeout_seconds: Tempo máximo de espera em segundos

        Returns:
            bool: True se o retreinamento foi concluído, False se timeout
        """
        start_time = time.time()

        while self.retraining_in_progress:
            await asyncio.sleep(5)

            # Verificar timeout
            if time.time() - start_time > timeout_seconds:
                logger.warning(f"Timeout aguardando retreinamento após {timeout_seconds} segundos")
                return False

        return True

    def get_retraining_status(self) -> dict:
        """
        Retorna o status atual do sistema de retreinamento.

        Returns:
            dict: Dicionário com informações sobre o status de retreinamento
        """
        with self.lock:
            # Calcular médias de erros se houver registros suficientes
            tp_error_mean = np.mean(self.tp_errors) if len(self.tp_errors) > 0 else 0
            sl_error_mean = np.mean(self.sl_errors) if len(self.sl_errors) > 0 else 0

            return {
                "retraining_in_progress": self.retraining_in_progress,
                "last_retraining_time": self.last_retraining_time.isoformat(),
                "hours_since_last_retraining": (datetime.now() - self.last_retraining_time).total_seconds() / 3600,
                "recent_error_count": len(self.recent_prediction_errors),
                "mean_error": np.mean(self.recent_prediction_errors) if self.recent_prediction_errors else 0,
                "tp_error_mean": tp_error_mean,
                "sl_error_mean": sl_error_mean,
                "tp_model_version": self.tp_model.config.version,
                "sl_model_version": self.sl_model.config.version,
                "next_check_in_cycles": self.check_interval_cycles - self.cycles_since_last_check,
                "models_updated_flag": self.models_updated.is_set()
            }
