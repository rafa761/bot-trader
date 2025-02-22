from tensorflow.keras import Sequential

from models.base.model import BaseModel, ModelConfig


class LSTMConfig(ModelConfig):
    """Configuração específica para LSTM"""
    units: int
    dropout: float
    return_sequences: bool


class LSTMModel(BaseModel):
    def __init__(self, config: LSTMConfig):
        super().__init__(config)
        self.model = self.build_model()

    def build_model(self) -> Sequential:
        """Implementação concreta para LSTM"""
        model = Sequential()
        # Arquitetura LSTM aqui
        return model

    def predict(self, input_data):
        # Implementação específica
        return self.model.predict(input_data)
