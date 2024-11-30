from abc import ABC, abstractmethod

from torch import nn


class ActionRecognitionModel(ABC):
    @abstractmethod
    def create_model(self, num_classes: int) -> nn.Module:
        pass
