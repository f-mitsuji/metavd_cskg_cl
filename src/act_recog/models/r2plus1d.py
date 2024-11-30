from torch import nn
from torchvision.models.video import R2Plus1D_18_Weights, r2plus1d_18

from src.act_recog.models.base import ActionRecognitionModel


class R2Plus1DModel(ActionRecognitionModel):
    def create_model(self, num_classes: int) -> nn.Module:
        model = r2plus1d_18(weights=R2Plus1D_18_Weights.KINETICS400_V1)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
        nn.init.normal_(model.fc.weight, mean=0.0, std=0.01)
        nn.init.constant_(model.fc.bias, 0)
        return model
