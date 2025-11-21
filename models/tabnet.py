import torch
import torch.nn as nn
from pytorch_tabnet.tab_model import TabNetClassifier

class TabNetBrainTumorClassifier(nn.Module):
    """
    Wrapper for TabNetClassifier for feature-based or flattened CNN outputs.
    For direct image data, you would use CNN to extract features first.
    """
    def __init__(self, input_dim=512, num_classes=4):
        super(TabNetBrainTumorClassifier, self).__init__()
        self.tabnet = TabNetClassifier(
            input_dim=input_dim,
            output_dim=num_classes,
            n_d=64, n_a=64,
            n_steps=5,
            gamma=1.5,
            n_independent=2, n_shared=2,
            momentum=0.02,
        )

    def forward(self, x):
        """
        Forward pass for compatibility with PyTorch-style evaluation.
        """
        return self.tabnet.predict(x)
