# src/models/tabnet.py
from pytorch_tabnet.tab_model import TabNetClassifier
import torch

def create_tabnet_model():
    """
    Creates and configures a TabNetClassifier model.
    """
    # These are common hyperparameters for TabNet.
    # The learning rate (lr) is often set higher than for other models.
    tabnet_params = dict(
        optimizer_fn=torch.optim.Adam,
        optimizer_params=dict(lr=2e-2),
        scheduler_params={"step_size": 10, "gamma": 0.9},
        scheduler_fn=torch.optim.lr_scheduler.StepLR,
        mask_type='sparsemax'  # This is a key part of the TabNet architecture
    )

    model = TabNetClassifier(**tabnet_params)
    
    return model