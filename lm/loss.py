"""Defines different loss functions."""
import torch
import torch.nn as nn
from absl import flags

FLAGS = flags.FLAGS


def cross_entropy_loss(reduction='mean'):
    """Computes cross entropy loss."""
    return nn.CrossEntropyLoss(reduction=reduction)

def bce_loss(reduction='mean'):
    """Computes binary cross entropy loss."""
    # ? is this equivalent to tf.keras.losses.BinaryCrossentropy(from_logits=True, reduction=tf.keras.losses.Reduction.NONE)
    return nn.BCEWithLogitsLoss(reduction=reduction)   


def cross_entropy_loss_smooth(label_smoothing, reduction='mean'):
    """Computes cross entropy loss with label smoothing."""
    return nn.CrossEntropyLoss(label_smoothing=label_smoothing, reduction=reduction)