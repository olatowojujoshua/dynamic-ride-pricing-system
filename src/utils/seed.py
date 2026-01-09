"""
Random seed utilities for reproducibility
"""

import random
import numpy as np
from typing import Optional

def set_random_seed(seed: int = 42) -> None:
    """
    Set random seeds for Python, NumPy, and other libraries
    
    Args:
        seed: Random seed value
    """
    random.seed(seed)
    np.random.seed(seed)
    
    # Note: TensorFlow/PyTorch seeds would be set here if used
    # import tensorflow as tf
    # tf.random.set_seed(seed)
    
    # import torch
    # torch.manual_seed(seed)
    # if torch.cuda.is_available():
    #     torch.cuda.manual_seed(seed)
    #     torch.cuda.manual_seed_all(seed)
