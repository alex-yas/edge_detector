from typing import Tuple

import numpy as np


def precision_recall_f1(prediction: np.array, 
                        target: np.array) -> Tuple[float, float, float]:
    """ Calculates pixel-wise precision, recall, f1-score metrics

    Args:
        prediction (np.array): model prediction - 2d array
        target (np.array): target mask - 2d array

    Returns:
        Tuple[float, float, float]: precision, recall, f1-score metrics
    """
    pixel_prediction = (prediction > 0).astype(np.float32)
    opencv_pixel = (target > 0).astype(np.float32)
    
    true_positive = (pixel_prediction * opencv_pixel).sum()
    false_positive = (pixel_prediction * (1 - opencv_pixel)).sum()
    false_negative = ((1 - pixel_prediction) * opencv_pixel).sum()
    
    precision = true_positive / (true_positive + false_positive)
    recall = true_positive / (true_positive + false_negative)
    f1 = 2 * (precision * recall) / (precision + recall)
    
    return precision.item(), recall.item(), f1.item()