import torch
import torch.nn.functional as F
import numpy as np
from scipy import linalg

def calculate_fretchet_inception_distance(
    mu1: np.ndarray, sigma1: np.ndarray,
    mu2: np.ndarray, sigma2: np.ndarray,
    eps: float = 1e-6):

    mu1 = np.atleast_1d(mu1)
    mu2 = np.atleast_1d(mu2)

    sigma1 = np.atleast_2d(sigma1)
    sigma2 = np.atleast_2d(sigma2)

    diff = mu1 - mu2

    covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
    if not np.isfinite(covmean).all():
        offset = np.eye(sigma1.shape[0]) * eps
        covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))
    
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    
    covmean_trace = np.trace(covmean)
    fid = (diff.dot(diff) + np.trace(sigma1) + np.trace(sigma2)) - 2 * covmean_trace
    return fid

def calculate_activation_statistics(self, prediction: np.ndarray):
    mu = np.mean(prediction, axis=0)
    sigma = np.cov(prediction, rowvar=False)
    return mu, sigma
    



