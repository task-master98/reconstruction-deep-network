import torch
import torch.nn.functional as F
import numpy as np
from scipy import linalg

def calculate_fretchet_inception_distance(
    mu1: np.ndarray, sigma1: np.ndarray,
    mu2: np.ndarray, sigma2: np.ndarray,
    eps: float = 1e-5):

    mu1 = np.atleast_1d(mu1)
    mu2 = np.atleast_1d(mu2)

    sigma1 = np.atleast_2d(sigma1)
    sigma2 = np.atleast_2d(sigma2)

    sigma1 += eps * np.eye(sigma1.shape[0])
    sigma2 += eps * np.eye(sigma2.shape[0])

    ssdiff = np.sum((mu1 - mu2)**2.0)


    covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)    
    
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    
    fid = ssdiff + np.trace(sigma1 + sigma2 - 2.0 * covmean)
    fid = np.round(fid, decimals=5)
    return fid

def calculate_activation_statistics(prediction: np.ndarray):
    if prediction.ndim > 2:
        prediction = np.squeeze(prediction, axis=(2, 3))
    mu = np.mean(prediction, axis=0)
    sigma = np.cov(prediction, rowvar=False)
    return mu, sigma
    
if __name__ == "__main__":
    ...
# define two collections of activations
    act1 = np.random.randn(10, 2048)
    act2 = np.random.randn(10, 2048)    
    

    mu1, sigma1 = calculate_activation_statistics(act1)
    mu2, sigma2 = calculate_activation_statistics(act2)

    fid_diff = calculate_fretchet_inception_distance(mu1, sigma1, mu2, sigma2)
    fid_same = calculate_fretchet_inception_distance(mu1, sigma1, mu1, sigma1)
    
    print(fid_diff)
    print(fid_same)



