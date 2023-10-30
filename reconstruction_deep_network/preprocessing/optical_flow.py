import numpy as np
import cv2
import math
import matplotlib.pyplot as plt
import os

import reconstruction_deep_network
from reconstruction_deep_network.data_loader.matterport import MatterPortData
from reconstruction_deep_network.preprocessing.feature_points import FeaturePointDetector

package_dir = reconstruction_deep_network.__path__[0]
result_dir = os.path.join(package_dir, "results")

class FeatureMatching:

    flann_index_kdtree = 1
    index_params = {
        "surf": {
            "algorithm": 1,
            "trees": 5
        },
        "orb": {
            "algorithm": 6,
            "table_number": 6,
            "key_size": 12,
            "multi_probe_level": 1
        },
        "sift": {
            "algorithm": 1,
            "trees": 5
        }
    }


    def __init__(self):
        pass

    def brute_force_matching(self, img_kp1: np.ndarray, img_kp2: np.ndarray):
        bf = cv2.BFMatcher()
        matches = bf.knnMatch(img_kp1, img_kp2, k = 2)
        good_matches = []
        for m, n in matches:
            if m.distance < 0.75 * n.distance:
                good_matches.append([m])
        
        return good_matches
    
    def flann_matching(self, des1: np.ndarray, des2: np.ndarray, keypoint_type: str, ratio_thr: float):
        flann_index_params = self.index_params[keypoint_type]
        search_params = {"checks": 50} 
        flann = cv2.FlannBasedMatcher(flann_index_params, search_params)
        matches = flann.knnMatch(des1, des2, k=2)
        matches_mask = [[0, 0] for i in range(len(matches))]

        for i, (m, n) in enumerate(matches):
            if m.distance < ratio_thr * n.distance:
                matches_mask[i] = [1, 0]
        
        return matches, matches_mask

class MatchVisualizer:

    draw_params = {
        "matchColor": (0, 255, 0),
        "singlePointColor": (255, 0, 0),
        "flags": cv2.DrawMatchesFlags_DEFAULT

    }

    
    def visualize_matches(self, img1: np.ndarray, img2: np.ndarray,
                          kp1: np.ndarray, kp2: np.ndarray,  
                          macthes: np.ndarray, mask = None):
        if mask is not None:
            self.draw_params["matchesMask"] = mask       
        
        img3 = cv2.drawMatchesKnn(img1, kp1, img2, kp2, macthes, None, **self.draw_params)
        return img3
    
    @staticmethod
    def plot_image(image: np.ndarray):
        plt.imshow(image)
        plt.show()
    
    @staticmethod
    def save_figure(image: np.ndarray, filename: str):
        plt.imshow(image)
        plt.savefig(os.path.join(result_dir, f"{filename}.jpg"))
        


if __name__ == "__main__":
    scan_hash = "17DRP5sb8fy"
    matterport = MatterPortData(scan_hash)
    panorama_id = "0f37bd0737e349de9d536263a4bdd60d"
    color_image_1 = matterport.load_color_image("0f37bd0737e349de9d536263a4bdd60d", 1, 4)
    color_image_2 = matterport.load_color_image("0f37bd0737e349de9d536263a4bdd60d", 2, 4)

    feature_point_det = FeaturePointDetector()

    img1_kp, des1 = feature_point_det.sift_corners(color_image_1)
    img2_kp, des2 = feature_point_det.sift_corners(color_image_2)

    feature_matcher = FeatureMatching()
    matches, mask = feature_matcher.flann_matching(des1, des2, "sift")

    match_visualizer = MatchVisualizer()
    img = match_visualizer.visualize_matches(color_image_1, color_image_2, img1_kp, img2_kp, matches, mask)
    # match_visualizer.plot_image(img)
    match_visualizer.save_figure(img, f"{panorama_id}_{14}_{24}")
    









