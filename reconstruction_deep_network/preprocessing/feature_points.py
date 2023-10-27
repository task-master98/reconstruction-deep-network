import numpy as np
import cv2

from reconstruction_deep_network.data_loader.matterport import MatterPortData

class FeaturePointDetector:

    harris_params = {
        "blockSize": 2,
        "ksize": 3,
        "k": 0.04
    }

    shi_thomas_params = {
        "maxCorners": 2000,
        "qualityLevel": 0.02,
        "minDistance": 20
    }

    def __init__(self):
        pass

    @staticmethod
    def convert_to_grayscale(image: np.ndarray):
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        gray_image = np.float32(gray_image)
        return gray_image
    
    def corner_harris(self, raw_image: np.ndarray):
        image = self.convert_to_grayscale(raw_image)
        dst = cv2.cornerHarris(image, **self.harris_params)
        dst = cv2.dilate(dst, None)        
        corners = dst > 0.01 * dst.max()
        return corners
    
    def shi_tomasi_corner(self, raw_image):
        image = self.convert_to_grayscale(raw_image)
        corners = cv2.goodFeaturesToTrack(image, **self.shi_thomas_params)
        corners = np.float32(corners)
        return corners.squeeze()
    
    def sift_corners(self, raw_image: np.ndarray):
        image = self.convert_to_grayscale(raw_image)
        sift = cv2.xfeatures2d.SIFT_create()
        image = image.astype(np.uint8)
        kp, des = sift.detectAndCompute(image, None)
        return kp
    
    def surf_corners(self, raw_image: np.ndarray):
        image = self.convert_to_grayscale(raw_image)
        image = image.astype(np.uint8)
        fast = cv2.FastFeatureDetector.create()
        fast.setNonmaxSuppression(False)
        kp = fast.detect(image, None)
        return kp
    
class VisualizeCorners:

    @staticmethod
    def visualize_corners(image: np.ndarray, feature_type: str):
        cv2.imshow(feature_type, image)
        cv2.waitKey()
    
    @staticmethod
    def standardize_features(image: np.ndarray, corners: np.ndarray):
        if corners.dtype == np.float32:
            corners = corners.astype(int)        
            for corner in corners:
                x, y = corner
                image[y, x] = [0, 255, 0]
        else:
            image[corners] = [0, 255, 0]
        return image
    
    @staticmethod
    def visualize_keypoints(image: np.ndarray, keypoints: np.ndarray):
        kp_image = cv2.drawKeypoints(image, keypoints, None, color=(0, 255, 0))
        return kp_image


if __name__ == "__main__":
    scan_hash = "17DRP5sb8fy"
    matterport = MatterPortData(scan_hash)
    color_image = matterport.load_color_image("0f37bd0737e349de9d536263a4bdd60d", 2, 4)

    feature_extractor = FeaturePointDetector()
    # harris_corners = feature_extractor.corner_harris(color_image)
    # shi_thomas_corners = feature_extractor.shi_tomasi_corner(color_image)
    sift_corners = feature_extractor.sift_corners(color_image)
    surf_corners = feature_extractor.surf_corners(color_image)

    visualizer = VisualizeCorners()
    # transformed_image = visualizer.standardize_features(color_image, shi_thomas_corners)
    # visualizer.visualize_corners(transformed_image, "shi-thomas")
    kp_image = visualizer.visualize_keypoints(color_image, surf_corners)
    visualizer.visualize_corners(kp_image, "surf")
