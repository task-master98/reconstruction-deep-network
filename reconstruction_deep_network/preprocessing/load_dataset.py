from PIL import Image
import numpy as np
import pandas as pd

import os

module_dir = os.path.dirname(os.path.abspath(__file__))
package_dir = os.path.dirname(module_dir)
parent_dir = os.path.join(package_dir, "data")

class NYUv2Dataset:

    dataset_name = "nyu_data"

    def __init__(self, dataset_mode: str = "train"):
        self.metadata_df = self.load_metadata(dataset_mode)
        
    
    def load_metadata(self, dataset_mode: str):
        metadata_path = os.path.join(parent_dir, self.dataset_name, f"nyu2_{dataset_mode}_cleaned.csv")
        metadata_df = pd.read_csv(metadata_path)
        return metadata_df
    
    def extract_example_by_category(self, category: str):
        all_categories = list(self.metadata_df["category"].unique())
        assert category in all_categories

        category_df = self.metadata_df.loc[self.metadata_df["category"] == category].sample()
        colors_path = category_df["color_path"].values[0]
        depth_path = category_df["depth_path"].values[0]

        raw_color, raw_depth = self._load_example(colors_path), self._load_example(depth_path)
        print(f"Image Path: {colors_path}")
        return raw_color, raw_depth
    
    def _load_example(self, img_path):
        abs_img_path = os.path.join(parent_dir, self.dataset_name, img_path)
        img = Image.open(abs_img_path)
        img_arr = np.asarray(img, dtype=np.int32)
        return img_arr

if __name__ == "__main__":
    nyu2_train = NYUv2Dataset("train")
    color, depth = nyu2_train.extract_example_by_category("living_room_0038_out")

