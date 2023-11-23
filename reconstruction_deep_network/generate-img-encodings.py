import numpy as np
import os
import torch
from argparse import ArgumentParser
import yaml

import reconstruction_deep_network
from reconstruction_deep_network.trainer.trainer import ModelTrainer
from reconstruction_deep_network.data_loader.custom_loader import CustomDataLoader

module_dir = reconstruction_deep_network.__path__[0]
root_dir = os.path.dirname(module_dir)
data_dir = os.path.join(root_dir, "data", "v1")

def save_image_encoding(img_encoding: np.ndarray, scan_id: str, img_name: str, img_encoding_dir: str):
    dir_name = os.path.join(img_encoding_dir, scan_id)
    if not os.path.isdir(dir_name):
        os.makedirs(dir_name)
    
    file_name = os.path.join(dir_name, f"{img_name}.npz")    
    np.savez(file_name, latent = img_encoding)
    

def parse_args():
    parser = ArgumentParser()
    parser.add_argument("--data_path", dest = "data_path", type = str, default = data_dir)
    parser.add_argument("--debug", dest = "debug", type = bool, default = True)
    parser.add_argument("--id", dest = "id", type = str, default = None)
    # parser.add_argument("--file_type", dest = "file_type", type = str)
    # parser.add_argument("--dataset_mode", dest = "dataset_mode", type = str)

    args = parser.parse_args()
    return args


def main(args):
    
    main_data_dir = args.data_path
    scans_dir = os.path.join(main_data_dir, "scans")
    img_latents_dir = os.path.join(main_data_dir, "image_latents")
    if not os.path.isdir(img_latents_dir):
        os.makedirs(img_latents_dir)

    # initialize model trainer to encode images
    model_trainer = ModelTrainer()

    house_scans = None
    if args.debug:
        house_scans = ["17DRP5sb8fy"]
    else:
        if args.id == "ALL":
            house_scans = os.listdir(scans_dir)
        else:
            house_scans = [args.id]

    for itr, house_scan in enumerate(house_scans):

        dataset = CustomDataLoader(debug = False)
        dataset.metadata = dataset._limit_dataset(scan_id = house_scan)

        print(f"House Scan: {house_scan}, Number of Scenes: {len(dataset)}")

        for scene_idx in range(len(dataset)):

            scene_dict = dataset[scene_idx]
            img_paths = scene_dict["images_paths"]
            path_components = img_paths[0].split("/")
            img_name = path_components[-1].split("_")[0]
            if scene_dict["img_encoding"] is not None: continue

            images = scene_dict["images"]
            images = torch.from_numpy(images)
            images = torch.unsqueeze(images, 0)
            img_encoding = model_trainer.encode_image(images, model_trainer.vae)
            img_encoding = img_encoding.numpy().squeeze()

            save_image_encoding(img_encoding, house_scan, img_name, img_latents_dir)
            print(f"Saved encoding: {house_scan}, scene: {img_name}")


if __name__ == "__main__":
    args = parse_args()
    main(args)
            







