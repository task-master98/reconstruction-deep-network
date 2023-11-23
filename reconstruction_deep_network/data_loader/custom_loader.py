import numpy as np
import os
import pandas as pd
import cv2
import itertools
import random
import yaml
import torch
import reconstruction_deep_network

module_dir = reconstruction_deep_network.__path__[0]
root_dir = os.path.dirname(module_dir)
default_config_file = os.path.join(module_dir, "data_loader", "data_loader_config.yaml")

shape_dictionary = {
    "images": (8, 512, 512, 3),
    "R": (8, 3, 3),
    "K": (8, 3, 3),
    "text_encoding": (8, 77, 1024),
    "img_encoding": (8, 4, 64, 64)
}

def get_K_R(FOV, THETA, PHI, height, width):
    f = 0.5 * width * 1 / np.tan(0.5 * FOV / 180.0 * np.pi)
    cx = (width - 1) / 2.0
    cy = (height - 1) / 2.0
    K = np.array([
        [f, 0, cx],
        [0, f, cy],
        [0, 0,  1],
    ], np.float32)

    y_axis = np.array([0.0, 1.0, 0.0], np.float32)
    x_axis = np.array([1.0, 0.0, 0.0], np.float32)
    R1, _ = cv2.Rodrigues(y_axis * np.radians(THETA))
    R2, _ = cv2.Rodrigues(np.dot(R1, x_axis) * np.radians(PHI))
    R = R2 @ R1
    return K, R

def warp_img(fov, theta, phi, images, vx, vy):
    img_combine = np.zeros(images[0].shape).astype(np.uint8)

    min_theta = 10000
    for i, img in enumerate(images):
        _theta = vx[i]-theta
        _phi = vy[i]-phi

        if i == 2 and theta > 270:
            _theta = max(360-theta, _theta)
        if _phi == 0 and np.absolute(_theta) > 90:
            continue

        if i > 0 and i < 5 and np.absolute(_theta) < min_theta:
            min_theta = _theta
            min_idx = i

        im_h, im_w, _ = img.shape
        K, R = get_K_R(fov, _theta, _phi, im_h, im_w)
        homo_matrix = K@R@np.linalg.inv(K)
        img_warp1 = cv2.warpPerspective(img, homo_matrix, (im_w, im_h))
        if i == 0:
            img_warp1[im_h//2:] = 0
        elif i == 5:
            img_warp1[:im_h//2] = 0

        img_combine += img_warp1  # *255).astype(np.uint8)
    return img_combine

class CustomDataLoader(torch.utils.data.Dataset):

    def __init__(self, config_file: str=default_config_file, mode: str="train", debug: bool = True):
        with open(config_file, 'rb') as f:
            self.config = yaml.load(f, Loader=yaml.FullLoader)
        
        self.mode = mode
        random.seed(self.config["seed"])
        metadata_file_path = os.path.join(module_dir, "data_loader", f"{mode}.npy")
        metadata = np.load(metadata_file_path)
        str_arr = metadata.astype(str)
        self.metadata = str_arr.tolist()
        if debug:
            self.metadata = self._limit_dataset("17DRP5sb8fy")
        
        self.image_dir = os.path.join(root_dir, self.config["image_dir"])
        self.prompts_dir = os.path.join(root_dir, self.config["prompts_dir"])
        self.text_embeddings_dir = os.path.join(root_dir, self.config["text_embeddings_dir"])
        self.img_embeddings_dir = os.path.join(root_dir, self.config["img_embeddings_dir"])
        self.vx = self.config["vx"]
        self.vy = self.config["vy"]
        self.rot = self.config["rot"]
        self.fov = self.config["fov"]
        self.resolution = self.config["resolution"]
        self.crop_size = self.config["crop_size"]
        self.data_type = self.config["data_type"]
        self.skybox_indices = self.config["skybox_indices"]
    
    def __len__(self):
        return len(self.metadata)
    
    def load_skybox_image(self, img_name: str):        
        img_path = os.path.join(self.image_dir, img_name)
        img = cv2.imread(img_path)
        return img
    
    def load_text_encoding(self, img_path: str):
        path_components = img_path.split("/")
        scan_id, img_name = path_components[0], path_components[-1].split("_")[0]
        file_name = os.path.join(self.text_embeddings_dir, scan_id, f"{img_name}.npz")
        try:
            np_tensor = np.load(file_name, allow_pickle=True)
            embedding_vec = []
            for key in np_tensor.files:
                data_dict = np_tensor[key].item()
                embeddings = torch.from_numpy(data_dict["embeddings_1"])
                embedding_vec.append(embeddings)
            
            torch_tsr = torch.stack(embedding_vec, dim=1).numpy()
            
        except Exception as e:
            torch_tsr = None
        
        return torch_tsr
    
    def load_img_encoding(self, img_path: str):
        path_components = img_path.split("/")
        scan_id, img_name = path_components[0], path_components[-1].split("_")[0]
        file_name = os.path.join(self.img_embeddings_dir, scan_id, f"{img_name}.npz")
        try:
            np_tensor = np.load(file_name)["latent"]
        except Exception as e:
            np_tensor = None
        return np_tensor
    
    def load_prompt(self, scan_id: str, img_name: str, rotation: int):
        file_name = f"{img_name}_{rotation}.txt"
        model = "blip3"
        file_path = os.path.join(self.prompts_dir, scan_id, model, file_name)
        with open(file_path, 'r') as f:
            prompt = f.readlines()[0]
        return prompt
    
    def crop_img(self, img, K):
        margin = (self.resolution-self.crop_size)//2
        img_crop = img[margin:-margin, margin:-margin]
        K=copy.deepcopy(K)
        K[0, 2] -= margin
        K[1, 2] -= margin

        return img_crop, K
    
    def _limit_dataset(self, scan_id):
        start_index, end_index = None, None
        data_list = self.metadata

        for i, sub_list in enumerate(data_list):
            if sub_list[0].startswith(scan_id):
                if start_index is None:
                    start_index = i
                end_index = i

        return data_list[start_index:end_index+1]
    
    # def __getitem__(self, idx):
    #     idx = np.random.randint(0, 1000)
    #     np.random.seed(idx)
    #     R = np.random.rand(*shape_dictionary["R"]).astype(np.float32)
    #     K = np.random.rand(*shape_dictionary["K"]).astype(np.float32)
    #     text_encoding = np.random.rand(*shape_dictionary["text_encoding"]).astype(np.float32)
    #     img_encoding = np.random.rand(*shape_dictionary["img_encoding"]).astype(np.float32)
    #     images = np.random.rand(*shape_dictionary["images"]).astype(np.float32)

    #     return {
    #         "images": images,
    #         "text_embedding": text_encoding,
    #         "img_encoding": img_encoding,
    #         "R": R,
    #         "K": K
    #     }
    
    def __getitem__(self, idx):
        img_paths = self.metadata[idx]
        raw_images = [self.load_skybox_image(path) for path in img_paths]
        raw_images = [cv2.cvtColor(raw_img, cv2.COLOR_BGR2RGB) for raw_img in raw_images]

        text_encoding = self.load_text_encoding(img_paths[0])
        img_encoding = self.load_img_encoding(img_paths[0])

        if text_encoding is not None:
            text_encoding = text_encoding.squeeze()
        if img_encoding is not None:
            img_encoding = img_encoding.squeeze()

        imgs = []
        Rs = []
        num_views=8

        init_degree = random.randint(0, 359) if self.mode == "train" else 0

        for i in range(num_views):
            _degree = (init_degree+self.rot*i) % 360
            img = warp_img(
                90, _degree, 0, raw_images, self.vx, self.vy)
            img = cv2.resize(img, (self.resolution, self.resolution))
            
            

            K, R = get_K_R(90, _degree, 0,
                           self.resolution, self.resolution)
            if self.crop_size!=self.resolution:
                img, K = self.crop_img(img, K)
            Rs.append(R)
            imgs.append(img)
        
        images = (np.stack(imgs).astype(np.float32)/127.5)-1

        K = np.stack([K]*len(Rs)).astype(np.float32)
        R = np.stack(Rs).astype(np.float32)

        prompts = []
        scan_id = img_paths[0].split("/")[0]
        img_name = img_paths[0].split("/")[-1].split("_")[0]
        for itr in range(num_views):
            _degree = (init_degree+itr*self.rot) % 360
            _degree = int(np.round(_degree/45)*45) % 360
            prompt = self.load_prompt(scan_id, img_name, _degree)
            prompts.append("This is one view of a scene. " + prompt) #TODO: current text embeddings do not include the prefix: "This is one view of a scene"
        
        return {
            "images_paths": img_paths,
            "images": images,
            "prompt": prompts,
            "text_embedding": text_encoding,
            "img_encoding": img_encoding,
            "R": R,
            "K": K
        }
        


if __name__ == "__main__":

    config_file = os.path.join(module_dir, "data_loader", "data_loader_config.yaml")
    data_loader = CustomDataLoader(config_file, "train")

    item = data_loader[0]
    


