"""
#TODO: 
-> implement FID score in validation_step
-> better metric than FID? Correlation between prompt and generated image
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import CosineAnnealingLR
import os
import pytorch_lightning as pl
import yaml
from einops import rearrange

import reconstruction_deep_network 
from reconstruction_deep_network.models.base_model import MultiViewBaseModel
from reconstruction_deep_network.models.inception import InceptionV3
from reconstruction_deep_network.models.pretrained import (load_pretrained_model_img,
                                                        load_pretrained_model_text)
from reconstruction_deep_network.metrics.metrics import (calculate_activation_statistics,
                                                        calculate_fretchet_inception_distance)


module_dir = reconstruction_deep_network.__path__[0]
root_dir = os.path.dirname(module_dir)
default_config_file = os.path.join(module_dir, "trainer", "trainer_config.yaml")


class ModelTrainer(pl.LightningModule):

    def __init__(self, trainer_config: str = default_config_file):
        super().__init__()

        with open(trainer_config, 'r') as f:
            self.config = yaml.load(f, Loader=yaml.FullLoader)
        
        self.lr = self.config["train"]["lr"]
        self.max_epochs = self.config["train"]["max_epochs"] if "max_epochs" in self.config["train"] else 0
        self.diffusion_timestep = self.config["model"]["diffusion_timestep"]
        self.guidance_scale = self.config["model"]["guidance_scale"]
        self.model_id = self.config["model"]["model_id"]

        # text embeddings
        if self.config["load_only_text"]:
            print("Loading text embedders...")
            self.tokenizer = load_pretrained_model_text(self.model_id, "tokenizer")
            self.text_encoder = load_pretrained_model_text(self.model_id, "text_encoder")

        if self.config["load_only_image"]:
            print("Loading image encoders...")
            self.vae = load_pretrained_model_img(self.model_id, "vae")
        
        if self.config["load_diffusion"]:
            print("Loading diffusion models...")
            self.scheduler = load_pretrained_model_img(self.model_id, "scheduler")
            self.mv_base_model = MultiViewBaseModel()
#             self.inception_model = InceptionV3()
            self.trainable_params = self.mv_base_model.trainable_parameters

        self.save_hyperparameters()
    
    def load_null_embedding(self):
        text_embeddings_dir = os.path.join(root_dir, self.config["dataset"]["text_embeddings_dir"])
        null_prompt_dir = os.path.join(text_embeddings_dir, "null")
        file_name = os.path.join(null_prompt_dir, "null.npz")
        null_prompt = np.load(file_name, allow_pickle=True)
        embeddings_dict = null_prompt["null"].item()
        embedding = torch.from_numpy(embeddings_dict["embeddings_1"])
        return embedding
    
    @torch.no_grad()
    def encode_text(self, text, device):
        text_inputs = self.tokenizer(
            text, padding = "max_length", max_length = self.tokenizer.model_max_length,
            truncation = True, return_tensors = "pt"
        )

        text_input_ids = text_inputs.input_ids
        if hasattr(self.text_encoder.config, "use_attention_mask") and self.text_encoder.config.use_attention_mask:
            attention_mask = text_inputs.attention_mask.cuda()
        else:
            attention_mask = None
        prompt_embeds = self.text_encoder(
            text_input_ids.to(device), attention_mask=attention_mask)
        
        return prompt_embeds[0].float(), prompt_embeds[1]
    
    @torch.no_grad()
    def encode_image(self, x_image, vae):
        # input shape bs, 2, height, width, channels
        batch_size = x_image.shape[0]

        x_image = x_image.permute(0, 1, 4, 2, 3)
        x_image = x_image.reshape(-1, x_image.shape[-3], x_image.shape[-2], x_image.shape[-1])

        z = vae.encode(x_image).latent_dist

        z = z.sample()
        z = z.reshape(batch_size, -1, z.shape[-3], z.shape[-2],
                      z.shape[-1])

        z = z * vae.config.scaling_factor
        z = z.float()
        return z
    
    @torch.no_grad()
    def decode_latent(self, latents, vae):
        batch_size, m = latents.shape[0:2]
        latents = (1 / vae.config.scaling_factor * latents)
        images = []

        for i in range(m):
            image = vae.decode(latents[:, i]).sample
            images.append(image)
        
        image = torch.stack(images, dim=1)
        image = (image / 2 + 0.5).clamp(0, 1)
        image = image.cpu().permute(0, 1, 3, 4, 2).float()        

        return image
    
    def configure_optimizers(self):
        param_groups = []
        for params, lr_scale in self.trainable_params:
            param_groups.append({"params": params, "lr": self.lr * lr_scale})
        
        optimizer = torch.optim.AdamW(param_groups)
        lr_scheduler = {
            "scheduler": CosineAnnealingLR(optimizer, T_max=self.max_epochs, eta_min=1e-7),
            "interval": "epoch",
            "name": "cosine_annealing_lr"
        }
        return {"optimizer": optimizer, "lr_scheduler": lr_scheduler}
    
    def training_step(self, batch):
        meta = {
            "K": batch["K"],
            "R": batch["R"]
        }

        device = batch["text_embedding"].device

        # # encode text
        # prompt_embeddings = []
        # for prompt in batch["prompt"]:
        #     embedding = self.encode_text(prompt, device)[0]
        #     prompt_embeddings.append(embedding)
        
        prompt_embeddings = batch["text_embedding"]

        # encode image
        latents = batch["img_encoding"]
        
        # timestep
        t = torch.randint(0, self.scheduler.num_train_timesteps, (latents.shape[0],), device = latents.device).long()

        # noise vector
        noise = torch.randn_like(latents)
        noise_z = self.scheduler.add_noise(latents, noise, t)

        # reshape timestep embedding
        t = t[:, None].repeat(1, latents.shape[1])

        denoise = self.mv_base_model(noise_z, t, prompt_embeddings, meta)
        target = noise

        loss = F.mse_loss(target, denoise)
        self.log("train_loss", loss)
        return loss
    
    def class_free_guidance_pair(self, latents, timestep, prompt_embed, batch):
        latents = torch.cat([latents] * 2)
        timestep = torch.cat([timestep] * 2)

        R = torch.cat([batch["R"]] * 2)
        K = torch.cat([batch["K"]] * 2)

        meta = {
            "K": K,
            "R": R
        }

        return latents, timestep, prompt_embed, meta
    
    @torch.no_grad()
    def forward_class_free(self, latents_high_res, timestep, prompt_embed, batch, model):
        latents, timestep, prompt_embed, meta = self.class_free_guidance_pair(
            latents_high_res, timestep, prompt_embed, batch
        )

        noise_pred = model(latents, timestep, prompt_embed, meta)
        noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
        noise_pred = noise_pred_uncond + self.guidance_scale * \
            (noise_pred_text - noise_pred_uncond)

        return noise_pred
    
    @torch.no_grad()
    def validation_step(self, batch, batch_idx):        
        images = batch["images"].detach().cpu()
        images_pred = self.inference(batch).to(images.device)
#         print(f"Images: {images.device}")
#         print(f"Prediction: {images_pred.device}")
#         fid_score = self.fretchet_inception_distance(images, images_pred)

        # images_pred = (images_pred.cpu().numpy() * 255).round().astype('uint8')

        # images = ((batch['images']/2+0.5)
        #                   * 255).cpu().numpy().astype(np.uint8)
        
        val_loss = F.mse_loss(images, images_pred)       

        self.log("val_loss", val_loss)
#         self.log("fid_score", fid_score)
        return val_loss
    
    @torch.no_grad()
    def inference(self, batch):
        images = batch["images"]
        bs, m, height, width, _ = images.shape
        device = images.device

        latents = torch.randn(
            bs, m, 4, height//8, width//8, device=device)
        
        prompt_embed = batch["text_embedding"]
        
        # prompt_embed = []
        # for prompt in batch["prompt"]:
        #     embedding = self.encode_text(prompt, device)[0]
        #     prompt_embed.append(embedding)
        
        # prompt_null = ""
        null_embedding = self.load_null_embedding()
        null_embedding = null_embedding[:, None].repeat(1, m, 1, 1).to(device)        
        complete_embedding = torch.cat([null_embedding, prompt_embed])

        self.scheduler.set_timesteps(self.diffusion_timestep, device = device)
        timesteps = self.scheduler.timesteps
        
        for i, t in enumerate(timesteps):
            _timestep = torch.cat([t[None, None]]*m, dim=1)

            noise_pred = self.forward_class_free(
                latents, _timestep, complete_embedding, batch, self.mv_base_model)
            
            latents = self.scheduler.step(
                noise_pred, t, latents).prev_sample
        
        images_pred = self.decode_latent(
            latents, self.vae)
        
        return images_pred
    
    @torch.no_grad()
    def fretchet_inception_distance(self, images: np.ndarray, images_pred: np.ndarray):
        images_pred = rearrange(images_pred, 'b m c h w -> (b m) c h w')
        images = rearrange(images, 'b m c h w -> (b m) c h w')

        images = images.permute(0, 3, 1, 2)
        images_pred = images_pred.permute(0, 3, 1, 2)
        
        real_img_incp_latent = self.inception_model(images)
        fake_img_incp_latent = self.inception_model(images_pred)

        real_img_incp_latent = real_img_incp_latent.cpu().numpy()
        fake_img_incp_latent = fake_img_incp_latent.cpu().numpy()

        mu1, sigma1 = calculate_activation_statistics(real_img_incp_latent)
        mu2, sigma2 = calculate_activation_statistics(fake_img_incp_latent)

        fid_score = calculate_fretchet_inception_distance(mu1, sigma1, mu2, sigma2)
        return fid_score



if __name__ == "__main__":
    trainer = ModelTrainer()
    null_embedding = trainer.load_null_embedding()
    # images = torch.randn(4, 2, 512, 512, 3)
    # latents = trainer.encode_image(images, trainer.vae)
    # print(latents.shape)




