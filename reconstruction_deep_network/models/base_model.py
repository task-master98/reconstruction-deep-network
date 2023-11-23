import torch
import torch.nn as nn
from einops import rearrange
import yaml
import os

import reconstruction_deep_network
from reconstruction_deep_network.models.pretrained import load_pretrained_model_img
from reconstruction_deep_network.models.utils import *

module_dir = reconstruction_deep_network.__path__[0]
default_config_path = os.path.join(module_dir, "models", "model_config.yaml")

class MultiViewBaseModel(nn.Module):

    def __init__(self, model_config_path: str = default_config_path):
        super().__init__()
        with open(model_config_path, 'r') as f:
            self.config = yaml.load(f, Loader=yaml.FullLoader)

        self.unet = load_pretrained_model_img(self.config["model_id"], "unet")
        self.single_image_ft = self.config["single_image_ft"]

        if self.single_image_ft:
            self.trainable_parameters = [(self.unet.parameters(), 0.01)]
    
    def forward(self, latents, timestep, prompt_embed, meta):
        # camera intrinsic
        K = meta["K"]
        # rotation
        R = meta["R"]

        b, m, c, h, w = latents.shape
        img_h, img_w = h*8, w*8
        # correspondences=get_correspondences(R, K, img_h, img_w)

        hidden_states = rearrange(latents, 'b m c h w -> (b m) c h w')
        prompt_embed = rearrange(prompt_embed, 'b m l c -> (b m) l c')

        timestep = timestep.reshape(-1)
        t_projection = self.unet.time_proj(timestep)
        t_embed = self.unet.time_embedding(t_projection)

        hidden_states = self.unet.conv_in(
            hidden_states)  # bs*m, 320, 64, 64
        
        down_block_res_samples = (hidden_states,)

        for i, downsample_block in enumerate(self.unet.down_blocks):
            for resnet in downsample_block.resnets:
                hidden_states = resnet(hidden_states, t_embed)
                down_block_res_samples += (hidden_states,)
            
            if downsample_block.downsamplers is not None:
                for downsample in downsample_block.downsamplers:
                    hidden_states = downsample(hidden_states)
                down_block_res_samples += (hidden_states,)
        
        hidden_states = self.unet.mid_block.resnets[0](hidden_states, t_embed)

        for attn, resnet in zip(self.unet.mid_block.attentions, self.unet.mid_block.resnets[1:]):
            hidden_states = attn(
                hidden_states, encoder_hidden_states=prompt_embed
            ).sample
            hidden_states = resnet(hidden_states, t_embed)
        
        h, w = hidden_states.shape[-2:]

        for i, upsample_block in enumerate(self.unet.up_blocks):
            res_samples = down_block_res_samples[-len(upsample_block.resnets):]
            down_block_res_samples = down_block_res_samples[:-len(
                upsample_block.resnets)]
            for resnet in upsample_block.resnets:
                res_hidden_states = res_samples[-1]
                res_samples = res_samples[:-1]
                hidden_states = torch.cat(
                    [hidden_states, res_hidden_states], dim=1)
                hidden_states = resnet(hidden_states, t_embed)
        
            if upsample_block.upsamplers is not None:
                for upsample in upsample_block.upsamplers:
                    hidden_states = upsample(hidden_states)

        sample = self.unet.conv_norm_out(hidden_states)
        sample = self.unet.conv_act(sample)
        sample = self.unet.conv_out(sample)
        sample = rearrange(sample, '(b m) c h w -> b m c h w', m=m)
        return sample


if __name__ == "__main__":

    model = MultiViewBaseModel()
    print("Model Initialized")




