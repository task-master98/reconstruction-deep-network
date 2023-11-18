import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
import numpy as np


class InceptionV3(nn.Module):

    dimensional_mapping = {
        64:   0,
        192:  1,
        768:  2,
        2048: 3
    }
    
    def __init__(self, apply_transforms: bool = True,                 
                 output_blocks: list = [0, 1, 2, 3],
                 required_latent_dim: int = 2048,
                 requires_grad = False):
        super().__init__()

        self.apply_transforms = apply_transforms
        self.output_blocks = sorted(output_blocks)
        self.last_needed_block = max(output_blocks)
        self.latent_dim_idx = self.dimensional_mapping[required_latent_dim]
        inception = torch.hub.load('pytorch/vision:v0.10.0', 'inception_v3', pretrained=True)
        inception.eval()

        self.blocks = nn.ModuleList()
        block0 = [
            inception.Conv2d_1a_3x3,
            inception.Conv2d_2a_3x3,
            inception.Conv2d_2b_3x3,
            nn.MaxPool2d(kernel_size=3, stride=2)
        ]
        self.blocks.append(nn.Sequential(*block0))
        if self.last_needed_block >= 1:
            block1 = [
                inception.Conv2d_3b_1x1,
                inception.Conv2d_4a_3x3,
                nn.MaxPool2d(kernel_size=3, stride=2)
            ]
            self.blocks.append(nn.Sequential(*block1))
        
        if self.last_needed_block >= 2:
            block2 = [
                inception.Mixed_5b,
                inception.Mixed_5c,
                inception.Mixed_5d,
                inception.Mixed_6a,
                inception.Mixed_6b,
                inception.Mixed_6c,
                inception.Mixed_6d,
                inception.Mixed_6e,
            ]
            self.blocks.append(nn.Sequential(*block2))

        # Block 3: aux classifier to final avgpool
        if self.last_needed_block >= 3:
            block3 = [
                inception.Mixed_7a,
                inception.Mixed_7b,
                inception.Mixed_7c,
                nn.AdaptiveAvgPool2d(output_size=(1, 1))
            ]
            self.blocks.append(nn.Sequential(*block3))
        
        for param in self.parameters():
            param.requires_grad = requires_grad
    
    def forward(self, x_image):

        outputs = []

        if self.apply_transforms:
            preprocess = transforms.Compose(
                [
                 transforms.Resize(299),
                 transforms.CenterCrop(299),                 
                 transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])]
            )

            x_inp = preprocess(x_image)            

            for idx, block in enumerate(self.blocks):
                x_inp = block(x_inp)

                if idx in self.output_blocks:
                    outputs.append(x_inp)
                
                elif idx == self.last_needed_block:
                    break
        
        return outputs[self.latent_dim_idx]


if __name__ == "__main__":   

    incp = InceptionV3()
    img = torch.randn(64, 3, 512, 512)    
    
    outputs = incp(img)
    print(outputs[0].shape)
    print(outputs[1].shape)
    print(outputs[2].shape)
    print(outputs[3].shape)

   

