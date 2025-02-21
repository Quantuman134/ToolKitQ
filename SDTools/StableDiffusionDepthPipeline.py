# Experimental

import torch
from ToolKitQ.SDTools.StableDiffusionPipeline import StableDiffusionPipeline

class StableDiffusionDepthPipeline(StableDiffusionPipeline):
    def __init__(self, lib='', device='cpu'):
        super().__init__(lib, device)
    
    def latent_denoise_step(self, latent, depth_map, text_embeddings, t, guidance_scale=7.5, grad_required=False):
        '''
        depth_map: [1, 1, H, W]
        '''
        latent_model_input = torch.cat([latent] * 2)
        latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)
        depth_map_model_input = torch.cat([depth_map] * 2)
        latent_model_input = torch.cat([latent_model_input, depth_map_model_input], dim=1)
        
        if grad_required:
            noise_pred = self.unet(latent_model_input, t, encoder_hidden_states=text_embeddings)[0]
        else:
            # predict noise, and disable the gradient calculate within unet to avoid tremendous memory prossession
            with torch.no_grad():
                noise_pred = self.unet(latent_model_input, t, encoder_hidden_states=text_embeddings)[0]

        noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
        noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

        # according to predicted noise, update the latents
        latent = self.scheduler.step(noise_pred, t, latent).prev_sample 

        return latent

    def text2img(self, depth_map, text_prompt, num_inference_step=50, negative_text_prompt="", guidance_scale=7.5):
        '''
        depth_map: [1, 1, H, W]
        '''
        latent = self.gaussian_noise_gen()
        self.ddim_config(num_inference_step)
        text_embeddings = self.text_encoding(text_prompt, negative_text_prompt)

        for timestep in self.scheduler.timesteps:
            latent = self.latent_denoise_step(latent, depth_map, text_embeddings, timestep, guidance_scale=guidance_scale)

        image = self.latent_decoding(latent)
        return image
