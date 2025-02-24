# Experimental

import torch
from ToolKitQ.SDTools.StableDiffusionPipeline import StableDiffusionPipeline

class StableDiffusionInpaintPipeline(StableDiffusionPipeline):
    def __init__(self, lib='', device='cpu'):
        super().__init__(lib, device)
    
    def latent_denoise_step(self, latent, mask, masked_image_latent, text_embeddings, t, guidance_scale=7.5, grad_required=False):
        '''
        mask: [1, 1, H, W], size of latent
        masked_image_latent: [1, 4, H, W]
        '''
        latent_model_input = torch.cat([latent] * 2)
        latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)
        mask_model_input = torch.cat([mask] * 2)
        masked_image_latent_model_input = torch.cat([masked_image_latent] * 2)
        latent_model_input = torch.cat([latent_model_input, mask_model_input, masked_image_latent_model_input], dim=1)
        
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

    def inpaint_img(self, img, mask, text_prompt, num_inference_step=50, negative_text_prompt="", guidance_scale=7.5):
        '''
        mask: [1, 1, H, W], size of rgb image
        img: [1, 3, H, W]
        '''
        latent = self.gaussian_noise_gen()
        
        masked_img = ((img * 2.0 - 1.0) * (mask < 0.5) + 1.0) / 2.0 # value 1.0 of mask correspond to 0.5 value on rgb image
        masked_img_latent = self.img_encoding(masked_img)
        
        mask = torch.nn.functional.interpolate(mask, size=(64, 64))
        
        self.ddim_config(num_inference_step)
        text_embeddings = self.text_encoding(text_prompt, negative_text_prompt)

        for timestep in self.scheduler.timesteps:
            latent = self.latent_denoise_step(latent, mask, masked_img_latent, text_embeddings, timestep, guidance_scale=guidance_scale)
            image = self.latent_decoding(latent)

        image = self.latent_decoding(latent)
        return image

if __name__ == '__main__':
    from ToolKitQ.PyTools.PytorchTools import torch_device_config, import_image_tensor, export_image_tensor
    
    device = torch_device_config()
    img_tensor = import_image_tensor('./ToolKitQ/SDTools/sample/dog_bench.png', size=(512, 512), device=device)[:, 0:3, :, :]
    mask_tensor = import_image_tensor('./ToolKitQ/SDTools/sample/mask.png', size=(512, 512), device=device)[:, 0:1, :, :]
    
    text_prompt = ''
    
    sd_inpaint_pipe = StableDiffusionInpaintPipeline('stabilityai/stable-diffusion-2-inpainting', device=device)
    inpaint_img_tensor = sd_inpaint_pipe.inpaint_img(img_tensor, mask_tensor, text_prompt)
    
    export_image_tensor(inpaint_img_tensor, './ToolKitQ/SDTools/sample/inpaint_result.png')
    
