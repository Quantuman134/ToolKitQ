import torch
from ToolKitQ.PyTools.PytorchTools import seed_everything, torch_device_config, export_image_tensor
from diffusers import UNet2DConditionModel, AutoencoderKL, DDIMScheduler
from transformers import CLIPTextModel, CLIPTokenizer
import matplotlib.pyplot as plt

class StableDiffusionPipeline:
    def __init__(self, lib='', device='cpu'):
        '''
        lib: path of model, or repo name in huggingface
        '''
        self.tokenizer = CLIPTokenizer.from_pretrained(lib, subfolder="tokenizer")
        self.unet = UNet2DConditionModel.from_pretrained(lib, subfolder="unet").to(device)
        self.scheduler = DDIMScheduler.from_pretrained(lib, subfolder="scheduler")
        self.text_encoder = CLIPTextModel.from_pretrained(lib, subfolder="text_encoder").to(device)
        self.vae = AutoencoderKL.from_pretrained(lib, subfolder="vae").to(device)
        self.device = device

    def ddim_config(self, num_inference_steps=50):
        self.scheduler.set_timesteps(num_inference_steps)

    def gaussian_noise_gen(self, size=(1, 4, 64, 64)):
        noise = torch.randn(size=size, dtype=torch.float32, device=self.device)
        return noise

    def img_encoding(self, image, grad_required=False):
        if(grad_required):
            # range of color value [0, 1.0]
            image = 2 * image - 1
            latent = self.vae.encode(image).latent_dist.sample() * self.vae.config.scaling_factor
        else:
            with torch.no_grad():
                image = 2 * image - 1
                latent = self.vae.encode(image).latent_dist.sample() * self.vae.config.scaling_factor
        return latent

    def latent_decoding(self, latent, grad_required=False):
        if(grad_required):
            latent = latent / self.vae.config.scaling_factor 
            image = self.vae.decode(latent).sample
            image = (image + 1) / 2.0 
        else:
            with torch.no_grad():
                latent = latent / self.vae.config.scaling_factor
                image = self.vae.decode(latent).sample
                image = (image + 1) / 2.0 

        return image

    def text_encoding(self, text_prompt="", negative_text_prompt=""):
        text_tokens = self.tokenizer([text_prompt], padding="max_length", 
                                     max_length=self.tokenizer.model_max_length, 
                                     truncation=True, return_tensors="pt")
        text_embeddings = self.text_encoder(text_tokens.input_ids.to(self.device))[0]
        uncond_tokens = self.tokenizer([negative_text_prompt], padding="max_length", 
                                       max_length=self.tokenizer.model_max_length, 
                                       truncation=True, return_tensors="pt")
        uncond_embeddings = self.text_encoder(uncond_tokens.input_ids.to(self.device))[0]
        embeddings = torch.cat([uncond_embeddings, text_embeddings])

        return embeddings

    def add_noise(self, latent, t, noise=None):
        if noise is None:
            noise = torch.randn_like(latent)
            
        latent_noise = self.scheduler.add_noise(latent, noise, t)
        return latent_noise

    def latent_denoise_step(self, latent, text_embeddings, t, guidance_scale=7.5, grad_required=False):
        latent_model_input = torch.cat([latent] * 2)
        latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)
        
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

    def text2img(self, text_prompt, num_inference_step=50, negative_text_prompt="", guidance_scale=7.5):
        latent = self.gaussian_noise_gen()
        self.ddim_config(num_inference_step)
        text_embeddings = self.text_encoding(text_prompt, negative_text_prompt)

        for timestep in self.scheduler.timesteps:
            latent = self.latent_denoise_step(latent, text_embeddings, timestep, guidance_scale=guidance_scale)

        image = self.latent_decoding(latent)
        return image

if __name__ == "__main__":
    device = torch_device_config()
    seed_everything()
    sd = StableDiffusionPipeline("runwayml/stable-diffusion-v1-5", device)
    image = (sd.text2img(text_prompt="a cat"))
    export_image_tensor(image, "./ToolKitQ/TestSets/cat.png")
    
