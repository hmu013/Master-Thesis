from concurrent.futures import ThreadPoolExecutor
import torch
import os
import numpy as np
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm
from PIL import Image
from diffusers import StableDiffusionPipeline, DDIMInverseScheduler, AutoencoderKL, DDIMScheduler
from .UnprocessedDataset import UnprocessedDataset

#tpe to ofload saving to cpu 
executor = ThreadPoolExecutor(max_workers=4)

#set for better performance
torch.set_grad_enabled(False)
torch.backends.cudnn.benchmark = True

def img_to_latents(images: torch.Tensor, vae: AutoencoderKL):
    images = 2 * images - 1       # to go from [0,1] to [-1,1]
    posterior = vae.encode(images).latent_dist
    latents = posterior.mean * vae.config.scaling_factor
    return latents

def latents_to_img(latents, vae):
    decoded = vae.decode(latents / vae.config.scaling_factor).sample
    decoded = (decoded / 2 + 0.5).clamp(0, 1)
    decoded = decoded.mul(255).add_(0.5).clamp_(0, 255).permute(0, 2, 3, 1)
    return decoded.to("cpu", torch.uint8).numpy()
      
def to_uint8(t):
    return (
        t.mul(255)
         .add_(0.5)
         .clamp_(0, 255)
         .to(torch.uint8)
         .permute(0, 2, 3, 1)
         .cpu() # moves to cpu before saving 
         .numpy()
    )

def save_batch(original_numpy, inv_numpy, reconstructed_numpy, names, output_path):
    #loop for saving images
            for original, inverted, reconstructed, name in zip(original_numpy, inv_numpy, reconstructed_numpy, names):
                
                #remove filextension from image name
                base_name = os.path.splitext(name)[0] 
                
                original_image_path = os.path.join(output_path, 'original', f"{base_name}.png")
                inv_image_path = os.path.join(output_path, 'inverted', f"{base_name}.png")
                rec_image_path = os.path.join(output_path, 'reconstructed', f"{base_name}.png")
                
                #make sure subfolders exists
                os.makedirs(os.path.dirname(original_image_path), exist_ok=True)
                os.makedirs(os.path.dirname(inv_image_path), exist_ok=True)
                os.makedirs(os.path.dirname(rec_image_path), exist_ok=True)

                inverted_PIL = Image.fromarray(inverted)
                inverted_PIL.save(inv_image_path)

                reconstructed_PIL = Image.fromarray(reconstructed)
                reconstructed_PIL.save(rec_image_path)
                #convert original image tensor to pil image before saving
                original_pil = Image.fromarray(original)
                original_pil.save(original_image_path)

def invert_and_reconstruct( loader , output_path):
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    dtype = torch.float16

    inverse_scheduler = DDIMInverseScheduler.from_pretrained('runwayml/stable-diffusion-v1-5', subfolder='scheduler')
    forward_scheduler = DDIMScheduler.from_pretrained('runwayml/stable-diffusion-v1-5', subfolder= 'scheduler')

    pipe = StableDiffusionPipeline.from_pretrained('runwayml/stable-diffusion-v1-5',
                                                    scheduler=inverse_scheduler, #start with inverse sheduler, gets swaped in loop
                                                    safety_checker=None,
                                                    torch_dtype=dtype)
    pipe.to(device)

    #disables tqdm progress bar for generation pipeline, reducing clutter in console
    pipe.set_progress_bar_config(disable=True)  #comment out to see per image progress bar

    #channels_last shoul help speed by using a more efficent format
    pipe.unet.to(memory_format=torch.channels_last)
    pipe.vae.to(memory_format=torch.channels_last)

    #use torch.compile to omptimise pipeline - small speed up for linux systems- does not wok on windows!!
    if hasattr(torch, 'compile'):
        print("Compiling u-net")
        pipe.unet = torch.compile(pipe.unet, mode="max-autotune", fullgraph=True)

    #sett all models in pipe to eval mode
    pipe.vae.eval()
    pipe.text_encoder.eval()
    vae = pipe.vae

    with torch.inference_mode():
        for image_tensors, names, captions in tqdm(loader):
            gpu_image_tensors = image_tensors.to(device=device, dtype=vae.dtype, memory_format=torch.channels_last)

            #make batch of images into batch of latents
            latents = img_to_latents(gpu_image_tensors, vae=vae)

            #calculate the inverted (noisemap) latents
            inv_latents, _ = pipe(prompt = list(captions), 
                                negative_prompt="", 
                                guidance_scale=1.,
                                num_inference_steps=50, 
                                return_dict=False,
                                latents=latents,
                                output_type='latent')
            
            #set pipe to use forward scheduler
            pipe.scheduler = forward_scheduler
            
            #reconstruct img from inv latent 
            reconstructed_tensors = pipe(prompt=list(captions), 
                                        negative_prompt="", 
                                        guidance_scale=1.,
                                        num_inference_steps=50,
                                        latents=inv_latents,
                                        output_type="pt").images

            #reset to inverse shedueler again
            pipe.scheduler = inverse_scheduler

            #offloading tensors and prepaaring for upload
            original_numpy = to_uint8(gpu_image_tensors)
            reconstructed_numpy = to_uint8(reconstructed_tensors)
            inv_latents_numpy = latents_to_img(inv_latents, vae)
            
            
            executor.submit(
                 save_batch,
                 original_numpy,
                 inv_latents_numpy,
                 reconstructed_numpy,
                 names,
                 output_path
            )

            #reset to inverse shedueler again
            pipe.scheduler = inverse_scheduler
    
    print("Waiting for pending saves...")
    executor.shutdown(wait=True)
    print("Done.")