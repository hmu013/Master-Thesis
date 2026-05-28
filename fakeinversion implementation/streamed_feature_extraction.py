import webdataset as wds
import torch
import os
import albumentations as A
import torchvision.transforms as transforms
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm
from diffusers import StableDiffusionPipeline, DDIMInverseScheduler, AutoencoderKL, DDIMScheduler
from huggingface_hub import HfApi

executor = ThreadPoolExecutor(max_workers=4)

#set for better performance
torch.set_grad_enabled(False)
torch.backends.cudnn.benchmark = True

UPLOAD_REPO = "hmu013/SynRIS-extracted-features" # to upload to
URLS = "https://huggingface.co/datasets/hmu013/SynRIS-captioned/resolve/main/{00067..00089}.tar"
START_SHARD = 67#change the startshard of the sink writer if you are resuming. i.e. not starting from shard 0

#---------------------Helpers-------------------------------------

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

#------------- to save and upload results ---------------------

api = HfApi()
MAX_SHARDS = 300
shards_written = 0

def save_batch(batch_list):
    for sample in batch_list:
        sink.write(sample)

def upload_and_cleanup(shard_path):
    filename = os.path.basename(shard_path)
    try:
        api.upload_file(
            path_or_fileobj=shard_path,
            path_in_repo=filename,
            repo_id=UPLOAD_REPO,
            repo_type="dataset",
        )
        print(f"uploaded: {filename}")
        os.remove(shard_path) 
    except Exception as e:
        print(f"Failed {filename}. Keeping on disk. Error: {e}")

def on_shard_close(shard_name):
    global shards_written
    shards_written += 1
    print(f"Shard {shards_written}/{MAX_SHARDS} closed: {shard_name}")
    executor.submit(upload_and_cleanup, shard_name)

sink = wds.ShardWriter(
    "%05d.tar",
    maxcount=1000,
    post=on_shard_close,
    start_shard=START_SHARD
)

#-------------------Feature extraction---------------------------

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
    
    # Warmup: compile happens fist time pipe si called, if not done here connection to hf will time 
    # out leading to dropped shards while waitong for benchmarking.
    warmup_batch_size = 40 
    dummy_latent = torch.randn((warmup_batch_size, 4, 64, 64), device=device, dtype=dtype)
    dummy_prompt = ["a photo of a cat"] * warmup_batch_size
    _ = pipe(prompt=dummy_prompt, latents=dummy_latent, num_inference_steps=2) 
    print("warmup complete")

#sett all models in pipe to eval mode
pipe.vae.eval()
pipe.text_encoder.eval()
vae = pipe.vae

#----------------Loading WebDataset------------------------------------------------
image_transform = A.Compose([
    A.SmallestMaxSize(512), 
    A.CenterCrop(512, 512),
])  #standard method from resnet to rezise images and keep aspect ratio - not needed for the processed datasets on hf
    #should be used on any other data to keep correct aspecratio and force 512x512 for inversion pipeline

trainset_transform =  transforms.ToTensor() #only transform needed for trainset that has been preprocessed and uploded tp hf

dataset = (
    wds.WebDataset(
        urls= URLS,  
        handler=wds.warn_and_continue, 
        resampled=False, 
        shardshuffle=False,
        )
    .decode("pil")
    .select(lambda x: "png" in x and "json" in x) #make sure all instnaces have both image and caption
    .to_tuple("png", "json", "__key__")
    .map_tuple(
        lambda i: trainset_transform(i),
        lambda j: j["caption"],
        lambda k: k,           
    )
)

loader = wds.WebLoader(
    dataset,
    batch_size=40, #scale up for rented 5090 instances!
    shuffle = False,
    num_workers=1, #we use one worker to easily resume progress. Multiple workers will pull from multiple shards and does not ensure 50/50 split
    #slight loss in efficency but key for stability and letting us scae to multiple instances.
    pin_memory=True,
    prefetch_factor = 2,
    persistent_workers = True,
)

#------------------------------Feature extraction loop -----------------------------------------------

#loop through batches
with torch.inference_mode():
    for image_tensors, captions, keys in tqdm(loader):
    
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

        for i in range(len(keys)):
            sample = {
                "__key__": keys[i],
                "json": {"caption" : captions[i]},
                "org.png": original_numpy[i],
                "inv.png": inv_latents_numpy[i], 
                "rec.png": reconstructed_numpy[i],
            }
            sink.write(sample)

print("Waiting for pending saves...")
sink.close()
executor.shutdown(wait=True)
print("Done.") 