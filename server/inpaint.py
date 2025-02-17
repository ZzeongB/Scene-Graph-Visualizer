import argparse, os, sys, glob
sys.path.append('.')
from omegaconf import OmegaConf
from PIL import Image
from tqdm import tqdm
import numpy as np
import torch
                
import cv2
import einops
import numpy as np
import random
from omegaconf import OmegaConf

from AnyDoor.cldm.model import create_model, load_state_dict
from AnyDoor.cldm.ddim_hacked import DDIMSampler
from AnyDoor.cldm.hack import disable_verbosity, enable_sliced_attention
from latentDiffusion.datasets.data_utils import * 
from latentDiffusion.main import instantiate_from_config
from server_utils.anydoor_utils import *

def make_batch(image, mask, device):
    image = np.array(Image.open(image).convert("RGB"))
    image = image.astype(np.float32)/255.0
    image = image[None].transpose(0,3,1,2)
    image = torch.from_numpy(image)

    mask = np.array(Image.open(mask).convert("L"))
    mask = mask.astype(np.float32)/255.0
    mask = mask[None,None]
    mask[mask < 0.5] = 0
    mask[mask >= 0.5] = 1
    mask = torch.from_numpy(mask)

    masked_image = (1-mask)*image

    batch = {"image": image, "mask": mask, "masked_image": masked_image}
    for k in batch:
        batch[k] = batch[k].to(device=device)
        batch[k] = batch[k]*2.0-1.0
    return batch

def AnyDoor(background, reference, mask, movement, anydoor_model, anydoor_ddim_sampler):
    """
    background: PIL Image or numpy array of background image
    reference: PIL Image or numpy array of reference image
    mask: PIL Image or numpy array of reference mask (black/white)
    movement: tuple of (x_shift, y_shift) for mask translation
    """
    print("AnyDoor is the best!")
    
    # Convert inputs to numpy arrays if needed
    if isinstance(background, Image.Image):
        background = np.array(background.convert("RGB"))
    if isinstance(reference, Image.Image):
        reference = np.array(reference.convert("RGB"))
    if isinstance(mask, Image.Image):
        mask = np.array(mask.convert("L"))
    
    x_shift, y_shift = movement
    
    # Process masks
    ref_mask = process_binary_mask(mask)
    target_mask = shift_mask(ref_mask, x_shift, y_shift)
    
    # Default parameters
    strength = 1.0
    ddim_steps = 30
    scale = 4.5
    seed = -1
    enable_shape_control = True
    
    if seed != -1:
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)
    
    # Run inference with clone() for tensors or copy() for numpy arrays
    synthesis = inference_single_image(
        reference if isinstance(reference, np.ndarray) else reference.clone(),
        ref_mask if isinstance(ref_mask, np.ndarray) else ref_mask.clone(),
        background if isinstance(background, np.ndarray) else background.clone(),
        target_mask if isinstance(target_mask, np.ndarray) else target_mask.clone(),
        strength=1.0,
        ddim_steps=30,
        scale=4.5,
        seed=-1,
        enable_shape_control=True,
        anydoor_model=anydoor_model,
        anydoor_ddim_sampler=anydoor_ddim_sampler,
    )
    
    
    return synthesis


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--indir",
        type=str,
        nargs="?",
        help="dir containing image-mask pairs (`example.png` and `example_mask.png`)",
    )
    parser.add_argument(
        "--outdir",
        type=str,
        nargs="?",
        help="dir to write results to",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=50,
        help="number of ddim sampling steps",
    )
    opt = parser.parse_args()

    masks = sorted(glob.glob(os.path.join(opt.indir, "*_mask.png")))
    images = [x.replace("_mask.png", ".png") for x in masks]
    print(f"Found {len(masks)} inputs.")

    config = OmegaConf.load("latentDiffusion/models/ldm/inpainting_big/config.yaml")
    latent_model = instantiate_from_config(config.model)
    latent_model.load_state_dict(torch.load("latentDiffusion/models/ldm/inpainting_big/last.ckpt")["state_dict"],
                          strict=False)

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    latent_model = latent_model.to(device)
    from latentDiffusion.ldm.models.diffusion.ddim import DDIMSampler

    latent_sampler = DDIMSampler(latent_model)
    
    # Initialize AnyDoor model
    cv2.setNumThreads(0)
    cv2.ocl.setUseOpenCL(False)
    
    save_memory = False
    disable_verbosity()
    if save_memory:
        enable_sliced_attention()
    
    anydoor_config = OmegaConf.load('./configs/anydoor/demo.yaml')
    anydoor_model_ckpt = anydoor_config.pretrained_model
    anydoor_model_config = anydoor_config.config_file
    
    anydoor_model = create_model(anydoor_model_config).cpu()
    anydoor_model.load_state_dict(load_state_dict(anydoor_model_ckpt, location='cuda'))
    anydoor_model = anydoor_model.cuda()
    from AnyDoor.ldm.models.diffusion.ddim import DDIMSampler

    anydoor_ddim_sampler = DDIMSampler(anydoor_model)

    os.makedirs(opt.outdir, exist_ok=True)
    with torch.no_grad():
        with latent_model.ema_scope():
            for image, mask in tqdm(zip(images, masks)):
                outpath = os.path.join(opt.outdir, os.path.split(image)[1])
                outpath_ = os.path.join(opt.outdir, os.path.split(image)[1]+"_anydoor.png")

                batch = make_batch(image, mask, device=device)

                # encode masked image and concat downsampled mask
                c = latent_model.cond_stage_model.encode(batch["masked_image"])
                cc = torch.nn.functional.interpolate(batch["mask"],
                                                     size=c.shape[-2:])
                c = torch.cat((c, cc), dim=1)

                shape = (c.shape[1]-1,)+c.shape[2:]
                samples_ddim, _ = latent_sampler.sample(S=opt.steps,
                                                 conditioning=c,
                                                 batch_size=c.shape[0],
                                                 shape=shape,
                                                 verbose=False)
                x_samples_ddim = latent_model.decode_first_stage(samples_ddim)

                image = torch.clamp((batch["image"]+1.0)/2.0,
                                    min=0.0, max=1.0)
                mask = torch.clamp((batch["mask"]+1.0)/2.0,
                                   min=0.0, max=1.0)
                
                # CPU로 이동하고 numpy로 변환하여 dilation 적용
                mask_np = mask.cpu().numpy().squeeze()
                kernel_size = 21  # 10픽셀 양방향으로 확장 (홀수여야 함)
                kernel = np.ones((kernel_size, kernel_size), np.uint8)
                dilated_mask = cv2.dilate(mask_np, kernel, iterations=1)

                # 다시 torch tensor로 변환
                mask = torch.from_numpy(dilated_mask).float().to(mask.device)
                if len(mask.shape) == 2:
                    mask = mask.unsqueeze(0).unsqueeze(0)  # 다시 [1,1,H,W] 형태로
    
                predicted_image = torch.clamp((x_samples_ddim+1.0)/2.0,
                                              min=0.0, max=1.0)

                inpainted = (1-mask)*image+mask*predicted_image
                inpainted = inpainted.cpu().numpy().transpose(0,2,3,1)[0]*255
                Image.fromarray(inpainted.astype(np.uint8)).save(outpath)
                
                # AnyDoor 입력을 위한 이미지와 마스크 전처리
                reference_image = batch["image"].cpu().numpy().transpose(0,2,3,1)[0]  # [H,W,3]로 변환
                reference_image = ((reference_image + 1.0) * 127.5).astype(np.uint8)  # [-1,1] -> [0,255]

                # AnyDoor 실행
                synthesis = AnyDoor(inpainted.astype(np.uint8), reference_image, mask, (100, 0), anydoor_model, anydoor_ddim_sampler)
                Image.fromarray(synthesis.astype(np.uint8)).save(outpath_)

