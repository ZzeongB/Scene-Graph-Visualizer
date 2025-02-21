import gc
import torch
from diffusers import (
    AutoencoderKL,
    UNet2DConditionModel,
    StableDiffusionXLPipeline, StableDiffusionInpaintPipeline

)
from transformers import AutoTokenizer, PretrainedConfig
from LAION.sgEncoderTraining.sgEncoder.create_sg_encoder import create_model_and_transforms
import torch.utils.checkpoint
from omegaconf import OmegaConf
import time
from diffusers import AutoPipelineForInpainting, StableDiffusionInpaintPipeline

class ModelSet:
    def __init__(self, name, device="cuda"):
        self.name = name
        self.device = device
        self.models = {}
        
    def unload(self):
        for model_name in list(self.models.keys()):
            del self.models[model_name]
        torch.cuda.empty_cache()
        
    def get_model(self, model_name):
        return self.models.get(model_name)


class GenerationModelSet(ModelSet):
    def __init__(self, device="cuda"):
        super().__init__("generation", device)
        from configs.configs_laion import parse_args #####
        args = parse_args()
        self.args = args
    
    
    def import_model_class_from_model_name_or_path(
        self, pretrained_model_name_or_path: str,  subfolder: str = "text_encoder"
    ):
        text_encoder_config = PretrainedConfig.from_pretrained(
            pretrained_model_name_or_path, subfolder=subfolder,
        )
        model_class = text_encoder_config.architectures[0]

        if model_class == "CLIPTextModel":
            from transformers import CLIPTextModel

            return CLIPTextModel
        elif model_class == "CLIPTextModelWithProjection":
            from transformers import CLIPTextModelWithProjection

            return CLIPTextModelWithProjection
        else:
            raise ValueError(f"{model_class} is not supported.")

        
    def load(self):
        print("Loading generation models...")
        start = time.time()

        # VAE
        vae = AutoencoderKL.from_pretrained(
            self.args.stable_diffusion_checkpoint,
            subfolder="vae",
            variant="fp16",
            cache_dir=self.args.cache_dir
        ).to(self.device)
        
        # Tokenizers
        tokenizer_one = AutoTokenizer.from_pretrained(
            self.args.stable_diffusion_checkpoint,
            subfolder="tokenizer",
            use_fast=False,
            cache_dir=self.args.cache_dir
        )
        tokenizer_two = AutoTokenizer.from_pretrained(
            self.args.stable_diffusion_checkpoint,
            subfolder="tokenizer_2",
            use_fast=False,
            cache_dir=self.args.cache_dir
        )
        
        # Text Encoders
        text_encoder_cls_one = self.import_model_class_from_model_name_or_path(
            self.args.stable_diffusion_checkpoint
        )
        text_encoder_cls_two = self.import_model_class_from_model_name_or_path(
            self.args.stable_diffusion_checkpoint, 
            subfolder="text_encoder_2"
        )
        
        text_encoder_one = text_encoder_cls_one.from_pretrained(
            self.args.stable_diffusion_checkpoint, 
            subfolder="text_encoder", 
            variant="fp16",
            cache_dir=self.args.cache_dir
        ).to(self.device)
        
        text_encoder_two = text_encoder_cls_two.from_pretrained(
            self.args.stable_diffusion_checkpoint, 
            subfolder="text_encoder_2", 
            variant="fp16",
            cache_dir=self.args.cache_dir
        ).to(self.device)
        
        # UNet
        unet = UNet2DConditionModel.from_pretrained(
            self.args.stable_diffusion_checkpoint, 
            subfolder="unet", 
            variant="fp16", 
            cache_dir=self.args.cache_dir
        ).to(self.device)
        
        # Pipelines
        gen_pipeline = StableDiffusionXLPipeline.from_pretrained(
            self.args.stable_diffusion_checkpoint,
            vae=vae,
            text_encoder=text_encoder_one,
            text_encoder_2=text_encoder_two,
            unet=unet,
            torch_dtype='fp16',
            cache_dir=self.args.cache_dir
        )
        
        gen_model = create_model_and_transforms(
            self.args,
            text_encoders=[text_encoder_one, text_encoder_two],
            tokenizers=[tokenizer_one, tokenizer_two],
            model_config_json=self.args.model_config_json,
            precision=self.args.precision,
            device=self.device,
            force_quick_gelu=self.args.force_quick_gelu,
            pretrained_image=self.args.pretrained_image,
        ).to(self.device)
        
        checkpoint = torch.load("/content/drive/MyDrive/SG-server/LAION_100.pt", map_location=self.device)
        gen_model.load_state_dict(checkpoint['state_dict'])
        print("Generation models loaded in", time.time() - start, "seconds.")
        
        self.models = {
            "gen_pipeline": gen_pipeline,
            "gen_model": gen_model,
            "vae": vae,
            "text_encoder_one": text_encoder_one,
            "text_encoder_two": text_encoder_two,
            "unet": unet,
            "tokenizer_one": tokenizer_one,
            "tokenizer_two": tokenizer_two
        }
        
        print("Generation models loaded.")


class MovementModelSet(ModelSet):
    def __init__(self, device="cuda"):
        super().__init__("movement", device)
        
    def load(self, use_sdxl=True):
        from AnyDoor.cldm.model import create_model, load_state_dict
        from latentDiffusion.main import instantiate_from_config

        # # Latent Diffusion models
        # from latentDiffusion.ldm.models.diffusion.ddim import DDIMSampler
        
        # config = OmegaConf.load("latentDiffusion/models/ldm/inpainting_big/config.yaml")
        # latent_model = instantiate_from_config(config.model)
        # latent_model.load_state_dict(
        #     torch.load("latentDiffusion/models/ldm/inpainting_big/last.ckpt")["state_dict"],
        #     strict=False
        # )
        # latent_model = latent_model.to(self.device)
        # latent_sampler = DDIMSampler(latent_model)
        
        # Use StableDiffusion2 and StableDiffusionXL. If specified to use one, do not use SDXL
        print("Loading Stable Diffusion models...")
        pipe = StableDiffusionInpaintPipeline.from_pretrained(
            "stabilityai/stable-diffusion-2-inpainting",
            torch_dtype=torch.float16,
        )
        pipe.to(self.device)
        
        if(use_sdxl):
            pipe_xl = AutoPipelineForInpainting.from_pretrained(
                "diffusers/stable-diffusion-xl-1.0-inpainting-0.1",
                torch_dtype=torch.float16,
                variant="fp16"
            ).to(self.device)
        else:
            pipe_xl = None
        
        # AnyDoor models
        print("Loading AnyDoor models...")
        from AnyDoor.ldm.models.diffusion.ddim import DDIMSampler

        anydoor_config = OmegaConf.load('./configs/anydoor/demo.yaml')
        anydoor_model = create_model(anydoor_config.config_file).cpu()
        anydoor_model.load_state_dict(
            load_state_dict(anydoor_config.pretrained_model, location='cuda')
        )
        anydoor_model = anydoor_model.cuda()
        anydoor_ddim_sampler = DDIMSampler(anydoor_model)
        
        self.models = {
            "SD2_pipe": pipe,
            "SDXL_pipe": pipe_xl,
            "anydoor_model": anydoor_model,
            "anydoor_sampler": anydoor_ddim_sampler
        }


class ModelManager:
    def __init__(self, device="cuda"):
        self.device = device
        self.sets = {}
        
    def load_generation_models(self):
        gen_set = GenerationModelSet(self.device)
        gen_set.load()
        self.sets["generation"] = gen_set
        return gen_set
        
    def load_movement_models(self):
        print("Loading movement models...")
        move_set = MovementModelSet(self.device)
        move_set.load(use_sdxl=False)
        self.sets["movement"] = move_set
        return move_set
    
    def unload_set(self, set_name):
        if set_name in self.sets:
            self.sets[set_name].unload()
            del self.sets[set_name]
            
    def unload_all(self):
        for set_name in list(self.sets.keys()):
            self.unload_set(set_name)
        gc.collect()
