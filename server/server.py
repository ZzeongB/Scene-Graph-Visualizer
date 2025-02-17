from flask import Flask, request, jsonify, Response, current_app
from diffusers import StableDiffusionXLPipeline, StableDiffusionInpaintPipeline
import torch.utils.checkpoint
from LAION.sgEncoderTraining.datasets.laion_dataset import build_laion_loaders #####
from configs.configs_laion import parse_args #####
from transformers import AutoTokenizer, PretrainedConfig
from LAION.sgEncoderTraining.sgEncoder.create_sg_encoder import create_model_and_transforms
import os
from flask_cors import CORS
from PIL import Image
import io
import base64
import random
import numpy as np
import torch
import time
from datetime import datetime
from server_utils.graph_utils import scene_graph_to_triples, generate_prompt
from utils.Segment.seg_utils import construct_node_masks

def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# Set the seed
# set_random_seed(42)

from diffusers import (
    AutoencoderKL,
    UNet2DConditionModel,
)

args = parse_args()

device = torch.device("cuda")

def import_model_class_from_model_name_or_path(
    pretrained_model_name_or_path: str,  subfolder: str = "text_encoder"
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


vae = AutoencoderKL.from_pretrained(
        args.stable_diffusion_checkpoint,
        subfolder="vae",
        variant="fp16",
        cache_dir=args.cache_dir
    ).to(device)

tokenizer_one = AutoTokenizer.from_pretrained(
        args.stable_diffusion_checkpoint,
        subfolder="tokenizer",
        use_fast=False,
        cache_dir=args.cache_dir
    )

tokenizer_two = AutoTokenizer.from_pretrained(
        args.stable_diffusion_checkpoint,
        subfolder="tokenizer_2",
        use_fast=False,
        cache_dir=args.cache_dir
    )

text_encoder_cls_one = import_model_class_from_model_name_or_path(
        args.stable_diffusion_checkpoint
    )

text_encoder_cls_two = import_model_class_from_model_name_or_path(
        args.stable_diffusion_checkpoint, subfolder="text_encoder_2"
    )


text_encoder_one = text_encoder_cls_one.from_pretrained(
        args.stable_diffusion_checkpoint, subfolder="text_encoder", variant="fp16",cache_dir=args.cache_dir).to(device)
text_encoder_two = text_encoder_cls_two.from_pretrained(
        args.stable_diffusion_checkpoint, subfolder="text_encoder_2", variant="fp16",cache_dir=args.cache_dir).to(device)

unet = UNet2DConditionModel.from_pretrained(
    args.stable_diffusion_checkpoint, subfolder="unet", variant="fp16", cache_dir=args.cache_dir).to(device)


gen_pipeline = StableDiffusionXLPipeline.from_pretrained(
    args.stable_diffusion_checkpoint,
    vae=vae,
    text_encoder=text_encoder_one,
    text_encoder_2=text_encoder_two,
    unet=unet,
    torch_dtype='fp16',
    cache_dir=args.cache_dir
)

gen_model = create_model_and_transforms(
        args,
        text_encoders=[text_encoder_one, text_encoder_two],
        tokenizers =[tokenizer_one,tokenizer_two],
        model_config_json=args.model_config_json,
        precision=args.precision,
        device=device,
        force_quick_gelu=args.force_quick_gelu,
        pretrained_image=args.pretrained_image,
    ).to(device)

checkpoint = torch.load("./checkpoints/LAION_100.pt", map_location=device)

edit_pipeline = StableDiffusionInpaintPipeline.from_pretrained(
    "stabilityai/stable-diffusion-2-inpainting",
    torch_dtype=torch.float16,
)
edit_pipeline.to("cuda")

app = Flask(__name__)
CORS(app, resources={r"/generate": {"origins": "http://localhost:3000"}}, supports_credentials=True)
CORS(app, resources={r"/edit": {"origins": "http://localhost:3000"}}, supports_credentials=True)

@app.route('/generate', methods=['POST'])
def generate():
    try:
        data = request.get_json()
        scene_graph = data.get('scene_graph')
        size = data.get('size')
        
        print("Sucessfully received data")
        print("Scene Graphs", scene_graph)

        if not scene_graph or not size:
            return jsonify({"error": "Missing scene_graph or size"}), 400

        triples, global_ids, isolated_items = scene_graph_to_triples(scene_graph)
        all_triples = [triples]
        all_global_ids = [global_ids]
        all_isolated_items = [isolated_items]

        print("Sucessfully converted scene graph to triples\n")
        print("\tTriples", all_triples)
        print("\tIsolated Items", all_isolated_items)
        
        prompt_embeds, pooled_embeds = gen_model(all_triples, all_isolated_items, all_global_ids)

        print("Sucessfully generated embeddings")
        
        img = gen_pipeline(
            prompt_embeds=prompt_embeds,
            num_inference_steps=40,
            pooled_prompt_embeds=pooled_embeds,
            width=size,
            height=size,
        ).images[0]
        
        print("Sucessfully generated image")
        
        # Ensure the directory exists
        output_dir = "./demo/images"
        os.makedirs(output_dir, exist_ok=True)

        # Save the image with a unique name based on the current timestamp
        image_name = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        image_path = os.path.join(output_dir, f"{image_name}.png")
        img.save(image_path)

        print("Sucessfully saved image")
        
        basic_scene_graph = {
            "objects": [node['name'] for node in scene_graph['objects']],  # extract object names from scene_graph
            "tuples": [[r['source'], r['relation'], r['target']] for r in scene_graph['relationships']]  # extract tuples from scene_graph
        }
        
        print("Basic Scene Graph", basic_scene_graph)
            
        
        mask_output_dir = f"./demo/masks/{image_name}"
        os.makedirs(mask_output_dir, exist_ok=True)
        sg_with_bbox = construct_node_masks(image_path, basic_scene_graph, mask_output_dir)
        
        print("Sucessfully generated masks")

        buffered = io.BytesIO()
        img.save(buffered, format="PNG")
        img_str = base64.b64encode(buffered.getvalue()).decode('utf-8')
        
        masks = []
        for f in os.listdir(mask_output_dir):
            if f.endswith("_mask.png") or f.endswith("_mask.jpg"):
                with open(f"{mask_output_dir}/{f}", "rb") as img_file:
                    mask_str = base64.b64encode(img_file.read()).decode('utf-8')
                    masks.append({"mask": mask_str, "name": f.split("_mask")[0]})
        print("Sucessfully converted masks to base64")

        return jsonify({'image': img_str, 'original_sg': scene_graph, 'image_path': image_path, 'mask_path': mask_output_dir, "masks": masks}), 200

    except Exception as e:
        # 예외가 발생한 경우 로그와 함께 오류 메시지를 반환
        print(f"Error: {str(e)}")  # 콘솔에 에러 메시지 출력
        return jsonify({"error": str(e)}), 500
   
@app.route('/edit', methods=['POST'])
def edit():
    try:
        data = request.get_json()
        scene_graph = data.get('scene_graph')
        image_metadata = data.get('image_metadata') # {image_path, mask_path, original_sg}
        graph_changes = data.get('graph_changes')
        
        if not scene_graph or not image_metadata or not graph_changes:
            return jsonify({"error": "Missing scene_graph, image_path or mask_path"}), 400
        print("Sucessfully received data")
        print("Scene Graphs", scene_graph)
        print("Image Metadata", image_metadata)
        print("Graph Changes", graph_changes)
    
        prompt_and_mask = generate_prompt(image_metadata['original_sg'], scene_graph, graph_changes)
        print("Sucessfully generated prompt and mask")
        
        original_image = Image.open(image_metadata['image_path']).convert("RGB")
        # Apply the prompt and mask to the image
        for (prompt, mask_image_path) in prompt_and_mask:
            print("\tPrompt and mask", prompt, mask_image_path)
            mask_image = Image.open(f"{image_metadata['mask_path']}/{mask_image_path}_mask.png").convert("RGB") 
            result_image = edit_pipeline(prompt=prompt, image=original_image, mask_image=mask_image, guidance_scale=8.5).images[0]
            original_image = result_image
            
        print("Sucessfully edited image")
        
        # Ensure the directory exists
        output_dir = "./demo/images"
        os.makedirs(output_dir, exist_ok=True)

        # Save the image with a unique name based on the current timestamp
        image_name = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        image_path = os.path.join(output_dir, f"{image_name}.png")
        result_image.save(image_path)

        print("Sucessfully saved image")
        
        basic_scene_graph = {
            "objects": [node['name'] for node in scene_graph['objects']],  # extract object names from scene_graph
            "tuples": [[r['source'], r['relation'], r['target']] for r in scene_graph['relationships']]  # extract tuples from scene_graph
        }
        
        print("Basic Scene Graph", basic_scene_graph)
            
        mask_output_dir = f"./demo/masks/{image_name}"
        os.makedirs(mask_output_dir, exist_ok=True)
        sg_with_bbox = construct_node_masks(image_path, basic_scene_graph, mask_output_dir)
        
        print("Sucessfully generated masks")

        buffered = io.BytesIO()
        result_image.save(buffered, format="PNG")
        img_str = base64.b64encode(buffered.getvalue()).decode('utf-8')
        
        masks = []
        for f in os.listdir(mask_output_dir):
            if f.endswith("_mask.png") or f.endswith("_mask.jpg"):
                with open(f"{mask_output_dir}/{f}", "rb") as img_file:
                    mask_str = base64.b64encode(img_file.read()).decode('utf-8')
                    masks.append({"mask": mask_str, "name": f.split("_mask")[0]})

        print("Sucessfully converted masks to base64")
                    
        return jsonify({'image': img_str, 'original_sg': scene_graph, 'image_path': image_path, 'mask_path': mask_output_dir, 'masks': masks}), 200

    
    except Exception as e:
        print(f"Error: {str(e)}")
        return jsonify({"error": str(e)}), 500
        
        
if __name__ == '__main__':
    app.run(debug=True, use_reloader=False)
