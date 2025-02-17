
# image_path: "./demo/images/20250217_115048_944691.png",
# mask_path: "./demo/masks/20250217_115048_944691",

# 위 두 path에 잇는 이미지들을 불러와 bit64로 인코딩

import base64
import os
import json

def image_to_base64(image_path):
    with open(image_path, "rb") as img_file:
        return base64.b64encode(img_file.read()).decode('utf-8')
    
def mask_to_base64(mask_path):
    masks = []
    for root, dirs, files in os.walk(mask_path):
        for file in files:
            with open(os.path.join(root, file), "rb") as mask_file:
                masks.append({"mask": base64.b64encode(mask_file.read()).decode('utf-8'), "name": file.split("_")[0]})
                
    
    return masks
    
image_path = "./demo/images/20250217_115048_944691.png"
mask_path = "./demo/masks/20250217_115048_944691" # this is a directory

image_base64 = image_to_base64(image_path)
mask_base64 = mask_to_base64(mask_path)

# JavaScript 형식으로 export 문 작성
txt = "export const INIT_IMAGE = '" + image_base64 + "';\n\n"
txt += "export const INIT_MASKS = " + json.dumps(mask_base64, indent=2) + ";\n"

# 파일 저장
with open("/home/jipark/Scene-Graph-Visualizer/src/constant/initialConstants.js", "w") as f:
    f.write(txt)