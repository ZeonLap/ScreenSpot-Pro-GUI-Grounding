# TODO: If you are running this code for the first time, follow the instructions from this link to fix the codebase first: https://github.com/OpenGVLab/InternVL/issues/405#issuecomment-2357514453
import json
import os
import re
import tempfile
import base64
from PIL import Image
from io import BytesIO
import torch
import torchvision.transforms as T
from torchvision.transforms.functional import InterpolationMode

from transformers import AutoModel, AutoTokenizer, AutoProcessor
from transformers.generation import GenerationConfig

from transformers import FuyuProcessor, FuyuForCausalLM


# bbox -> point (str)
def bbox_2_point(bbox, dig=2):
    # bbox [left, top, right, bottom]
    point = [(bbox[0]+bbox[2])/2, (bbox[1]+bbox[3])/2]
    point = [f"{item:.2f}" for item in point]
    point_str = "({},{})".format(point[0], point[1])
    return point_str

# bbox -> bbox (str)
def bbox_2_bbox(bbox, dig=2):
    bbox = [f"{item:.2f}" for item in bbox]
    bbox_str = "({},{},{},{})".format(bbox[0], bbox[1], bbox[2], bbox[3])
    return bbox_str

# point (str) -> point
def pred_2_point(s):
    floats = re.findall(r'-?\d+\.?\d*', s)
    floats = [float(num) for num in floats]
    if len(floats) == 2:
        return floats
    elif len(floats) == 4:
        return [(floats[0]+floats[2])/2, (floats[1]+floats[3])/2]
    else:
        return None

# bbox (qwen str) -> bbox
import re

def extract_bbox(s):
    # Extract text between <|box_start|> and <|box_end|> tags
    match = re.search(r'<\|box_start\|\>(.*?)<\|box_end\|\>', s)
    
    if match:
        # Get the text between the tags
        extracted_text = match.group(1)
        
        # Remove parentheses and brackets
        cleaned_text = re.sub(r'[()\[\]]', '', extracted_text)
        
        # Extract four numbers from the cleaned text
        pattern = r"(\d+),\s*(\d+),\s*(\d+),\s*(\d+)"
        numbers = re.findall(pattern, cleaned_text)
        
        if numbers:
            # Return the first match as tuples of integers
            x1, y1, x2, y2 = numbers[0]
            return (int(x1), int(y1)), (int(x2), int(y2))
    
    return None



def image_to_temp_filename(image):
    temp_file = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
    image.save(temp_file.name)
    print(f"Image saved to temporary file: {temp_file.name}")
    return temp_file.name


class FuyuModel():
    def load_model(self, model_name_or_path="adept/fuyu-8b", device="cuda"):
        self.device = device
        self.model = FuyuForCausalLM.from_pretrained(
            model_name_or_path, 
            device_map=device, 
            torch_dtype=torch.bfloat16,
        ).eval()
        self.processor = FuyuProcessor.from_pretrained(model_name_or_path)

        # Setting default generation config
        # self.generation_config = GenerationConfig.from_pretrained("adept/fuyu-8b", trust_remote_code=True).to_dict()
        self.generation_config = dict()
        self.set_generation_config(
            max_length=4096,
            do_sample=False,
            temperature=0.0
        )

    def set_generation_config(self, **kwargs):
        self.generation_config.update(**kwargs)
        self.model.generation_config = GenerationConfig(**self.generation_config)


    def ground_only_positive(self, instruction, image):
        if isinstance(image, str):
            image_path = image
            assert os.path.exists(image_path) and os.path.isfile(image_path), "Invalid input image path."
            image = Image.open(image_path).convert('RGB')
        assert isinstance(image, Image.Image), "Invalid input image."

        prompt_origin = 'When presented with a box, perform OCR to extract text contained within it. If provided with text, generate the corresponding bounding box.\n{}'
        full_prompt = prompt_origin.format(instruction)

        inputs = self.processor(text=full_prompt, images=image, return_tensors="pt").to(self.device)
        generation_output = self.model.generate(**inputs)
        response = self.processor.batch_decode(generation_output, skip_special_tokens=True)[0]
        print(response)

        result_dict = {
            "result": "positive",
            "format": "x1y1x2y2",
            "raw_response": response,
            "bbox": None,
            "point": None
        }

        if '<|box_start|>' in response and '<|box_end|>' in response:
            pred_bbox = extract_bbox(response)
            if pred_bbox is not None:
                (x1, y1), (x2, y2) = pred_bbox
                pred_bbox = [pos / 1000 for pos in [x1, y1, x2, y2]]
                click_point = [(pred_bbox[0] + pred_bbox[2]) / 2, (pred_bbox[1] + pred_bbox[3]) / 2]
                
                result_dict["bbox"] = pred_bbox
                result_dict["point"] = click_point
        else:
            print('---------------')
            print(response)
            click_point = pred_2_point(response)
            result_dict["point"] = click_point  # can be none
        
        return result_dict


    def ground_allow_negative(self, instruction, image):
        if isinstance(image, str):
            image_path = image
            assert os.path.exists(image_path) and os.path.isfile(image_path), "Invalid input image path."
            image = Image.open(image_path).convert('RGB')
        assert isinstance(image, Image.Image), "Invalid input image."

        prompt_origin = 'When presented with a box, perform OCR to extract text contained within it. If provided with text, generate the corresponding bounding box.\n{}'
        full_prompt = prompt_origin.format(instruction)

        inputs = self.processor(text=full_prompt, images=image, return_tensors="pt").to
        generation_output = self.model.generate(**inputs)
        response = self.processor.batch_decode(generation_output, skip_special_tokens=True)[0]
        
        result_dict = {
            "result": None,
            "format": "x1y1x2y2",
            "raw_response": response,
            "bbox": None,
            "point": None
        }

        if '<|box_start|>' in response and '<|box_end|>' in response:
            pred_bbox = extract_bbox(response)
            if pred_bbox is not None:
                (x1, y1), (x2, y2) = pred_bbox
                pred_bbox = [pos / 1000 for pos in [x1, y1, x2, y2]]
                click_point = [(pred_bbox[0] + pred_bbox[2]) / 2, (pred_bbox[1] + pred_bbox[3]) / 2]

                result_dict["bbox"] = pred_bbox
                result_dict["point"] = click_point
        else:
            print('---------------')
            print(response)
            click_point = pred_2_point(response)
            result_dict["point"] = click_point  # can be none

        # set result status
        if result_dict["bbox"] or result_dict["point"]:
            result_status = "positive"
        elif "Target does not exist".lower() in response.lower():
            result_status = "negative"
        else:
            result_status = "wrong_format"
        result_dict["result"] = result_status

        return result_dict

