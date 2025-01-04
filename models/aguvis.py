from transformers import Qwen2VLForConditionalGeneration, Qwen2VLProcessor
from transformers.generation import GenerationConfig
import re
import os
from PIL import Image
import tempfile
from qwen_vl_utils import process_vision_info


def parse_pyautogui(pred):
    pattern = r'pyautogui\.click\((?:x=)?(\d+\.\d+)(?:,\s*y=|,\s*)(\d+\.\d+)\)'

    match = re.search(pattern, pred)
    if match:
        x, y = match.groups()
        return [float(x), float(y)]
    else:
        print(f"Failed to parse pyautogui: {pred}")
        return [0, 0]


def image_to_temp_filename(image):
    temp_file = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
    image.save(temp_file.name)
    print(f"Image saved to temporary file: {temp_file.name}")
    return temp_file.name


class AguvisModel():

    def __init__(self, device="cuda"):
        self.device = device
        self.grounding_system_message = "You are a GUI agent. You are given a task and a screenshot of the screen. You need to perform a series of pyautogui actions to complete the task."
        self.user_message = """Please complete the following tasks by clicking using `pyautogui.click`:\n"""

    def load_model(self, model_name_or_path="Qwen/Qwen2-VL-7B-Instruct"):
        self.model = Qwen2VLForConditionalGeneration.from_pretrained(model_name_or_path)
        self.processor = Qwen2VLProcessor.from_pretrained(model_name_or_path)
        self.tokenizer = self.processor.tokenizer
        self.model.to(self.device)
        self.model.tie_weights()

    def set_generation_config(self, **kwargs):
        pass

    def ground_only_positive(self, instruction, image):
        if not isinstance(image, str):
            assert isinstance(image, Image.Image)
            image_path = image_to_temp_filename(image)
        else:
            image_path = image
        assert os.path.exists(image_path) and os.path.isfile(image_path), "Invalid input image path."

        messages = [
            {
                "role": "system",
                "content": self.grounding_system_message
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "image": image_path,
                    },
                    {"type": "text", "text": self.user_message + instruction},
                ],
            }
        ]
        # Preparation for inference
        text_input = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = self.processor(
            text=[text_input],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )
        inputs = inputs.to(self.device)

        cont = self.model.generate(**inputs, temperature=0.0, max_new_tokens=1024)

        cont_toks = cont.tolist()[0][len(inputs.input_ids[0]) :]
        text_outputs = self.tokenizer.decode(cont_toks, skip_special_tokens=True).strip()

        result_dict = {
            "result": "positive",
            "format": "x1y1x2y2",
            "raw_response": text_outputs,
            "bbox": None,
            "point": None
        }

        click_point = parse_pyautogui(text_outputs)
        result_dict["point"] = click_point
        return result_dict

    def ground_allow_negative(self, instruction, image):
        raise NotImplementedError
