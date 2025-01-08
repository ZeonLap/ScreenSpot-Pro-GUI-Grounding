import base64
import re

from openai import OpenAI
from io import BytesIO
from PIL import Image

NUM_SECONDS_TO_SLEEP = 5


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


class Aguvis72BModel():

    def __init__(self, base_url="http://10.77.245.212:8908/v1"):
        self.base_url = base_url
        self.client = OpenAI(base_url=base_url, api_key="abc-123")
        self.grounding_system_message = "You are a GUI agent. You are given a task and a screenshot of the screen. You need to perform a series of pyautogui actions to complete the task."
        self.user_message = """Please complete the following tasks by clicking using `pyautogui.click`:\n"""

    def load_model(self, model_name_or_path="Qwen/Qwen2-VL-7B-Instruct"):
        pass

    def set_generation_config(self, **kwargs):
        pass

    def ground_only_positive(self, instruction, image):

        def encode_image(image):
            if not isinstance(image, Image.Image):
                image = Image.open(image)
            output_buffer = BytesIO()
            image.save(output_buffer, format="PNG")
            byte_data = output_buffer.getvalue()
            base64_str = base64.b64encode(byte_data).decode("utf-8")
            return base64_str

        image = encode_image(image)
        messages = [
            {
                "role": "system",
                "content": self.grounding_system_message
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/png;base64,{image}"},
                    },
                    {"type": "text", "text": self.user_message + instruction},
                ],
            }
        ]
        payload = {
            "messages": messages,
            "max_tokens": 1024,
            "temperature": 0.0
        }

        for attempt in range(5):
            try:
                completion = self.client.chat.completions.create(
                    model="aguvis-72b-500",
                    messages=payload["messages"],
                    temperature=payload["temperature"],
                    max_tokens=payload["max_tokens"],
                )
                response_text = completion.choices[0].message.content
                break  # If successful, break out of the loop

            except Exception as e:
                print(f"Attempt {attempt + 1} failed with error: {str(e)}.")
                if attempt <= 5:
                    time.sleep(NUM_SECONDS_TO_SLEEP)
                else:  # If this was the last attempt, log and return empty string
                    print(f"All 5 attempts failed. Last error message: {str(e)}.\n")
                    response_text = ""

        result_dict = {
            "result": "positive",
            "format": "x1y1x2y2",
            "raw_response": response_text,
            "bbox": None,
            "point": None
        }

        click_point = parse_pyautogui(response_text)
        result_dict["point"] = click_point
        return result_dict

    def ground_allow_negative(self, instruction, image):
        raise NotImplementedError
