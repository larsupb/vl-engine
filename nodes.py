import base64
import io

import requests
import numpy as np
from PIL import Image


class EndpointTypes:
    OpenAI = "OpenAI"


def tensor_to_pil(image_tensor, batch_index=0) -> Image:
    # Convert tensor of shape [batch, height, width, channels] at the batch_index to PIL Image
    image_tensor = image_tensor[batch_index].unsqueeze(0)
    i = 255.0 * image_tensor.cpu().numpy()
    img = Image.fromarray(np.clip(i, 0, 255).astype(np.uint8).squeeze())
    return img


class APIEndpoint:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "endpoint_type": (["OpenAI"], {"default": "OpenAI", "tooltip": "Specify the type of the API endpoint."}),
                "endpoint_url": ("STRING", {"default": "", "multiline": False}),
                "api_key": ("STRING", {"default": ""}),
            },
        }

    RETURN_TYPES = ("API_ENDPOINT", "STRING")
    RETURN_NAMES = ("API_ENDPOINT", "MODELS")
    FUNCTION = "get_endpoint"
    CATEGORY = "VL-Engine"
    DESCRIPTION = "Specifies VL Engine API Endpoint"
    OUTPUT_NODE = True

    def get_endpoint(self, endpoint_type, endpoint_url: str, api_key: str) -> tuple[dict, str]:
        """
        Get API endpoint details.

        Returns dictionary with URL, type, and API key and list of supported models.
        """
        settings = {
            "endpoint_type": endpoint_type,
            "endpoint_url": endpoint_url,
            "api_key": api_key,
        }
        models = []
        if endpoint_type == EndpointTypes.OpenAI:
            # ask endpoint /api/tags for supported models
            response = requests.get(
                f"{endpoint_url}/api/tags",
                headers={"Authorization": f"Bearer {api_key}"},
            )
            if response.status_code == 200:
                data = response.json()
                models = data.get("models", [])
                print(models)
            else:
                print(f"Error fetching models: {response.status_code} - {response.text}")

        model_names = [model["name"] for model in models]
        return settings, "\n".join(model_names),


class ImageDescriptor:
    def __init__(self):
        self.model_checkpoint = None
        self.processor = None
        self.model = None

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "text": ("STRING", {"default": "", "multiline": True}),
                "model": ("STRING", {"default": "qwen2.5-vl"}),
                "api_endpoint": ("API_ENDPOINT",),
                "temperature": (
                    "FLOAT",
                    {"default": 0.7, "min": 0, "max": 1, "step": 0.1},
                ),
                "max_new_tokens": (
                    "INT",
                    {"default": 512, "min": 128, "max": 2048, "step": 1},
                ),
                "seed": ("INT", {"default": -1}),
            },
            "optional": {
                "image1": ("IMAGE",),
                "image2": ("IMAGE",),
                "image3": ("IMAGE",),
                "video_path": ("STRING", {"default": ""}),
            },
        }

    RETURN_TYPES = ("STRING",)
    FUNCTION = "inference"
    CATEGORY = "VL-Engine"

    def inference(self, text: str, model: str, api_endpoint: dict,
                  temperature: float, max_new_tokens: int, seed: int, image1=None, image2=None, image3=None,
                  video_path="") -> tuple[str]:
        """
        Generate image description using VL Engine API.

        Returns generated description as a string.
        """
        api_type = api_endpoint["endpoint_type"]
        url = api_endpoint["endpoint_url"]
        api_key = api_endpoint["api_key"]

        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }

        payload = {
            "model": model,
            "prompt": text,
            "stream": False,
            "temperature": temperature,
            "max_new_tokens": max_new_tokens,
            "seed": seed,
        }

        image_list = [image1, image2, image3]
        # Remove None images and encode to base64
        image_list = [img for img in image_list if img is not None]
        # Convert image tensors to PIL images and then to base64
        image_list = [tensor_to_pil(img) for img in image_list]

        image_list_encoded = []
        for idx, img in enumerate(image_list):
            if img is not None:
                if img.mode not in ("RGB", "RGBA"):
                    img = img.convert("RGB")
                buf = io.BytesIO()
                img.save(buf, format="PNG")
                image_base64 = base64.b64encode(buf.getvalue()).decode("utf-8")
                image_list_encoded.append(image_base64)
        if image_list_encoded:
            payload["images"] = image_list_encoded

        response = requests.post(
            f"{url}/api/generate",
            json=payload,
            headers=headers,
            timeout=120,
        )

        if response.status_code == 200:
            data = response.json()
            print(data)
            description = data.get("response", "no response generated")
            return (description,)
        else:
            raise Exception(f"Error during inference: {response.status_code} - {response.text}")
