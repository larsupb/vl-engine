import base64
import hashlib
import io
import json
import os

import comfy.utils
import requests
import numpy as np
import yaml
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
        self.last_input_hash = None
        self.last_result = None

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
                "image3": ("IMAGE",)
            },
        }

    RETURN_TYPES = ("STRING",)
    FUNCTION = "inference"
    CATEGORY = "VL-Engine"

    def _compute_input_hash(self, text: str, model: str, api_endpoint: dict,
                           temperature: float, max_new_tokens: int, seed: int,
                           image_list_encoded: list) -> str:
        """
        Compute a hash of all inputs to detect if they have changed.
        """
        hasher = hashlib.sha256()
        hasher.update(text.encode('utf-8'))
        hasher.update(model.encode('utf-8'))
        hasher.update(api_endpoint["endpoint_url"].encode('utf-8'))
        hasher.update(str(temperature).encode('utf-8'))
        hasher.update(str(max_new_tokens).encode('utf-8'))
        hasher.update(str(seed).encode('utf-8'))
        for img_base64 in image_list_encoded:
            hasher.update(img_base64.encode('utf-8'))
        return hasher.hexdigest()

    def inference(self, text: str, model: str, api_endpoint: dict,
                  temperature: float, max_new_tokens: int, seed: int, image1=None, image2=None, image3=None) -> tuple[str]:
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

        # Compute hash of all inputs to check if we can use cached result
        current_input_hash = self._compute_input_hash(
            text, model, api_endpoint, temperature, max_new_tokens, seed, image_list_encoded
        )

        # Return cached result if inputs haven't changed
        if self.last_input_hash == current_input_hash and self.last_result is not None:
            print("Using cached result (inputs unchanged)")
            return (self.last_result,)

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
            # Cache the result and input hash
            self.last_input_hash = current_input_hash
            self.last_result = description
            return (description,)
        else:
            raise Exception(f"Error during inference: {response.status_code} - {response.text}")


class PromptImprove:
    templates = {}
    template_names = []

    def __init__(self):
        self.last_input_hash = None
        self.last_result = None
        # Load templates on first initialization
        if not PromptImprove.templates:
            PromptImprove._load_templates()

    @classmethod
    def _load_templates(cls):
        """Load prompt templates from YAML file."""
        templates_file = os.path.join(os.path.dirname(__file__), "prompt_templates.yml")
        try:
            with open(templates_file, 'r', encoding='utf-8') as f:
                data = yaml.safe_load(f)
                if data and 'templates' in data:
                    for template in data['templates']:
                        name = template.get('name')
                        instruction = template.get('instruction')
                        if name and instruction:
                            cls.templates[name] = instruction
                            cls.template_names.append(name)
            print(f"Loaded {len(cls.templates)} prompt templates")
        except Exception as e:
            print(f"Error loading prompt templates: {e}")
            # Provide a fallback template
            cls.templates["Default"] = "Improve the following prompt: {prompt}"
            cls.template_names = ["Default"]

    @classmethod
    def INPUT_TYPES(s):
        # Ensure templates are loaded
        if not PromptImprove.templates:
            PromptImprove._load_templates()

        return {
            "required": {
                "template": (PromptImprove.template_names, {"default": PromptImprove.template_names[0] if PromptImprove.template_names else "Default"}),
                "basic_prompt": ("STRING", {"default": "", "multiline": True}),
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
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("improved_prompts",)
    FUNCTION = "improve_prompt"
    CATEGORY = "VL-Engine"
    DESCRIPTION = "Improve a basic prompt using VL Engine API"
    OUTPUT_IS_LIST = (True,)

    def _compute_input_hash(self, template: str, basic_prompt: str, model: str, api_endpoint: dict,
                           temperature: float, max_new_tokens: int, seed: int) -> str:
        """
        Compute a hash of all inputs to detect if they have changed.
        """
        hasher = hashlib.sha256()
        hasher.update(template.encode('utf-8'))
        hasher.update(basic_prompt.encode('utf-8'))
        hasher.update(model.encode('utf-8'))
        hasher.update(api_endpoint["endpoint_url"].encode('utf-8'))
        hasher.update(str(temperature).encode('utf-8'))
        hasher.update(str(max_new_tokens).encode('utf-8'))
        hasher.update(str(seed).encode('utf-8'))
        return hasher.hexdigest()

    def improve_prompt(self, template: str, basic_prompt: str, model: str, api_endpoint: dict,
                      temperature: float, max_new_tokens: int, seed: int) -> tuple[list[str]]:
        """
        Improve a basic prompt using VL Engine API with a selected template.

        Returns improved prompt(s) as a list of strings.
        """
        url = api_endpoint["endpoint_url"]
        api_key = api_endpoint["api_key"]

        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }

        # Get the template instruction and format it with the basic prompt
        template_instruction = self.templates.get(template, "Improve the following prompt: {prompt}")
        formatted_prompt = template_instruction.format(prompt=basic_prompt)

        payload = {
            "model": model,
            "prompt": formatted_prompt,
            "format": "json",
            "stream": False,
            "temperature": temperature,
            "max_new_tokens": max_new_tokens,
            "seed": seed,
        }

        # Compute hash of all inputs to check if we can use cached result
        current_input_hash = self._compute_input_hash(
            template, basic_prompt, model, api_endpoint, temperature, max_new_tokens, seed
        )

        # Return cached result if inputs haven't changed
        if self.last_input_hash == current_input_hash and self.last_result is not None:
            print("Using cached result (inputs unchanged)")
            return (self.last_result,)

        response = requests.post(
            f"{url}/api/generate",
            json=payload,
            headers=headers,
            timeout=120,
        )

        if response.status_code == 200:
            data = response.json()
            print(data)
            raw_response = data.get("response", "")

            # Parse the JSON response from the LLM
            prompts = self._parse_llm_response(raw_response)

            # Cache the result and input hash
            self.last_input_hash = current_input_hash
            self.last_result = prompts
            return (prompts,)
        else:
            raise Exception(f"Error during prompt improvement: {response.status_code} - {response.text}")

    def _parse_llm_response(self, response: str) -> list[str]:
        """
        Parse the LLM response and extract prompts.
        Handles both single prompt format {"prompt": "..."} and
        multiple prompts format {"prompts": [...]} or {"plot": "...", "prompts": [...]}.

        Returns a list of prompt strings.
        """
        try:
            # Try to parse as JSON
            parsed = json.loads(response)

            # Check for different response formats
            if isinstance(parsed, dict):
                # Case 1: Single prompt - {"prompt": "..."}
                if "prompt" in parsed:
                    return [parsed["prompt"]]

                # Case 2: Multiple prompts - {"prompts": [...]}
                elif "prompts" in parsed:
                    prompts = parsed["prompts"]
                    if isinstance(prompts, list):
                        return [str(p) for p in prompts]
                    else:
                        return [str(prompts)]

                # Case 3: Story format with plot - {"plot": "...", "prompts": [...]}
                # Just return the prompts, ignore the plot for now
                elif "plot" in parsed and "prompts" in parsed:
                    prompts = parsed["prompts"]
                    if isinstance(prompts, list):
                        return [str(p) for p in prompts]
                    else:
                        return [str(prompts)]

                # Unknown format, return as single string
                else:
                    return [str(parsed)]

            # If it's already a list
            elif isinstance(parsed, list):
                return [str(p) for p in parsed]

            # Fallback: return as single string
            else:
                return [str(parsed)]

        except json.JSONDecodeError as e:
            # If JSON parsing fails, return the raw response as a single item
            print(f"Warning: Failed to parse JSON response: {e}")
            print(f"Raw response: {response}")
            return [response]


class StoryPlotGenerator:
    prompts = {}

    def __init__(self):
        self.last_input_hash = None
        self.last_result = None
        # Load prompts on first initialization
        if not StoryPlotGenerator.prompts:
            StoryPlotGenerator._load_prompts()

    @classmethod
    def _load_prompts(cls):
        """Load story plot prompts from YAML file."""
        prompts_file = os.path.join(os.path.dirname(__file__), "story_plot_prompts.yml")
        try:
            with open(prompts_file, 'r', encoding='utf-8') as f:
                cls.prompts = yaml.safe_load(f)
            print(f"Loaded story plot prompts from {prompts_file}")
        except Exception as e:
            print(f"Error loading story plot prompts: {e}")
            # Provide fallback prompts
            cls.prompts = {
                "step1_foundation": {"instruction": "Create a story foundation based on: {basic_prompt}"},
                "step2_scenes": {"instruction": "Create 6 scenes based on the plot."},
                "step3_refine": {"instruction": "Refine the scene."}
            }

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "basic_prompt": ("STRING", {"default": "", "multiline": True}),
                "model": ("STRING", {"default": "qwen2.5-vl"}),
                "api_endpoint": ("API_ENDPOINT",),
                # Step 1: Plot Generation
                "plot_temperature": (
                    "FLOAT",
                    {"default": 0.8, "min": 0, "max": 1, "step": 0.1},
                ),
                "plot_max_tokens": (
                    "INT",
                    {"default": 512, "min": 128, "max": 2048, "step": 1},
                ),
                # Step 2: Scene Generation
                "scene_temperature": (
                    "FLOAT",
                    {"default": 0.7, "min": 0, "max": 1, "step": 0.1},
                ),
                "scene_max_tokens": (
                    "INT",
                    {"default": 1024, "min": 128, "max": 2048, "step": 1},
                ),
                # Step 3: Refinement
                "refine_temperature": (
                    "FLOAT",
                    {"default": 0.5, "min": 0, "max": 1, "step": 0.1},
                ),
                "refine_max_tokens": (
                    "INT",
                    {"default": 1024, "min": 128, "max": 2048, "step": 1},
                ),
                "seed": ("INT", {"default": -1}),
            },
        }

    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("refined_prompts", "plot_summary")
    FUNCTION = "generate_story_sequence"
    CATEGORY = "VL-Engine"
    DESCRIPTION = "Generate a 6-scene story sequence with consistent details using a 3-step LLM workflow"
    OUTPUT_IS_LIST = (True, False)

    def _compute_input_hash(self, basic_prompt: str, model: str, api_endpoint: dict,
                           plot_temperature: float, plot_max_tokens: int,
                           scene_temperature: float, scene_max_tokens: int,
                           refine_temperature: float, refine_max_tokens: int,
                           seed: int) -> str:
        """Compute a hash of all inputs to detect if they have changed."""
        hasher = hashlib.sha256()
        hasher.update(basic_prompt.encode('utf-8'))
        hasher.update(model.encode('utf-8'))
        hasher.update(api_endpoint["endpoint_url"].encode('utf-8'))
        hasher.update(str(plot_temperature).encode('utf-8'))
        hasher.update(str(plot_max_tokens).encode('utf-8'))
        hasher.update(str(scene_temperature).encode('utf-8'))
        hasher.update(str(scene_max_tokens).encode('utf-8'))
        hasher.update(str(refine_temperature).encode('utf-8'))
        hasher.update(str(refine_max_tokens).encode('utf-8'))
        hasher.update(str(seed).encode('utf-8'))
        return hasher.hexdigest()

    def _call_llm(self, url: str, api_key: str, model: str, prompt: str,
                  temperature: float, max_tokens: int, seed: int) -> dict:
        """Make a single LLM API call and return parsed JSON response."""
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }

        payload = {
            "model": model,
            "prompt": prompt,
            "format": "json",
            "stream": False,
            "temperature": temperature,
            "max_new_tokens": max_tokens,
            "seed": seed,
        }

        response = requests.post(
            f"{url}/api/generate",
            json=payload,
            headers=headers,
            timeout=120,
        )

        if response.status_code == 200:
            data = response.json()
            raw_response = data.get("response", "")
            try:
                return json.loads(raw_response)
            except json.JSONDecodeError as e:
                raise Exception(f"Failed to parse LLM JSON response: {e}\nRaw response: {raw_response}")
        else:
            raise Exception(f"LLM API error: {response.status_code} - {response.text}")

    def generate_story_sequence(self, basic_prompt: str, model: str, api_endpoint: dict,
                                plot_temperature: float, plot_max_tokens: int,
                                scene_temperature: float, scene_max_tokens: int,
                                refine_temperature: float, refine_max_tokens: int,
                                seed: int) -> tuple[list[str], str]:
        """
        Generate a 6-scene story sequence using a 3-step LLM workflow.

        Step 1: Generate plot foundation (subject, environment, style)
        Step 2: Generate 6 scene outlines
        Step 3: Refine each scene individually for full detail and consistency

        Returns: (list of 6 refined prompts, plot summary)
        """
        url = api_endpoint["endpoint_url"]
        api_key = api_endpoint["api_key"]

        # Total steps: 1 (foundation) + 1 (scenes) + 6 (refinement) = 8
        total_steps = 8
        pbar = comfy.utils.ProgressBar(total_steps)

        # Compute hash of all inputs to check if we can use cached result
        current_input_hash = self._compute_input_hash(
            basic_prompt, model, api_endpoint,
            plot_temperature, plot_max_tokens,
            scene_temperature, scene_max_tokens,
            refine_temperature, refine_max_tokens,
            seed
        )

        # Return cached result if inputs haven't changed
        if self.last_input_hash == current_input_hash and self.last_result is not None:
            print("Using cached result (inputs unchanged)")
            # Update progress bar to completion for cached results
            pbar.update_absolute(total_steps)
            return self.last_result

        # Step 1: Generate plot foundation
        print("Step 1/3: Generating plot foundation...")
        step1_template = self.prompts.get("step1_foundation", {}).get("instruction", "")
        step1_prompt = step1_template.format(basic_prompt=basic_prompt)

        foundation = self._call_llm(url, api_key, model, step1_prompt,
                                    plot_temperature, plot_max_tokens, seed)
        pbar.update(1)  # Progress: 1/8 complete

        subject = foundation.get("subject", "")
        environment = foundation.get("environment", "")
        plot = foundation.get("plot", "")
        style = foundation.get("style", "")

        print(f"Plot foundation created: {plot}")

        # Step 2: Generate 6 scene outlines
        print("Step 2/3: Generating 6 scene outlines...")
        step2_template = self.prompts.get("step2_scenes", {}).get("instruction", "")
        step2_prompt = step2_template.format(
            subject=subject,
            environment=environment,
            plot=plot,
            style=style
        )

        scenes_data = self._call_llm(url, api_key, model, step2_prompt,
                                     scene_temperature, scene_max_tokens, seed)
        pbar.update(1)  # Progress: 2/8 complete

        scene_outlines = scenes_data.get("scenes", [])
        if len(scene_outlines) != 6:
            raise Exception(f"Expected 6 scene outlines but got {len(scene_outlines)}")

        print(f"Generated {len(scene_outlines)} scene outlines")

        # Step 3: Refine each scene individually
        print("Step 3/3: Refining each scene for full detail and consistency...")
        refined_prompts = []

        for i, scene_outline in enumerate(scene_outlines, 1):
            print(f"  Refining scene {i}/6...")

            # Create context with all scene outlines for reference
            scenes_context = "\n".join([f"Scene {j+1}: {s}" for j, s in enumerate(scene_outlines)])

            step3_template = self.prompts.get("step3_refine", {}).get("instruction", "")
            step3_prompt = step3_template.format(
                subject=subject,
                environment=environment,
                plot=plot,
                style=style,
                scenes_context=scenes_context,
                scene_number=i,
                scene_outline=scene_outline
            )

            refined_data = self._call_llm(url, api_key, model, step3_prompt,
                                         refine_temperature, refine_max_tokens, seed)
            pbar.update(1)  # Progress: 3-8/8 complete (one for each scene)

            refined_prompt = refined_data.get("prompt", "")
            if not refined_prompt:
                raise Exception(f"Failed to get refined prompt for scene {i}")

            refined_prompts.append(refined_prompt)

        print(f"Story sequence generation complete: {len(refined_prompts)} refined prompts")

        # Cache the result
        result = (refined_prompts, plot)
        self.last_input_hash = current_input_hash
        self.last_result = result

        return result


class PromptEdit:
    prompts = {}

    def __init__(self):
        self.last_input_hash = None
        self.last_result = None
        # Load prompts on first initialization
        if not PromptEdit.prompts:
            PromptEdit._load_prompts()

    @classmethod
    def _load_prompts(cls):
        """Load prompt edit prompts from YAML file."""
        prompts_file = os.path.join(os.path.dirname(__file__), "prompt_edit_prompts.yml")
        try:
            with open(prompts_file, 'r', encoding='utf-8') as f:
                cls.prompts = yaml.safe_load(f)
            print(f"Loaded prompt edit prompts from {prompts_file}")
        except Exception as e:
            print(f"Error loading prompt edit prompts: {e}")
            # Provide fallback prompt
            cls.prompts = {
                "edit_prompt": {"instruction": "Modify this prompt: {original_prompt}\nRequest: {modification_request}"}
            }

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "prompt_list": ("STRING",),
                "modification_request": ("STRING", {"default": "", "multiline": True}),
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
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("modified_prompts",)
    FUNCTION = "edit_prompts"
    CATEGORY = "VL-Engine"
    DESCRIPTION = "Apply modifications to a list of prompts using LLM"
    INPUT_IS_LIST = True
    OUTPUT_IS_LIST = (True,)

    def _compute_input_hash(self, prompt_list: list[str], modification_request: str,
                           model: str, api_endpoint: dict, temperature: float,
                           max_new_tokens: int, seed: int) -> str:
        """Compute a hash of all inputs to detect if they have changed."""
        hasher = hashlib.sha256()
        hasher.update(modification_request.encode('utf-8'))
        for prompt in prompt_list:
            hasher.update(prompt.encode('utf-8'))
        hasher.update(model.encode('utf-8'))
        hasher.update(api_endpoint["endpoint_url"].encode('utf-8'))
        hasher.update(str(temperature).encode('utf-8'))
        hasher.update(str(max_new_tokens).encode('utf-8'))
        hasher.update(str(seed).encode('utf-8'))
        return hasher.hexdigest()

    def _call_llm(self, url: str, api_key: str, model: str, prompt: str,
                  temperature: float, max_tokens: int, seed: int) -> dict:
        """Make a single LLM API call and return parsed JSON response."""
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }

        payload = {
            "model": model,
            "prompt": prompt,
            "format": "json",
            "stream": False,
            "temperature": temperature,
            "max_new_tokens": max_tokens,
            "seed": seed,
        }

        response = requests.post(
            f"{url}/api/generate",
            json=payload,
            headers=headers,
            timeout=120,
        )

        if response.status_code == 200:
            data = response.json()
            raw_response = data.get("response", "")
            try:
                return json.loads(raw_response)
            except json.JSONDecodeError as e:
                raise Exception(f"Failed to parse LLM JSON response: {e}\nRaw response: {raw_response}")
        else:
            raise Exception(f"LLM API error: {response.status_code} - {response.text}")

    def edit_prompts(self, prompt_list: list[str], modification_request: list[str],
                    model: list[str], api_endpoint: list[dict], temperature: list[float],
                    max_new_tokens: list[int], seed: list[int]) -> tuple[list[str]]:
        """
        Apply user-specified modifications to a list of prompts.

        Each prompt is processed individually with the same modification request.
        Returns a list of modified prompts with 1:1 mapping to input prompts.

        Note: When INPUT_IS_LIST=True, all parameters come as lists.
        We extract the first element from non-list parameters.
        """
        # Extract single values from list parameters (INPUT_IS_LIST=True makes everything a list)
        modification_request = modification_request[0] if isinstance(modification_request, list) else modification_request
        model = model[0] if isinstance(model, list) else model
        api_endpoint = api_endpoint[0] if isinstance(api_endpoint, list) else api_endpoint
        temperature = temperature[0] if isinstance(temperature, list) else temperature
        max_new_tokens = max_new_tokens[0] if isinstance(max_new_tokens, list) else max_new_tokens
        seed = seed[0] if isinstance(seed, list) else seed

        # prompt_list is already the list of strings we want

        url = api_endpoint["endpoint_url"]
        api_key = api_endpoint["api_key"]

        # Setup progress bar
        total_steps = len(prompt_list)
        pbar = comfy.utils.ProgressBar(total_steps)

        # Compute hash of all inputs to check if we can use cached result
        current_input_hash = self._compute_input_hash(
            prompt_list, modification_request, model, api_endpoint,
            temperature, max_new_tokens, seed
        )

        # Return cached result if inputs haven't changed
        if self.last_input_hash == current_input_hash and self.last_result is not None:
            print("Using cached result (inputs unchanged)")
            pbar.update_absolute(total_steps)
            return (self.last_result,)

        # Process each prompt individually
        modified_prompts = []
        template = self.prompts.get("edit_prompt", {}).get("instruction", "")

        for i, original_prompt in enumerate(prompt_list, 1):
            print(f"Editing prompt {i}/{len(prompt_list)}...")

            # Format the LLM prompt with current prompt and modification request
            llm_prompt = template.format(
                original_prompt=original_prompt,
                modification_request=modification_request
            )

            try:
                # Call LLM
                result = self._call_llm(url, api_key, model, llm_prompt,
                                       temperature, max_new_tokens, seed)

                # Extract modified prompt
                modified_prompt = result.get("modified_prompt", "")
                if not modified_prompt:
                    print(f"Warning: No modified_prompt in response for prompt {i}, using original")
                    modified_prompt = original_prompt

                modified_prompts.append(modified_prompt)

            except Exception as e:
                print(f"Error editing prompt {i}: {e}")
                print(f"Using original prompt as fallback")
                modified_prompts.append(original_prompt)

            pbar.update(1)

        print(f"Prompt editing complete: {len(modified_prompts)} prompts modified")

        # Cache the result
        self.last_input_hash = current_input_hash
        self.last_result = modified_prompts

        return (modified_prompts,)
