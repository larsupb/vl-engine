from .nodes import APIEndpoint, ImageDescriptor, PromptImprove, StoryPlotGenerator, PromptEdit

version_code = [0, 1, 0]
version_str = f"V{version_code[0]}.{version_code[1]}" + (f'.{version_code[2]}' if len(version_code) > 2 else '')
print(f"### Loading: VL-Engine ({version_str})")

NODE_CLASS_MAPPINGS = {
    "VL Engine API Endpoint": APIEndpoint,
    "VL Engine Image Description": ImageDescriptor,
    "VL Engine Prompt Improve": PromptImprove,
    "VL Engine Story Plot Generator": StoryPlotGenerator,
    "VL Engine Prompt Edit": PromptEdit,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "VL Engine API Endpoint": "VL Engine API Endpoint",
    "VL Engine Image Description": "VL Engine Image Description",
    "VL Engine Prompt Improve": "VL Engine Prompt Improve",
    "VL Engine Story Plot Generator": "VL Engine Story Plot Generator",
    "VL Engine Prompt Edit": "VL Engine Prompt Edit",
}


WEB_DIRECTORY = "./js"

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS", "WEB_DIRECTORY"]