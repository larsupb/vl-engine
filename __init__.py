from .nodes import APIEndpoint, ImageDescriptor, PromptImprove, StoryPlotGenerator, PromptEdit
from .prompt_nodes import PromptAgentPipeline, PromptStateEditor, PromptAgentSettings, PromptAgentMonitor, PromptFinalizer

version_code = [0, 1, 0]
version_str = f"V{version_code[0]}.{version_code[1]}" + (f'.{version_code[2]}' if len(version_code) > 2 else '')
print(f"### Loading: VL-Engine ({version_str})")

NODE_CLASS_MAPPINGS = {
    "VL Engine API Endpoint": APIEndpoint,
    "VL Engine Image Description": ImageDescriptor,
    "VL Engine Prompt Improve": PromptImprove,
    "VL Engine Story Plot Generator": StoryPlotGenerator,
    "VL Engine Prompt Edit": PromptEdit,
    "VL Engine Prompt Agent Pipeline": PromptAgentPipeline,
    "VL Engine Prompt State Editor": PromptStateEditor,
    "VL Engine Prompt Agent Settings": PromptAgentSettings,
    "VL Engine Prompt Agent Monitor": PromptAgentMonitor,
    "VL Engine Prompt Finalizer": PromptFinalizer,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "VL Engine API Endpoint": "VL Engine API Endpoint",
    "VL Engine Image Description": "VL Engine Image Description",
    "VL Engine Prompt Improve": "VL Engine Prompt Improve",
    "VL Engine Story Plot Generator": "VL Engine Story Plot Generator",
    "VL Engine Prompt Edit": "VL Engine Prompt Edit",
    "VL Engine Prompt Agent Pipeline": "Prompt Agent Pipeline",
    "VL Engine Prompt State Editor": "Prompt State Editor",
    "VL Engine Prompt Agent Settings": "Prompt Agent Settings",
    "VL Engine Prompt Agent Monitor": "Prompt Agent Monitor",
    "VL Engine Prompt Finalizer": "Prompt Finalizer",
}


WEB_DIRECTORY = "./js"

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS", "WEB_DIRECTORY"]