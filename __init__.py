from .nodes import APIEndpoint, ImageDescriptor

version_code = [0, 1, 0]
version_str = f"V{version_code[0]}.{version_code[1]}" + (f'.{version_code[2]}' if len(version_code) > 2 else '')
print(f"### Loading: VL-Engine ({version_str})")

NODE_CLASS_MAPPINGS = {
    "VL Engine API Endpoint": APIEndpoint,
    "VL Engine Image Description": ImageDescriptor,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "VL Engine API Endpoint": "VL Engine API Endpoint",
    "VL Engine Image Description": "VL Engine Image Description",
}


WEB_DIRECTORY = "./js"

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS", "WEB_DIRECTORY"]