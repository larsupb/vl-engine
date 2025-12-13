# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

VL-Engine is a ComfyUI custom node plugin that enables vision-language model inference via remote OpenAI-compatible APIs. It provides two main nodes:
- **APIEndpoint**: Configures the remote API connection and discovers available models
- **ImageDescriptor**: Performs vision-language inference using text prompts and images

## Architecture

### Node System
ComfyUI uses a node-based architecture where each custom node is a Python class with specific methods:
- `INPUT_TYPES()`: Defines node inputs (required/optional parameters)
- `RETURN_TYPES/RETURN_NAMES`: Specifies output types and names
- Main function (specified by `FUNCTION` attribute): Performs the node's operation
- `CATEGORY`: Groups nodes in the ComfyUI UI

### Registration Pattern
Nodes are registered in `__init__.py` through two dictionaries:
- `NODE_CLASS_MAPPINGS`: Maps node names to Python classes
- `NODE_DISPLAY_NAME_MAPPINGS`: Maps internal names to UI display names

The `WEB_DIRECTORY` points to frontend JavaScript code (currently `./js` but directory doesn't exist yet).

### Data Flow
1. **APIEndpoint node** (`nodes.py:21-65`):
   - Takes endpoint URL, API key, and endpoint type
   - Fetches available models from `/api/tags` endpoint
   - Returns endpoint configuration dict and model list string
   - Custom type `API_ENDPOINT` is passed to ImageDescriptor

2. **ImageDescriptor node** (`nodes.py:68-161`):
   - Receives `API_ENDPOINT` from APIEndpoint node
   - Converts ComfyUI image tensors (shape: `[batch, height, width, channels]`) to PIL Images
   - Encodes images as base64 PNG
   - Sends requests to `/api/generate` endpoint with OpenAI-compatible payload
   - Returns text response as string

### Image Handling
ComfyUI represents images as PyTorch-like tensors with shape `[batch, height, width, channels]` and values in range [0, 1]. The `tensor_to_pil()` function (`nodes.py:13-18`) handles conversion:
- Extracts single image from batch at specified index
- Scales values from [0, 1] to [0, 255]
- Converts to numpy array and creates PIL Image

## Development Commands

### Testing in ComfyUI
This plugin must be tested within a running ComfyUI instance:
```bash
# Navigate to ComfyUI directory and run
python main.py
# or
python -m comfy.cli.main
```

### Dependencies
Install required packages:
```bash
pip install -r requirements.txt
# or
pip install requests Pillow numpy
```

### Code Quality
No linting or testing tools are currently configured. When adding them:
- Use `ruff` or `flake8` for Python linting
- Use `pytest` for testing
- Testing will require mocking ComfyUI's tensor types and node infrastructure

## API Contract

### Endpoint Discovery (GET `/api/tags`)
Expected response format:
```json
{
  "models": [
    {"name": "model-name-1"},
    {"name": "model-name-2"}
  ]
}
```

### Inference (POST `/api/generate`)
Request payload:
```json
{
  "model": "model-name",
  "prompt": "text prompt",
  "stream": false,
  "temperature": 0.7,
  "max_new_tokens": 512,
  "seed": -1,
  "images": ["base64-encoded-png", ...]
}
```

Expected response:
```json
{
  "response": "generated text description"
}
```

## Key Constraints

- This is a ComfyUI custom node, so it must be placed in ComfyUI's `custom_nodes/` directory
- Images must be converted from ComfyUI's tensor format to base64-encoded PNG
- API endpoints must be OpenAI-compatible (currently only OpenAI type is supported)
- The `WEB_DIRECTORY` references `./js` but this directory doesn't exist (create if adding frontend code)
- Timeout for inference requests is hardcoded to 120 seconds (`nodes.py:151`)
