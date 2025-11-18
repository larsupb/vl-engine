# VL-Engine for ComfyUI

A ComfyUI custom node plugin that enables vision-language model inference via remote OpenAI-compatible APIs.

## Overview

VL-Engine allows you to integrate vision-language models (VLMs) into your ComfyUI workflows by connecting to remote API endpoints. This is useful for running powerful VLMs without local GPU requirements, or for integrating with existing VLM API services.

## Features

- Connect to OpenAI-compatible VLM API endpoints
- Support for multiple images per inference (up to 3 images)
- Automatic model discovery from API endpoints
- Configurable generation parameters (temperature, max tokens, seed)
- Base64 image encoding for API transmission

## Installation

### Method 1: ComfyUI Manager (Recommended)

If you have [ComfyUI Manager](https://github.com/ltdrdata/ComfyUI-Manager) installed:

1. Open ComfyUI Manager
2. Search for "VL-Engine"
3. Click Install

### Method 2: Manual Installation

1. Navigate to your ComfyUI custom nodes directory:
   ```bash
   cd ComfyUI/custom_nodes/
   ```

2. Clone this repository:
   ```bash
   git clone https://github.com/larsupb/vl-engine.git
   ```

3. Install dependencies:
   ```bash
   cd vl-engine
   pip install -r requirements.txt
   ```

4. Restart ComfyUI

## Usage

VL-Engine provides two nodes that work together:

### 1. VL Engine API Endpoint

Configure your API connection:

- **Endpoint Type**: Select API type (currently supports OpenAI-compatible endpoints)
- **Endpoint URL**: The base URL of your VLM API (e.g., `http://localhost:11434`)
- **API Key**: Your API authentication key

The node will automatically query the endpoint for available models and output a list.

### 2. VL Engine Image Description

Perform vision-language inference:

**Required Inputs:**
- **text**: Your prompt/question about the image(s)
- **model**: Model name (e.g., `qwen2.5-vl`, `llava`, etc.)
- **api_endpoint**: Connect to the API Endpoint node output
- **temperature**: Controls randomness (0.0 = deterministic, 1.0 = creative)
- **max_new_tokens**: Maximum length of generated response (128-2048)
- **seed**: Random seed for reproducibility (-1 for random)

**Optional Inputs:**
- **image1, image2, image3**: Up to 3 images for multi-image reasoning
- **video_path**: Path to video file (future feature)

**Output:**
- Generated text description/response

## Example Workflow

```
[Load Image] → [VL Engine Image Description] → [Display Text]
                      ↑
[VL Engine API Endpoint]
```

1. Set up your API endpoint configuration using the **VL Engine API Endpoint** node
2. Load one or more images into your workflow
3. Connect images and the API endpoint to **VL Engine Image Description**
4. Enter your prompt in the text field
5. Execute the workflow to get your VLM response

## Compatible API Servers

This plugin works with OpenAI-compatible API servers. Examples include:

- [Ollama](https://ollama.ai/) with VLM models (llava, qwen2.5-vl, etc.)
- [vLLM](https://github.com/vllm-project/vllm) with OpenAI-compatible server
- [LM Studio](https://lmstudio.ai/) with vision model support
- Any OpenAI API-compatible VLM endpoint

### API Requirements

Your API server must support:
- `GET /api/tags` - Returns list of available models
- `POST /api/generate` - Accepts images as base64-encoded strings

## Configuration

### API Endpoint Format

The API endpoint should be the base URL without trailing paths:
- ✅ Correct: `http://localhost:11434`
- ❌ Incorrect: `http://localhost:11434/api/generate`

### Model Selection

After connecting to an endpoint, available models will be displayed in the API Endpoint node output. Copy the model name to use in the Image Description node.

## Troubleshooting

### "Error fetching models" message

- Verify your endpoint URL is correct and accessible
- Check that your API key is valid (if required)
- Ensure the API server is running

### "Error during inference" message

- Confirm the model name matches exactly (case-sensitive)
- Verify your API endpoint is still connected
- Check API server logs for detailed error messages
- Ensure images are properly connected to the node

### Timeout errors

The default timeout is 120 seconds. For very large images or slow models, you may need to modify this in `nodes.py:151`.

## License

See [LICENSE](LICENSE) file for details.

## Contributing

Contributions are welcome! Please feel free to submit issues or pull requests.

## Repository

https://github.com/larsupb/vl-engine

## Changelog

### Version 1.0.0
- Initial release
- OpenAI-compatible API support
- Multi-image input (up to 3 images)
- Automatic model discovery
