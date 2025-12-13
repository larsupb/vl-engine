"""
ComfyUI nodes for the Diffusion Prompt Agent System

Provides three nodes:
1. PromptAgentPipeline - Main orchestrator node
2. PromptStateEditor - JSON editor for manual state tweaking
3. PromptFinalizer - Extracts final outputs from JSON state
"""

import json
import comfy.utils
from .agent_system import PromptOrchestrator, PromptState, ExpertSettings, ExpertConfig


class PromptAgentPipeline:
    """
    Main orchestrator node that runs all 7 agents sequentially
    to enhance a user prompt for diffusion models
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "user_prompt": ("STRING", {
                    "multiline": True,
                    "default": "a cat in a garden",
                    "tooltip": "Your original prompt to enhance"
                }),
                "api_endpoint": ("API_ENDPOINT", {
                    "tooltip": "API endpoint configuration from APIEndpoint node"
                }),
                "model_primary": ("STRING", {
                    "default": "",
                    "tooltip": "Primary model for fast agents (Intent, Subject, Scene, Action, Negative)"
                }),
            },
            "optional": {
                "model_complex": ("STRING", {
                    "default": "",
                    "tooltip": "Model for complex agents (Technical, Style, Validator). Leave empty to use model_primary."
                }),
                "expert_settings": ("EXPERT_SETTINGS", {
                    "tooltip": "Optional expert settings from PromptAgentSettings node"
                }),
                "enable_logging": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Print intermediate agent states to console for debugging"
                }),
                "timeout": ("INT", {
                    "default": 300,
                    "min": 30,
                    "max": 600,
                    "step": 30,
                    "tooltip": "Timeout in seconds for each agent LLM call (default: 300s)"
                }),
            }
        }

    RETURN_TYPES = ("STRING", "STRING", "STRING", "STRING")
    RETURN_NAMES = ("state_json", "final_prompt", "negative_prompt", "agent_log")
    FUNCTION = "process_prompt"
    CATEGORY = "VL-Engine/Prompt Enhancement"
    OUTPUT_NODE = True

    def process_prompt(self, user_prompt, api_endpoint, model_primary, model_complex="", expert_settings=None, enable_logging=False, timeout=300):
        """
        Process user prompt through all agents

        Args:
            user_prompt: Original user prompt string
            api_endpoint: API endpoint configuration dict from APIEndpoint node
            model_primary: Model name for fast agents
            model_complex: Model name for complex agents (optional, falls back to model_primary)
            enable_logging: Whether to print intermediate states
            timeout: Timeout in seconds for each agent LLM call

        Returns:
            Tuple of (state_json, final_prompt, negative_prompt, agent_log)
        """
        try:
            # Use model_primary for complex agents if model_complex not specified
            if not model_complex or model_complex.strip() == "":
                model_complex = model_primary

            # Build API configs for both model tiers
            api_config_primary = {
                "endpoint_url": api_endpoint.get("endpoint_url", ""),
                "api_key": api_endpoint.get("api_key", ""),
                "endpoint_type": api_endpoint.get("endpoint_type", "OpenAI"),
                "model": model_primary,
                "timeout": timeout,
            }

            api_config_complex = {
                "endpoint_url": api_endpoint.get("endpoint_url", ""),
                "api_key": api_endpoint.get("api_key", ""),
                "endpoint_type": api_endpoint.get("endpoint_type", "OpenAI"),
                "model": model_complex,
                "timeout": timeout,
            }

            # Create progress bar for 8 agents
            total_agents = 8
            pbar = comfy.utils.ProgressBar(total_agents)

            # Progress callback to update the progress bar
            def update_progress(step, agent_name):
                pbar.update(1)

            # Parse expert settings if provided
            expert_settings_obj = None
            if expert_settings:
                expert_settings_obj = ExpertSettings(**json.loads(expert_settings))

            # Initialize orchestrator with both API configs and progress callback
            orchestrator = PromptOrchestrator(
                api_config_primary=api_config_primary,
                api_config_complex=api_config_complex,
                expert_settings=expert_settings_obj,
                enable_logging=enable_logging,
                progress_callback=update_progress
            )

            # Process through all agents
            state, agent_updates = orchestrator.process(user_prompt)

            # Serialize state to JSON
            state_json = state.model_dump_json(indent=2)

            # Extract outputs
            final_prompt = state.final_prompt
            negative_prompt = state.negative

            # Format agent updates for display
            updates_text = "\n".join(agent_updates)

            # Return outputs including agent log
            return {
                "ui": {
                    "text": [updates_text]
                },
                "result": (state_json, final_prompt, negative_prompt, updates_text)
            }

        except Exception as e:
            error_msg = f"PromptAgentPipeline failed: {str(e)}"
            print(f"ERROR: {error_msg}")

            # Return error information
            error_state = {
                "error": error_msg,
                "user_prompt": user_prompt,
            }
            return {
                "ui": {
                    "text": [f"❌ ERROR: {error_msg}"]
                },
                "result": (
                    json.dumps(error_state, indent=2),
                    f"ERROR: {error_msg}",
                    "",
                    f"❌ ERROR: {error_msg}"
                )
            }


class PromptStateEditor:
    """
    Optional node for manually editing the JSON state
    between pipeline and finalizer
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "state_json": ("STRING", {
                    "multiline": True,
                    "forceInput": True,
                    "tooltip": "JSON state from PromptAgentPipeline"
                }),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("edited_state_json",)
    FUNCTION = "edit_state"
    CATEGORY = "VL-Engine/Prompt Enhancement"

    def edit_state(self, state_json):
        """
        Pass through the state JSON, allowing manual editing in the UI

        Args:
            state_json: JSON state string

        Returns:
            Tuple containing the (potentially edited) state JSON
        """
        try:
            # Validate that it's valid JSON
            json.loads(state_json)
            return (state_json,)
        except json.JSONDecodeError as e:
            error_msg = f"Invalid JSON: {str(e)}"
            print(f"ERROR: {error_msg}")
            return (json.dumps({"error": error_msg}, indent=2),)


class PromptAgentSettings:
    """
    Configure expert agent settings (enable/disable and custom instructions)
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "optional": {
                # SubjectExpert
                "subject_expert_enabled": ("BOOLEAN", {"default": True, "tooltip": "Enable SubjectExpert"}),
                "subject_expert_max_tokens": ("INT", {"default": 512, "min": 128, "max": 4096, "step": 128,
                                                      "tooltip": "Max tokens for SubjectExpert"}),
                "subject_expert_instructions": ("STRING", {"default": "", "multiline": True, "tooltip": "Custom instructions for SubjectExpert"}),

                # SceneExpert
                "scene_expert_enabled": ("BOOLEAN", {"default": True, "tooltip": "Enable SceneExpert"}),
                "scene_expert_max_tokens": ("INT", {"default": 512, "min": 128, "max": 4096, "step": 128,
                                                    "tooltip": "Max tokens for SceneExpert"}),
                "scene_expert_instructions": ("STRING", {"default": "", "multiline": True, "tooltip": "Custom instructions for SceneExpert"}),

                # ActionExpert
                "action_expert_enabled": ("BOOLEAN", {"default": True, "tooltip": "Enable ActionExpert"}),
                "action_expert_max_tokens": ("INT", {"default": 512, "min": 128, "max": 4096, "step": 128,
                                                     "tooltip": "Max tokens for ActionExpert"}),
                "action_expert_instructions": ("STRING", {"default": "", "multiline": True, "tooltip": "Custom instructions for ActionExpert"}),

                # TechnicalExpert
                "technical_expert_enabled": ("BOOLEAN", {"default": True, "tooltip": "Enable TechnicalExpert"}),
                "technical_expert_max_tokens": ("INT", {"default": 512, "min": 128, "max": 4096, "step": 128,
                                                        "tooltip": "Max tokens for TechnicalExpert"}),
                "technical_expert_instructions": ("STRING", {"default": "", "multiline": True, "tooltip": "Custom instructions for TechnicalExpert"}),

                # StyleHarmonizer
                "style_harmonizer_enabled": ("BOOLEAN", {"default": True, "tooltip": "Enable StyleHarmonizer"}),
                "style_harmonizer_max_tokens": ("INT", {"default": 512, "min": 128, "max": 4096, "step": 128,
                                                        "tooltip": "Max tokens for StyleHarmonizer"}),
                "style_harmonizer_instructions": ("STRING", {"default": "", "multiline": True, "tooltip": "Custom instructions for StyleHarmonizer"}),

                # NegativePromptGenerator
                "negative_prompt_generator_enabled": ("BOOLEAN", {"default": True, "tooltip": "Enable NegativePromptGenerator"}),
                "negative_prompt_generator_max_tokens": ("INT", {"default": 512, "min": 128, "max": 4096, "step": 128,
                                                                 "tooltip": "Max tokens for NegativePromptGenerator"}),
                "negative_prompt_generator_instructions": ("STRING", {"default": "", "multiline": True, "tooltip": "Custom instructions for NegativePromptGenerator"}),

                # ValidatorCompressor (always enabled, no toggle)
                "validator_compressor_instructions": ("STRING", {"default": "", "multiline": True, "tooltip": "Custom instructions for ValidatorCompressor"}),
                "validator_compressor_max_tokens": ("INT", {"default": 2048, "min": 128, "max": 4096, "step": 128, "tooltip": "Max tokens for ValidatorCompressor"}),
            }
        }

    RETURN_TYPES = ("EXPERT_SETTINGS",)
    RETURN_NAMES = ("expert_settings",)
    FUNCTION = "build_settings"
    CATEGORY = "VL-Engine/Prompt Enhancement"

    def build_settings(self, **kwargs):
        """
        Build expert settings from inputs

        Returns:
            Tuple containing ExpertSettings object serialized as JSON
        """
        settings = ExpertSettings(
            subject_expert=ExpertConfig(
                enabled=kwargs.get("subject_expert_enabled", True),
                custom_instructions=kwargs.get("subject_expert_instructions", ""),
                max_tokens=kwargs.get("subject_expert_max_tokens", 512)
            ),
            scene_expert=ExpertConfig(
                enabled=kwargs.get("scene_expert_enabled", True),
                custom_instructions=kwargs.get("scene_expert_instructions", ""),
                max_tokens=kwargs.get("scene_expert_max_tokens", 512)
            ),
            action_expert=ExpertConfig(
                enabled=kwargs.get("action_expert_enabled", True),
                custom_instructions=kwargs.get("action_expert_instructions", ""),
                max_tokens=kwargs.get("action_expert_max_tokens", 512)
            ),
            technical_expert=ExpertConfig(
                enabled=kwargs.get("technical_expert_enabled", True),
                custom_instructions=kwargs.get("technical_expert_instructions", ""),
                max_tokens=kwargs.get("technical_expert_max_tokens", 512)
            ),
            style_harmonizer=ExpertConfig(
                enabled=kwargs.get("style_harmonizer_enabled", True),
                custom_instructions=kwargs.get("style_harmonizer_instructions", ""),
                max_tokens=kwargs.get("style_harmonizer_max_tokens", 512)
            ),
            negative_prompt_generator=ExpertConfig(
                enabled=kwargs.get("negative_prompt_generator_enabled", True),
                custom_instructions=kwargs.get("negative_prompt_generator_instructions", ""),
                max_tokens=kwargs.get("negative_prompt_generator_max_tokens", 512)
            ),
            validator_compressor=ExpertConfig(
                enabled=True,  # Always enabled
                custom_instructions=kwargs.get("validator_compressor_instructions", ""),
                max_tokens=kwargs.get("validator_compressor_max_tokens", 2048)
            ),
        )

        # Serialize to JSON string for passing between nodes
        return (settings.model_dump_json(),)


class PromptAgentMonitor:
    """
    Displays the agent log in a readable format
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "agent_log": ("STRING", {
                    "forceInput": True,
                    "tooltip": "Agent log from PromptAgentPipeline"
                }),
            }
        }

    RETURN_TYPES = ()
    FUNCTION = "display_log"
    CATEGORY = "VL-Engine/Prompt Enhancement"
    OUTPUT_NODE = True

    def display_log(self, agent_log):
        """
        Display the agent log in the UI

        Args:
            agent_log: Text log from PromptAgentPipeline

        Returns:
            UI display dict
        """
        return {
            "ui": {
                "text": [agent_log]
            }
        }


class PromptFinalizer:
    """
    Extracts final prompts and metadata from the JSON state
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "state_json": ("STRING", {
                    "multiline": True,
                    "forceInput": True,
                    "tooltip": "JSON state from PromptAgentPipeline or PromptStateEditor"
                }),
            }
        }

    RETURN_TYPES = ("STRING", "STRING", "STRING", "STRING")
    RETURN_NAMES = ("final_prompt", "negative_prompt", "subjects", "style")
    FUNCTION = "finalize_prompt"
    CATEGORY = "VL-Engine/Prompt Enhancement"

    def finalize_prompt(self, state_json):
        """
        Extract final outputs from JSON state

        Args:
            state_json: JSON state string

        Returns:
            Tuple of (final_prompt, negative_prompt, subjects, style)
        """
        try:
            # Parse JSON state
            state_dict = json.loads(state_json)

            # Check for errors
            if "error" in state_dict:
                error_msg = state_dict["error"]
                return (f"ERROR: {error_msg}", "", "", "")

            # Reconstruct PromptState from JSON to validate
            state = PromptState(**state_dict)

            # Extract outputs
            final_prompt = state.final_prompt or ""
            negative_prompt = state.negative or ""
            subjects = state.subjects or state.intent.subject or ""
            style = state.style or state.intent.style or ""

            return (final_prompt, negative_prompt, subjects, style)

        except Exception as e:
            error_msg = f"PromptFinalizer failed: {str(e)}"
            print(f"ERROR: {error_msg}")
            return (f"ERROR: {error_msg}", "", "", "")


# ============================================================================
# Node Export Mappings
# ============================================================================

NODE_CLASS_MAPPINGS = {
    "PromptAgentPipeline": PromptAgentPipeline,
    "PromptStateEditor": PromptStateEditor,
    "PromptAgentSettings": PromptAgentSettings,
    "PromptAgentMonitor": PromptAgentMonitor,
    "PromptFinalizer": PromptFinalizer,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "PromptAgentPipeline": "Prompt Agent Pipeline",
    "PromptStateEditor": "Prompt State Editor",
    "PromptAgentSettings": "Prompt Agent Settings",
    "PromptAgentMonitor": "Prompt Agent Monitor",
    "PromptFinalizer": "Prompt Finalizer",
}
