"""
Diffusion Prompt Agent System - Core Logic

This module implements a multi-agent system for enhancing prompts for diffusion models.
Each agent specializes in a specific aspect of prompt enhancement.
"""

import json
import requests
from typing import Dict, List, Optional, Any
from pydantic import BaseModel, Field, ValidationError

from . import agent_prompts


# ============================================================================
# Pydantic Models for Structured State
# ============================================================================

class ExpertConfig(BaseModel):
    """Configuration for a single expert agent"""
    enabled: bool = Field(default=True, description="Whether this expert is enabled")
    custom_instructions: str = Field(default="", description="Additional instructions for this expert")
    max_tokens: int = Field(default=512, description="Maximum number of tokens to generate")


class ExpertSettings(BaseModel):
    """Settings for all expert agents (excluding IntentExtractor)"""
    subject_expert: ExpertConfig = Field(default_factory=ExpertConfig)
    scene_expert: ExpertConfig = Field(default_factory=ExpertConfig)
    action_expert: ExpertConfig = Field(default_factory=ExpertConfig)
    technical_expert: ExpertConfig = Field(default_factory=ExpertConfig)
    style_harmonizer: ExpertConfig = Field(default_factory=ExpertConfig)
    negative_prompt_generator: ExpertConfig = Field(default_factory=ExpertConfig)
    validator_compressor: ExpertConfig = Field(default_factory=ExpertConfig)


class IntentData(BaseModel):
    """Structured intent extracted from user prompt"""
    subject: str = Field(default="", description="Main subject of the image")
    scene: str = Field(default="", description="Scene or setting description")
    action: str = Field(default="", description="Action or activity being performed")
    style: str = Field(default="", description="Artistic style or aesthetic")
    must_include: List[str] = Field(default_factory=list, description="Terms that must appear in final prompt")
    must_not_include: List[str] = Field(default_factory=list, description="Terms to avoid")


class TechnicalDetails(BaseModel):
    """Technical aspects for realistic rendering"""
    lighting: str = Field(default="", description="Lighting setup and quality")
    composition: str = Field(default="", description="Framing and composition details")
    perspective: str = Field(default="", description="Camera perspective or angle")
    lens: str = Field(default="", description="Lens type or focal length")
    environment: str = Field(default="", description="Environmental details")


class PromptState(BaseModel):
    """Complete state of the prompt enhancement pipeline"""
    # User input
    user_prompt: str = Field(description="Original user-provided prompt")

    # Intent extraction
    intent: IntentData = Field(default_factory=IntentData)

    # Agent enhancements
    scene: str = Field(default="", description="Enhanced scene description")
    action: str = Field(default="", description="Enhanced action description")
    subjects: str = Field(default="", description="Enhanced subject description")
    style: str = Field(default="", description="Enhanced style description")
    technical: TechnicalDetails = Field(default_factory=TechnicalDetails)

    # Negative prompt
    negative: str = Field(default="", description="Negative prompt to avoid artifacts")

    # Metadata and debugging
    notes: Dict[str, str] = Field(default_factory=dict, description="Agent notes for debugging")

    # Final output
    final_prompt: str = Field(default="", description="Final enhanced prompt for diffusion model")

    class Config:
        """Pydantic configuration"""
        json_schema_extra = {
            "example": {
                "user_prompt": "a cat in a garden",
                "intent": {
                    "subject": "cat",
                    "scene": "garden",
                    "action": "",
                    "style": "",
                    "must_include": [],
                    "must_not_include": []
                },
                "scene": "lush garden with flowers",
                "action": "sitting peacefully",
                "subjects": "orange tabby cat",
                "style": "photorealistic",
                "technical": {
                    "lighting": "natural sunlight",
                    "composition": "rule of thirds",
                    "perspective": "eye level",
                    "lens": "50mm",
                    "environment": "outdoor daylight"
                },
                "negative": "blurry, distorted, low quality",
                "notes": {},
                "final_prompt": "orange tabby cat sitting peacefully in lush garden with flowers, photorealistic, natural sunlight, rule of thirds, eye level, 50mm"
            }
        }


# ============================================================================
# Base Agent Class
# ============================================================================

class BaseAgent:
    """Base class for all prompt enhancement agents"""

    def __init__(self, api_config: Dict[str, Any], custom_instructions: str = "", max_tokens: int = 512):
        """
        Initialize agent with API configuration

        Args:
            api_config: Dictionary with 'endpoint_url', 'api_key', 'model' keys
            custom_instructions: Optional custom instructions to append to system prompt
            max_tokens: Maximum number of tokens to generate
        """
        self.endpoint_url = api_config.get('endpoint_url', '')
        self.api_key = api_config.get('api_key', '')
        self.model = api_config.get('model', '')
        self.timeout = api_config.get('timeout', 120)
        self.custom_instructions = custom_instructions
        self.max_tokens = max_tokens

    def get_system_prompt(self) -> str:
        """
        Get the system prompt for this agent
        Must be overridden by subclasses
        """
        raise NotImplementedError("Subclasses must implement get_system_prompt()")

    def get_user_prompt(self, state: PromptState) -> str:
        """
        Construct user prompt from current state
        Must be overridden by subclasses
        """
        raise NotImplementedError("Subclasses must implement get_user_prompt()")

    def call_llm(self, system_prompt: str, user_prompt: str) -> str:
        """
        Call the LLM API with the given prompts

        Args:
            system_prompt: System/instruction prompt
            user_prompt: User query prompt

        Returns:
            LLM response text

        Raises:
            RuntimeError: If API call fails
        """
        headers = {
            "Content-Type": "application/json",
        }
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        payload = {
            "model": self.model,
            "prompt": f"{system_prompt}\n\nUser: {user_prompt}\n\nAssistant:",
            "stream": False,
            "temperature": 0.7,
            "max_new_tokens": self.max_tokens,
        }

        try:
            response = requests.post(
                f"{self.endpoint_url}/api/generate",
                headers=headers,
                json=payload,
                timeout=self.timeout
            )
            response.raise_for_status()
            result = response.json()
            return result.get("response", "")
        except requests.exceptions.RequestException as e:
            raise RuntimeError(f"LLM API call failed: {str(e)}")

    def parse_json_response(self, response: str) -> Dict[str, Any]:
        """
        Parse JSON from LLM response

        Args:
            response: Raw LLM response text

        Returns:
            Parsed JSON dictionary

        Raises:
            ValueError: If JSON parsing fails
        """
        # Try to extract JSON from markdown code blocks
        if "```json" in response:
            start = response.find("```json") + 7
            end = response.find("```", start)
            response = response[start:end].strip()
        elif "```" in response:
            start = response.find("```") + 3
            end = response.find("```", start)
            response = response[start:end].strip()

        try:
            return json.loads(response)
        except json.JSONDecodeError as e:
            raise ValueError(f"Failed to parse JSON: {str(e)}\nResponse: {response}")

    def process(self, state: PromptState) -> PromptState:
        """
        Process the prompt state through this agent

        Args:
            state: Current prompt state

        Returns:
            Updated prompt state

        Raises:
            RuntimeError: If processing fails
        """
        system_prompt = self.get_system_prompt()

        # Append custom instructions if provided
        if self.custom_instructions:
            system_prompt += f"\n\nADDITIONAL INSTRUCTIONS:\n{self.custom_instructions}"

        user_prompt = self.get_user_prompt(state)

        # Call LLM
        response = self.call_llm(system_prompt, user_prompt)

        # Parse and apply updates
        self.update_state(state, response)

        return state

    def update_state(self, state: PromptState, response: str):
        """
        Update state based on LLM response
        Must be overridden by subclasses

        Args:
            state: Current prompt state (modified in place)
            response: Raw LLM response
        """
        raise NotImplementedError("Subclasses must implement update_state()")


# ============================================================================
# Agent Implementations (placeholders - will be completed with prompts)
# ============================================================================

class IntentExtractor(BaseAgent):
    """Extracts structured intent from user prompt"""

    def get_system_prompt(self) -> str:
        return agent_prompts.INTENT_EXTRACTOR_PROMPT

    def get_user_prompt(self, state: PromptState) -> str:
        return state.user_prompt

    def update_state(self, state: PromptState, response: str):
        data = self.parse_json_response(response)
        state.intent = IntentData(**data)
        state.notes["intent_extractor"] = "Intent extracted successfully"


class SubjectExpert(BaseAgent):
    """Enhances subject descriptions with specific details"""

    def get_system_prompt(self) -> str:
        return agent_prompts.SUBJECT_EXPERT_PROMPT

    def get_user_prompt(self, state: PromptState) -> str:
        return f'subject="{state.intent.subject}", scene="{state.intent.scene}", action="{state.intent.action}"'

    def update_state(self, state: PromptState, response: str):
        data = self.parse_json_response(response)
        subjects_text = data.get("subjects", "")
        if not subjects_text:
            # Fallback: if LLM didn't return "subjects", use the original subject
            state.subjects = state.intent.subject
            state.notes["subject_expert"] = f"Subject enhanced (fallback used, no 'subjects' in response)"
        else:
            state.subjects = subjects_text
            state.notes["subject_expert"] = f"Subject enhanced: {subjects_text[:50]}..."


class SceneExpert(BaseAgent):
    """Enhances scene and environment descriptions"""

    def get_system_prompt(self) -> str:
        return agent_prompts.SCENE_EXPERT_PROMPT

    def get_user_prompt(self, state: PromptState) -> str:
        return f'subject="{state.subjects}", scene="{state.intent.scene}", action="{state.intent.action}"'

    def update_state(self, state: PromptState, response: str):
        data = self.parse_json_response(response)
        state.scene = data.get("scene", "")
        state.notes["scene_expert"] = "Scene enhanced"


class ActionExpert(BaseAgent):
    """Enhances action and activity descriptions"""

    def get_system_prompt(self) -> str:
        return agent_prompts.ACTION_EXPERT_PROMPT

    def get_user_prompt(self, state: PromptState) -> str:
        return f'subject="{state.intent.subject}", scene="{state.scene}", action="{state.intent.action}"'

    def update_state(self, state: PromptState, response: str):
        data = self.parse_json_response(response)
        state.action = data.get("action", "")
        state.notes["action_expert"] = "Action enhanced"


class TechnicalExpert(BaseAgent):
    """Adds technical photography/rendering details"""

    def get_system_prompt(self) -> str:
        return agent_prompts.TECHNICAL_EXPERT_PROMPT

    def get_user_prompt(self, state: PromptState) -> str:
        return f'subject="{state.intent.subject}", scene="{state.scene}", action="{state.action}", style="{state.intent.style}"'

    def update_state(self, state: PromptState, response: str):
        data = self.parse_json_response(response)
        state.technical = TechnicalDetails(**data.get("technical", {}))
        state.notes["technical_expert"] = "Technical details added"


class StyleHarmonizer(BaseAgent):
    """Ensures style coherence and removes conflicts"""

    def get_system_prompt(self) -> str:
        return agent_prompts.STYLE_HARMONIZER_PROMPT

    def get_user_prompt(self, state: PromptState) -> str:
        tech_summary = f"lighting={state.technical.lighting}, composition={state.technical.composition}"
        return f'style="{state.intent.style}", scene="{state.scene}", technical="{tech_summary}"'

    def update_state(self, state: PromptState, response: str):
        data = self.parse_json_response(response)
        state.style = data.get("style", "")
        state.notes["style_harmonizer"] = "Style harmonized"


class NegativePromptGenerator(BaseAgent):
    """Generates negative prompt to avoid artifacts"""

    def get_system_prompt(self) -> str:
        return agent_prompts.NEGATIVE_PROMPT_GENERATOR_PROMPT

    def get_user_prompt(self, state: PromptState) -> str:
        must_not = state.intent.must_not_include
        return f'style="{state.style}", must_not_include={must_not}'

    def update_state(self, state: PromptState, response: str):
        data = self.parse_json_response(response)
        state.negative = data.get("negative", "")
        state.notes["negative_generator"] = "Negative prompt generated"


class ValidatorCompressor(BaseAgent):
    """Validates requirements and compresses final prompt"""

    def get_system_prompt(self) -> str:
        return agent_prompts.VALIDATOR_COMPRESSOR_PROMPT

    def get_user_prompt(self, state: PromptState) -> str:
        tech_str = f"lighting={state.technical.lighting}, composition={state.technical.composition}, perspective={state.technical.perspective}, lens={state.technical.lens}"
        must_include = state.intent.must_include if state.intent.must_include else []
        # Use enhanced subjects instead of intent.subject
        return f"""subjects="{state.subjects}"
action="{state.action}"
scene="{state.scene}"
style="{state.style}"
technical="{tech_str}"
must_include={must_include}"""

    def update_state(self, state: PromptState, response: str):
        data = self.parse_json_response(response)
        state.final_prompt = data.get("final_prompt", "")
        # Don't overwrite subjects - it's already set by SubjectExpert
        state.notes["validator"] = "Prompt validated and compressed"


# ============================================================================
# Orchestrator
# ============================================================================

class PromptOrchestrator:
    """Orchestrates the sequential execution of all agents"""

    def __init__(self, api_config_primary: Dict[str, Any], api_config_complex: Dict[str, Any],
                 expert_settings: Optional[ExpertSettings] = None,
                 enable_logging: bool = False, progress_callback=None):
        """
        Initialize orchestrator with API configurations for two model tiers

        Args:
            api_config_primary: API config for fast agents (Intent, Subject, Scene, Action, Negative)
            api_config_complex: API config for complex agents (Technical, Style, Validator)
            expert_settings: Optional settings for enabling/disabling agents and custom instructions
            enable_logging: Whether to log intermediate states
            progress_callback: Optional callback function(current_step, agent_name) for progress updates
        """
        self.api_config_primary = api_config_primary
        self.api_config_complex = api_config_complex
        self.expert_settings = expert_settings or ExpertSettings()
        self.enable_logging = enable_logging
        self.progress_callback = progress_callback

        # Initialize all agents with appropriate config, custom instructions, and max_tokens
        # Fast agents use primary model
        self.agents = [
            IntentExtractor(api_config_primary),  # Always enabled, no custom instructions
            SubjectExpert(api_config_primary, self.expert_settings.subject_expert.custom_instructions, self.expert_settings.subject_expert.max_tokens),
            SceneExpert(api_config_primary, self.expert_settings.scene_expert.custom_instructions, self.expert_settings.scene_expert.max_tokens),
            ActionExpert(api_config_primary, self.expert_settings.action_expert.custom_instructions, self.expert_settings.action_expert.max_tokens),
            TechnicalExpert(api_config_complex, self.expert_settings.technical_expert.custom_instructions, self.expert_settings.technical_expert.max_tokens),      # Complex model
            StyleHarmonizer(api_config_complex, self.expert_settings.style_harmonizer.custom_instructions, self.expert_settings.style_harmonizer.max_tokens),      # Complex model
            NegativePromptGenerator(api_config_primary, self.expert_settings.negative_prompt_generator.custom_instructions, self.expert_settings.negative_prompt_generator.max_tokens),
            ValidatorCompressor(api_config_complex, self.expert_settings.validator_compressor.custom_instructions, self.expert_settings.validator_compressor.max_tokens),  # Complex model
        ]

        # Track which agents use which model for logging
        self.agent_model_map = {
            "IntentExtractor": api_config_primary["model"],
            "SubjectExpert": api_config_primary["model"],
            "SceneExpert": api_config_primary["model"],
            "ActionExpert": api_config_primary["model"],
            "TechnicalExpert": api_config_complex["model"],
            "StyleHarmonizer": api_config_complex["model"],
            "NegativePromptGenerator": api_config_primary["model"],
            "ValidatorCompressor": api_config_complex["model"],
        }

        # Track which agents are enabled
        self.agent_enabled_map = {
            "IntentExtractor": True,  # Always enabled
            "SubjectExpert": self.expert_settings.subject_expert.enabled,
            "SceneExpert": self.expert_settings.scene_expert.enabled,
            "ActionExpert": self.expert_settings.action_expert.enabled,
            "TechnicalExpert": self.expert_settings.technical_expert.enabled,
            "StyleHarmonizer": self.expert_settings.style_harmonizer.enabled,
            "NegativePromptGenerator": self.expert_settings.negative_prompt_generator.enabled,
            "ValidatorCompressor": self.expert_settings.validator_compressor.enabled,
        }

    def process(self, user_prompt: str) -> tuple[PromptState, List[str]]:
        """
        Process user prompt through all agents

        Args:
            user_prompt: Original user prompt

        Returns:
            Tuple of (final_state, agent_updates_list)
            - final_state: Final enhanced prompt state
            - agent_updates_list: List of strings showing what each agent did

        Raises:
            RuntimeError: If any agent fails
        """
        # Initialize state
        state = PromptState(user_prompt=user_prompt)
        agent_updates = []

        # Run through each agent sequentially
        for i, agent in enumerate(self.agents):
            agent_name = agent.__class__.__name__

            # Check if agent is enabled
            if not self.agent_enabled_map.get(agent_name, True):
                if self.enable_logging:
                    print(f"\n[{i+1}/{len(self.agents)}] Skipping {agent_name} (disabled)")
                agent_updates.append(f"⊘ {agent_name}: disabled")
                continue

            # Call progress callback if provided
            if self.progress_callback:
                self.progress_callback(i + 1, agent_name)

            if self.enable_logging:
                print(f"\n[{i+1}/{len(self.agents)}] Running {agent_name}...")

            try:
                state = agent.process(state)

                # Collect what this agent did
                update = self._format_agent_update(agent_name, state)
                agent_updates.append(update)

                if self.enable_logging:
                    print(f"✓ {agent_name} completed")
                    print(f"  State: {state.model_dump_json(indent=2)}")

            except Exception as e:
                error_msg = f"Agent {agent_name} failed: {str(e)}"
                agent_updates.append(f"❌ {agent_name}: {error_msg}")
                if self.enable_logging:
                    print(f"✗ {error_msg}")
                raise RuntimeError(error_msg)

        return state, agent_updates

    def _format_agent_update(self, agent_name: str, state: PromptState) -> str:
        """Format a readable update message for what an agent did"""
        model_name = self.agent_model_map.get(agent_name, "unknown")

        if agent_name == "IntentExtractor":
            return f"✓ IntentExtractor [{model_name}]: subject='{state.intent.subject}', scene='{state.intent.scene}', action='{state.intent.action}'"
        elif agent_name == "SubjectExpert":
            return f"✓ SubjectExpert [{model_name}]: {state.subjects}"
        elif agent_name == "SceneExpert":
            return f"✓ SceneExpert [{model_name}]: {state.scene}"
        elif agent_name == "ActionExpert":
            return f"✓ ActionExpert [{model_name}]: {state.action}"
        elif agent_name == "TechnicalExpert":
            return f"✓ TechnicalExpert [{model_name}]: lighting='{state.technical.lighting}', composition='{state.technical.composition}'"
        elif agent_name == "StyleHarmonizer":
            return f"✓ StyleHarmonizer [{model_name}]: {state.style}"
        elif agent_name == "NegativePromptGenerator":
            return f"✓ NegativePromptGenerator [{model_name}]: {state.negative}"
        elif agent_name == "ValidatorCompressor":
            return f"✓ ValidatorCompressor [{model_name}]: {state.final_prompt[:100]}..."
        else:
            return f"✓ {agent_name} [{model_name}]: completed"
