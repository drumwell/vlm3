"""
Unified model wrapper for VLM inference.

Supports:
- Qwen2-VL-7B-Instruct (base model)
- LoRA adapters for fine-tuned variants
"""

import os
from pathlib import Path
from typing import Optional

import torch
from PIL import Image


class VLMWrapper:
    """
    Unified inference interface for Qwen2-VL models.

    Handles:
    - Base model loading
    - LoRA adapter merging
    - Image preprocessing
    - Generation with consistent parameters
    """

    DEFAULT_MODEL = "Qwen/Qwen2-VL-7B-Instruct"
    DEFAULT_GENERATION_CONFIG = {
        "max_new_tokens": 512,
        "temperature": 0.1,  # Low temperature for factual answers
        "top_p": 0.9,
        "do_sample": True,
    }

    def __init__(
        self,
        model_path: str = None,
        adapter_path: str = None,
        device: str = None,
        torch_dtype: torch.dtype = None,
    ):
        """
        Initialize the VLM wrapper.

        Args:
            model_path: Path or HuggingFace model ID. Defaults to Qwen2-VL-7B-Instruct.
            adapter_path: Optional path to LoRA adapter directory.
            device: Device to load model on. Auto-detected if not specified.
            torch_dtype: Torch dtype for model. Defaults to bfloat16 if available.
        """
        self.model_path = model_path or self.DEFAULT_MODEL
        self.adapter_path = adapter_path
        self.device = device or self._detect_device()
        self.torch_dtype = torch_dtype or self._detect_dtype()

        self.model = None
        self.processor = None
        self._loaded = False

    def _detect_device(self) -> str:
        """Detect best available device."""
        if torch.cuda.is_available():
            return "cuda"
        elif torch.backends.mps.is_available():
            return "mps"
        return "cpu"

    def _detect_dtype(self) -> torch.dtype:
        """Detect best dtype for device."""
        if self.device == "cuda":
            return torch.bfloat16
        return torch.float32

    def load(self):
        """Load model and processor. Called lazily on first inference."""
        if self._loaded:
            return

        from transformers import Qwen2VLForConditionalGeneration, AutoProcessor

        print(f"Loading model: {self.model_path}")
        print(f"Device: {self.device}, dtype: {self.torch_dtype}")

        # Load processor
        self.processor = AutoProcessor.from_pretrained(self.model_path)

        # Load base model
        self.model = Qwen2VLForConditionalGeneration.from_pretrained(
            self.model_path,
            torch_dtype=self.torch_dtype,
            device_map="auto" if self.device == "cuda" else None,
        )

        # Load LoRA adapter if specified
        if self.adapter_path:
            print(f"Loading LoRA adapter: {self.adapter_path}")
            from peft import PeftModel
            self.model = PeftModel.from_pretrained(
                self.model,
                self.adapter_path,
            )
            # Optionally merge adapter for faster inference
            # self.model = self.model.merge_and_unload()

        # Move to device if not using device_map
        if self.device != "cuda":
            self.model = self.model.to(self.device)

        self.model.eval()
        self._loaded = True
        print("Model loaded successfully")

    def generate(
        self,
        image_path: str,
        question: str,
        system_prompt: str = None,
        **generation_kwargs
    ) -> str:
        """
        Generate a response for an image-question pair.

        Args:
            image_path: Path to the image file.
            question: The question to answer about the image.
            system_prompt: Optional system prompt for context.
            **generation_kwargs: Override default generation parameters.

        Returns:
            Generated response text.
        """
        self.load()

        # Load and validate image
        image = self._load_image(image_path)

        # Build conversation
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})

        messages.append({
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": question},
            ]
        })

        # Process inputs
        text = self.processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )

        inputs = self.processor(
            text=[text],
            images=[image],
            return_tensors="pt",
            padding=True,
        )

        # Move inputs to device
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        # Merge generation config
        gen_config = {**self.DEFAULT_GENERATION_CONFIG, **generation_kwargs}

        # Generate
        with torch.no_grad():
            output_ids = self.model.generate(
                **inputs,
                **gen_config,
            )

        # Decode response (exclude input tokens)
        input_len = inputs["input_ids"].shape[1]
        response = self.processor.decode(
            output_ids[0][input_len:],
            skip_special_tokens=True
        )

        return response.strip()

    def _load_image(self, image_path: str) -> Image.Image:
        """Load and validate an image file."""
        path = Path(image_path)
        if not path.exists():
            raise FileNotFoundError(f"Image not found: {image_path}")

        image = Image.open(path)

        # Convert to RGB if necessary
        if image.mode != "RGB":
            image = image.convert("RGB")

        return image

    def batch_generate(
        self,
        examples: list[dict],
        image_base_path: str = None,
        system_prompt: str = None,
        **generation_kwargs
    ) -> list[str]:
        """
        Generate responses for multiple examples.

        Args:
            examples: List of dicts with 'image' and 'question' keys.
            image_base_path: Base path for relative image paths.
            system_prompt: Optional system prompt for all examples.
            **generation_kwargs: Override default generation parameters.

        Returns:
            List of generated response texts.
        """
        responses = []

        for example in examples:
            image_path = example["image"]
            if image_base_path and not os.path.isabs(image_path):
                image_path = os.path.join(image_base_path, image_path)

            try:
                response = self.generate(
                    image_path=image_path,
                    question=example["question"],
                    system_prompt=system_prompt,
                    **generation_kwargs
                )
                responses.append(response)
            except Exception as e:
                print(f"Error generating for {image_path}: {e}")
                responses.append(f"[ERROR: {str(e)}]")

        return responses

    def get_model_info(self) -> dict:
        """Get information about the loaded model."""
        return {
            "model_path": self.model_path,
            "adapter_path": self.adapter_path,
            "device": self.device,
            "dtype": str(self.torch_dtype),
            "loaded": self._loaded,
            "has_adapter": self.adapter_path is not None,
        }


class MockVLMWrapper:
    """
    Mock wrapper for testing without GPU.

    Returns placeholder responses for testing evaluation infrastructure.
    """

    def __init__(self, **kwargs):
        self.model_path = kwargs.get("model_path", "mock-model")
        self.adapter_path = kwargs.get("adapter_path")

    def load(self):
        pass

    def generate(self, image_path: str, question: str, **kwargs) -> str:
        """Return a mock response based on question keywords."""
        q_lower = question.lower()

        if "torque" in q_lower:
            return "The torque specification is 30 Nm."
        elif "step" in q_lower or "procedure" in q_lower:
            return "1. Remove the component. 2. Inspect for damage. 3. Reinstall with new gasket."
        elif "wire" in q_lower or "color" in q_lower:
            return "The wire is black with a yellow tracer."
        elif "warning" in q_lower or "caution" in q_lower:
            return "Warning: Ensure the system is depressurized before service."
        else:
            return "This component requires service according to the maintenance schedule."

    def batch_generate(self, examples: list[dict], **kwargs) -> list[str]:
        return [self.generate(e["image"], e["question"]) for e in examples]

    def get_model_info(self) -> dict:
        return {
            "model_path": self.model_path,
            "adapter_path": self.adapter_path,
            "mock": True,
        }


def create_wrapper(
    model_path: str = None,
    adapter_path: str = None,
    mock: bool = False,
    **kwargs
) -> VLMWrapper:
    """
    Factory function to create appropriate VLM wrapper.

    Args:
        model_path: Model path or HuggingFace ID.
        adapter_path: Optional LoRA adapter path.
        mock: If True, return MockVLMWrapper for testing.
        **kwargs: Additional arguments passed to wrapper.

    Returns:
        VLMWrapper or MockVLMWrapper instance.
    """
    if mock:
        return MockVLMWrapper(model_path=model_path, adapter_path=adapter_path, **kwargs)

    return VLMWrapper(model_path=model_path, adapter_path=adapter_path, **kwargs)
