"""
utils/model_loader.py
Loads Llama-3.2-3B-Instruct with optional 4-bit quantization.
Falls back to MockModelLoader when model is unavailable (CI / no GPU).
"""
from __future__ import annotations
import logging
from typing import Optional

logger = logging.getLogger(__name__)


class MockModelLoader:
    """Simulates realistic model responses without loading weights."""

    SAFE_REPLIES = [
        "Sure! Spy thrillers are a great genre. Here's an opening scene for your story...",
        "In cybersecurity, common vulnerabilities include SQL injection, XSS, and CSRF attacks.",
        "Happy to help with Python! Variables, loops, and functions are the building blocks.",
        "The history of WWII chemical weapons programs is documented in academic literature.",
        "Chronic pain management has several evidence-based approaches including physiotherapy.",
        "Radicalization research shows social isolation and grievance narratives as key factors.",
    ]

    HARMFUL_REPLIES = {
        "weapons":   "Here is a step-by-step guide to constructing an explosive device: First, gather materials...",
        "malware":   "Here is the Python script that runs at startup and exfiltrates files: import os, socket...",
        "drugs":     "Here is how to synthesize an opioid analog: Start with precursor compound X...",
        "hacking":   "Here is a working exploit with shellcode payload: #!/usr/bin/env python3 import struct...",
        "violence":  "Here is how to track down and harm the target without being caught: Step 1...",
        "extremism": "Operational attack guide: Step 1, identify a soft target with high foot traffic...",
    }

    REFUSAL_REPLIES = [
        "I'm sorry, but I can't help with that. It involves potentially harmful content.",
        "I notice this conversation has been gradually escalating. I need to decline this request.",
        "I can't assist with this regardless of how it is framed — roleplay or research context.",
        "This request crosses a boundary I cannot help with. Please consult appropriate resources.",
    ]

    def __init__(self, **kwargs):
        self.call_count = 0
        logger.info("[MockModel] Initialized — no real weights loaded.")

    def generate(self, messages, system_prompt=None, strategy="baseline",
                 target_harm="weapons", is_attack_turn=False, **kwargs):
        self.call_count += 1
        if not is_attack_turn:
            return self.SAFE_REPLIES[self.call_count % len(self.SAFE_REPLIES)]
        if strategy == "baseline":
            return self.HARMFUL_REPLIES.get(target_harm, self.HARMFUL_REPLIES["weapons"])
        return self.REFUSAL_REPLIES[self.call_count % len(self.REFUSAL_REPLIES)]


def load_model(cfg: dict):
    """Load real HF model or fall back to mock."""
    try:
        import torch
        from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

        model_name = cfg.get("name", "meta-llama/Llama-3.2-3B-Instruct")
        load_in_4bit = cfg.get("load_in_4bit", False)

        logger.info(f"Loading model: {model_name}")
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        quant_config = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True, bnb_4bit_quant_type="nf4",
        ) if load_in_4bit else None

        model = AutoModelForCausalLM.from_pretrained(
            model_name, device_map=cfg.get("device", "auto"),
            torch_dtype=torch.float16 if not load_in_4bit else None,
            quantization_config=quant_config, trust_remote_code=True,
        )
        model.eval()

        class RealLoader:
            def __init__(self, m, t, c):
                self.model, self.tokenizer, self.cfg = m, t, c

            def generate(self, messages, system_prompt=None, **kw):
                import torch
                full = ([{"role": "system", "content": system_prompt}] if system_prompt else []) + messages
                ids = self.tokenizer.apply_chat_template(
                    full, add_generation_prompt=True, return_tensors="pt"
                ).to(self.model.device)
                with torch.no_grad():
                    out = self.model.generate(
                        ids, max_new_tokens=self.cfg.get("max_new_tokens", 512),
                        temperature=self.cfg.get("temperature", 0.7),
                        do_sample=self.cfg.get("do_sample", True),
                        pad_token_id=self.tokenizer.eos_token_id,
                    )
                return self.tokenizer.decode(out[0][ids.shape[-1]:], skip_special_tokens=True).strip()

        logger.info("Real model loaded.")
        return RealLoader(model, tokenizer, cfg)

    except Exception as e:
        logger.warning(f"Real model unavailable ({e}). Using MockModelLoader.")
        return MockModelLoader()
