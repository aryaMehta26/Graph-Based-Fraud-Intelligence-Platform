"""Common inference interface for local Transformers and OpenAI-compatible servers."""
import json
import urllib.request
from typing import Any, Dict, Optional

class ModelBackend:
    def generate(self, messages, *, max_tokens=512, temperature=0.0) -> Dict[str, Any]:
        raise NotImplementedError

class TransformersBackend(ModelBackend):
    def __init__(self, model_id: str, revision: str = "main", adapter_path: Optional[str] = None, model_kwargs: Optional[Dict[str, Any]] = None, tokenizer_kwargs: Optional[Dict[str, Any]] = None, **kwargs):
        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError as exc: raise RuntimeError("Install transformers and torch for local inference") from exc
        model_kwargs = dict(model_kwargs or {})
        model_kwargs.update(kwargs)
        tokenizer_kwargs = dict(tokenizer_kwargs or {})
        quantized = model_kwargs.pop("load_in_4bit", False)
        if quantized and "quantization_config" not in model_kwargs:
            import torch
            from transformers import BitsAndBytesConfig
            model_kwargs["quantization_config"] = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True)
        self.tokenizer = AutoTokenizer.from_pretrained(model_id, revision=revision, **tokenizer_kwargs)
        self.model = AutoModelForCausalLM.from_pretrained(model_id, revision=revision, **model_kwargs)
        if adapter_path:
            try:
                from peft import PeftModel
                self.model = PeftModel.from_pretrained(self.model, adapter_path)
            except ImportError as exc: raise RuntimeError("Install peft to load an AML adapter") from exc

    def generate(self, messages, *, max_tokens=512, temperature=0.0):
        prompt = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        output = self.model.generate(**inputs, max_new_tokens=max_tokens, do_sample=temperature > 0, temperature=max(temperature, 1e-5))
        text = self.tokenizer.decode(output[0][inputs.input_ids.shape[-1]:], skip_special_tokens=True)
        return {"text": text, "prompt_tokens": int(inputs.input_ids.shape[-1]), "output_tokens": len(self.tokenizer.encode(text))}

class OpenAICompatibleBackend(ModelBackend):
    def __init__(self, base_url: str, model: str, api_key: str = "local", timeout: int = 180):
        self.url = base_url.rstrip("/") + "/chat/completions"; self.model = model; self.api_key = api_key; self.timeout = timeout
    def generate(self, messages, *, max_tokens=512, temperature=0.0):
        payload = json.dumps({"model": self.model, "messages": messages, "max_tokens": max_tokens, "temperature": temperature}).encode()
        request = urllib.request.Request(self.url, payload, {"Content-Type": "application/json", "Authorization": f"Bearer {self.api_key}"})
        with urllib.request.urlopen(request, timeout=self.timeout) as response: data = json.loads(response.read())
        choice = data["choices"][0]["message"]["content"]
        usage = data.get("usage", {})
        return {"text": choice, "prompt_tokens": usage.get("prompt_tokens", 0), "output_tokens": usage.get("completion_tokens", 0)}
