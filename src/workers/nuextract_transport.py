"""Text-only NuExtract3 structured-task transport for shared Ollama.

The specialist's official template is a separate task, not an instruction
inside its default document-content task. Ordinary background calls stay on
OpenAI-compatible chat. Named-model ownership belongs to the existing factory.
"""

import time

import httpx
import structlog
from openai.types.chat import ChatCompletion

from src.api.config import settings
from src.workers.extraction_result import ExtractionOutputError

logger = structlog.get_logger("ice.workers.nuextract_transport")
PROTOCOL = "nuextract3-structured-text-v1"


def render_prompt(source, template, instructions=""):
    """Match NuExtract3's installed text-only, non-thinking chat template."""
    return ("<|im_start|>user\n【task】structured\n"
            f"【template_start】{template}【template_end】\n"
            + (f"【instructions_start】{instructions}【instructions_end】\n" if instructions else "")
            + f"【document_start】\n{source.strip()}\n【document_end】<|im_end|>\n"
              "<|im_start|>assistant\n<think>\n\n</think>\n\n")


def native_completion(**kwargs):
    """Called inside the factory's existing model lock, never a new client lease."""
    from src.workers.bg_client_factory import native_call_timings
    extra = dict(kwargs.get("extra_body") or {})
    template_args = extra.get("chat_template_kwargs")
    messages = kwargs.get("messages")
    if (extra.get("ice_template_protocol") != PROTOCOL or not isinstance(template_args, dict)
            or not isinstance(messages, list) or len(messages) != 1
            or not isinstance(messages[0], dict) or messages[0].get("role") != "user"
            or not isinstance(messages[0].get("content"), str)
            or not isinstance(template_args.get("template"), str)
            or not isinstance(template_args.get("instructions", ""), str)
            or type(extra.get("ice_context_window")) is not int or extra["ice_context_window"] <= 0
            or template_args.get("enable_thinking") is not False):
        raise ExtractionOutputError("invalid structured extraction request")
    prompt = render_prompt(messages[0]["content"], template_args["template"],
                           template_args.get("instructions", ""))
    response = httpx.post(settings.ollama_base_url.rstrip('/') + '/api/generate',
                          json={"model": kwargs["model"], "prompt": prompt, "raw": True,
                                "stream": False, "think": False, "keep_alive": -1,
                                "options": {"temperature": kwargs["temperature"],
                                            "num_ctx": extra["ice_context_window"],
                                            "num_predict": kwargs["max_tokens"]}},
                          timeout=kwargs["timeout"])
    response.raise_for_status()
    try:
        data = response.json()
    except ValueError as exc:
        raise ExtractionOutputError("invalid structured extraction provider JSON") from exc
    if (not isinstance(data, dict) or data.get("done") is not True
            or not isinstance(data.get("done_reason"), str)
            or data["done_reason"] not in {"stop", "length"}):
        raise ExtractionOutputError("incomplete structured extraction provider response")
    logger.info("nuextract_structured_call", model=kwargs["model"], protocol=PROTOCOL,
                finish_reason=data["done_reason"], prompt_tokens=data.get("prompt_eval_count"),
                completion_tokens=data.get("eval_count"), **native_call_timings(data))
    usage = None
    prompt_count, output_count = data.get("prompt_eval_count"), data.get("eval_count")
    if type(prompt_count) is int and type(output_count) is int and min(prompt_count, output_count) >= 0:
        usage = dict(prompt_tokens=prompt_count, completion_tokens=output_count,
                     total_tokens=prompt_count + output_count)
    return ChatCompletion(id='ice-local-extraction', created=int(time.time()), model=kwargs['model'],
                          object='chat.completion', usage=usage,
                          choices=[dict(index=0, finish_reason=data['done_reason'],
                                        message=dict(role='assistant', content=data.get('response')))])
