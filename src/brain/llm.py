"""Thin LiteLLM wrapper — three async functions, no classes.

Reads model strings from config and forwards to LiteLLM.
"""

from __future__ import annotations

import base64
import logging
from collections.abc import AsyncIterator
from pathlib import Path

import litellm

from brain.config import get_settings

log = logging.getLogger(__name__)


async def complete(
    messages: list[dict],
    *,
    model: str | None = None,
    stream: bool = False,
    **kwargs,
) -> str | AsyncIterator[str]:
    """Chat completion using the active profile's chat_model.

    When stream=True, returns an async iterator of string chunks.
    """
    cfg = get_settings().llm
    model = model or cfg.chat_model
    temperature = kwargs.pop("temperature", cfg.temperature)
    max_tokens = kwargs.pop("max_tokens", cfg.max_tokens)

    response = await litellm.acompletion(
        model=model,
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens,
        stream=stream,
        **kwargs,
    )

    if stream:

        async def _stream():
            async for chunk in response:
                delta = chunk.choices[0].delta.content
                if delta:
                    yield delta

        return _stream()

    return response.choices[0].message.content


async def embed(
    texts: list[str], *, model: str | None = None, batch_size: int = 128
) -> list[list[float]]:
    """Embed texts using the active profile's embed_model.

    Automatically batches large inputs to stay within API limits.
    """
    cfg = get_settings().llm
    model = model or cfg.embed_model

    if len(texts) <= batch_size:
        response = await litellm.aembedding(model=model, input=texts)
        return [item["embedding"] for item in response.data]

    # Batch large inputs
    all_embeddings: list[list[float]] = []
    total_batches = -(-len(texts) // batch_size)  # ceil division
    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        batch_num = i // batch_size + 1
        log.info("Embedding batch %d/%d (%d texts)", batch_num, total_batches, len(batch))
        response = await litellm.aembedding(model=model, input=batch)
        all_embeddings.extend(item["embedding"] for item in response.data)
    return all_embeddings


async def transcribe_image(
    image_source: str | bytes | Path,
    *,
    prompt: str = (
        "Transcribe all text and content visible in this image faithfully. "
        "Preserve structure (headings, lists, tables). "
        "Return the transcription as clean text, not a summary."
    ),
    model: str | None = None,
) -> str:
    """Transcribe an image to text using a vision model.

    image_source can be:
      - a base64 string
      - raw bytes
      - a Path to an image file
    """
    cfg = get_settings().llm
    model = model or cfg.vision_model

    # Normalise to base64
    if isinstance(image_source, Path):
        image_source = image_source.read_bytes()
    if isinstance(image_source, bytes):
        image_source = base64.b64encode(image_source).decode("utf-8")

    response = await litellm.acompletion(
        model=model,
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{image_source}",
                            "detail": "high",
                        },
                    },
                ],
            }
        ],
        max_tokens=cfg.max_tokens,
    )
    return response.choices[0].message.content
