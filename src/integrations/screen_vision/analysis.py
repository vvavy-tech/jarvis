"""On-demand, one-shot screen analysis via a separate Gemini request.

This path is deliberately decoupled from the realtime session's model: an
image is only ever sent to the model inside this dedicated, short-lived
``google.LLM`` request, the result comes back as plain text, and any failure
surfaces as a normal tool error. Nothing here touches the voice pipeline.
"""

from __future__ import annotations

import asyncio
import base64
import os
from io import BytesIO
from pathlib import Path
from typing import Any, Callable

from livekit.agents.llm import (
    ChatContext,
    ChatMessage,
    ImageContent,
    ToolError,
)
from PIL import Image

from integrations.screen_vision.capture import remove_temp_image

_MAX_LONG_EDGE = 1600
DEFAULT_MODEL = "gemini-3.6-flash"
_VISION_MODEL_CANDIDATES = (DEFAULT_MODEL, "gemini-2.0-flash")

_TIMEOUT_MESSAGE = (
    "Reading the screen is taking longer than expected. Try again in a "
    "moment, or ask me to look at the area around your cursor instead."
)
_FAILED_MESSAGE = (
    "I had trouble reading the screen just now. Please try again, or tell me "
    "exactly what you want me to look at."
)

_SYSTEM_PROMPT = (
    "You are JARVIS's screen-reading assistant. Answer only from what is "
    "actually visible in the image. Report on-screen text or error messages "
    "verbatim when asked. Describe layouts and states concisely. Never guess "
    "or invent content that is not visible."
)

LLMFactory = Callable[[], Any]


async def _collect_text(stream: Any) -> str:
    parts: list[str] = []
    async for chunk in stream.to_str_iterable():
        parts.append(str(chunk))
    return "".join(parts).strip()


def _image_to_data_url(image_path: Path) -> str:
    with Path(image_path).open("rb") as handle:
        image = Image.open(handle).convert("RGB")
    if image.width > _MAX_LONG_EDGE or image.height > _MAX_LONG_EDGE:
        image.thumbnail((_MAX_LONG_EDGE, _MAX_LONG_EDGE))
    buffer = BytesIO()
    image.save(buffer, format="PNG")
    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    return f"data:image/png;base64,{encoded}"


class VisionAnalyzer:
    """Runs a single text-answering Gemini request for one screenshot."""

    def __init__(
        self,
        llm_factory: LLMFactory | None = None,
        model: str | None = None,
        timeout_s: float | None = None,
    ) -> None:
        self._llm_factory = llm_factory
        self._model = model or os.environ.get("JARVIS_VISION_MODEL", DEFAULT_MODEL)
        default_timeout = float(os.environ.get("JARVIS_VISION_TIMEOUT_S", "45"))
        self._timeout = timeout_s or default_timeout
        self._llm: Any | None = None
        self._llm_model: str | None = None

    def configured(self) -> bool:
        return bool(os.environ.get("GOOGLE_API_KEY", "").strip())

    def _ensure_llm(self, model: str) -> Any:
        if self._llm is None or model != self._llm_model:
            if self._llm_factory is not None:
                self._llm = self._llm_factory()
            else:
                from livekit.plugins import google

                self._llm = google.LLM(model=model)
            self._llm_model = model
        return self._llm

    def _attempt_models(self) -> list[str]:
        """Ordered, de-duplicated candidates to try against Gemini.

        The user-supplied model (or the default) goes first; the fallback list
        heals from a retired model returning 404.
        """
        ordered = [self._model, *_VISION_MODEL_CANDIDATES]
        return list(dict.fromkeys(model for model in ordered if model))

    async def analyse(self, image_path: Path, question: str) -> str:
        """Describe an image based on ``question``. Returns plain text."""
        if not self.configured():
            raise ToolError(
                "Screen vision is not configured because no Gemini API key is "
                "set. Say you cannot see the screen and keep helping normally."
            )
        data_url = await asyncio.to_thread(_image_to_data_url, Path(image_path))
        requested = (question or "").strip() or "Describe what is on screen."
        message = ChatMessage(
            role="user",
            content=[
                _SYSTEM_PROMPT,
                "\nQuestion: " + requested,
                ImageContent(image=data_url),
            ],
        )
        for attempt in self._attempt_models():
            try:
                stream = self._ensure_llm(attempt).chat(
                    chat_ctx=ChatContext(items=[message])
                )
                text = await asyncio.wait_for(
                    _collect_text(stream), timeout=self._timeout
                )
            except TimeoutError:
                raise ToolError(_TIMEOUT_MESSAGE) from None
            except Exception as exc:
                # A retired/misconfigured model returns 404; retry once with
                # the known-good default before giving up.
                if getattr(exc, "status_code", None) != 404:
                    raise ToolError(_FAILED_MESSAGE) from None
                continue
            break
        else:
            raise ToolError(_FAILED_MESSAGE) from None
        if not text:
            raise ToolError("Screen analysis returned no useful description.")
        return text


async def analyse_captured(
    analyzer: VisionAnalyzer,
    image_path: Path,
    question: str,
    *,
    owned: bool = False,
) -> str:
    """Analyse a screenshot and always clean up JARVIS-created temp files."""
    try:
        return await analyzer.analyse(image_path, question)
    finally:
        if owned:
            await asyncio.to_thread(remove_temp_image, image_path)
