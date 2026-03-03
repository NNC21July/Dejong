from __future__ import annotations

import os
from typing import Any

try:
    from dotenv import load_dotenv
except ImportError:  # fallback if python-dotenv is not installed
    def load_dotenv(*_args, **_kwargs):
        return False


def get_openai_client() -> Any | None:
    load_dotenv()
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        return None

    from openai import OpenAI

    return OpenAI(api_key=api_key)
