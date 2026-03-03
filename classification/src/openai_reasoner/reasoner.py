from __future__ import annotations

import base64
import json
from pathlib import Path


def _image_to_data_url(image_path: Path) -> str:
    b64 = base64.b64encode(image_path.read_bytes()).decode("utf-8")
    return f"data:image/jpeg;base64,{b64}"


def _validate_response(payload: dict) -> dict:
    required = ["context_score", "scenario", "confidence", "rationale"]
    for key in required:
        if key not in payload:
            raise ValueError(f"Missing key in OpenAI response: {key}")

    payload["context_score"] = max(0.0, min(1.0, float(payload["context_score"])))
    payload["confidence"] = max(0.0, min(1.0, float(payload["confidence"])))
    if payload["scenario"] not in {"Emergency", "Hazard", "Elevated Risk", "No Fire Risk"}:
        payload["scenario"] = "Elevated Risk"
    if not isinstance(payload["rationale"], list):
        payload["rationale"] = [str(payload["rationale"])]

    return payload


def reason_with_openai(
    client,
    model: str,
    image_path: Path,
    metadata: dict,
    metrics: dict,
) -> dict:
    image_url = _image_to_data_url(image_path)

    system = (
        "You are a fire-risk context reasoner. "
        "Return JSON only with keys: context_score (0..1), scenario, confidence (0..1), rationale (list of short strings). "
        "Scenario must be one of Emergency, Hazard, Elevated Risk, No Fire Risk."
    )

    user = {
        "metadata": metadata,
        "metrics": metrics,
        "instruction": "Use both image and metrics to estimate contextual fire risk severity.",
    }

    resp = client.chat.completions.create(
        model=model,
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": system},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": json.dumps(user)},
                    {"type": "image_url", "image_url": {"url": image_url}},
                ],
            },
        ],
    )

    content = resp.choices[0].message.content
    payload = json.loads(content)
    return _validate_response(payload)
