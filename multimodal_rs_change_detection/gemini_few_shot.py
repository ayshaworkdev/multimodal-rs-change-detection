"""Few-shot change detection workflow powered by the Gemini Vision API.

This module wraps the multimodal Gemini client so that we can feed a support set
of pre/post remote-sensing image pairs (few-shot examples) along with a query
pair and obtain structured predictions.  It constructs prompts that highlight
representative change categories from the LEVIR-CC dataset and enforces a
machine-readable JSON schema for downstream evaluation.

The implementation intentionally avoids making any network calls during import
so that it can be reused in offline unit tests.  Actual API requests are
performed only when :meth:`GeminiFewShotChangeDetector.analyze_dataset` is
invoked.
"""

from __future__ import annotations

import base64
import json
import os
import textwrap
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

from dotenv import load_dotenv

from . import CHANGE_CATEGORIES, FEW_SHOT_EXAMPLES

try:
    import google.generativeai as genai
except ModuleNotFoundError as exc:  # pragma: no cover - exercised only in runtime envs
    raise ModuleNotFoundError(
        "google-generativeai is required. Install it via `pip install google-generativeai`."
    ) from exc


@dataclass(frozen=True)
class DetectionResult:
    """Structured output returned by Gemini for a single image pair."""

    image_pair: str
    classification: str
    caption: str
    confidence: float
    rationale: str
    category: Optional[str]

    def to_dict(self) -> Dict[str, object]:
        return {
            "image_pair": self.image_pair,
            "classification": self.classification,
            "caption": self.caption,
            "confidence": self.confidence,
            "rationale": self.rationale,
            "category": self.category,
        }


class GeminiFewShotChangeDetector:
    """Runs few-shot change detection experiments with Gemini Vision."""

    MODEL_NAME = "gemini-1.5-pro-latest"

    def __init__(
        self,
        dataset_path: Path,
        output_path: Path,
        num_samples: Optional[int] = None,
        env_path: Optional[Path] = None,
        support_dataset_path: Optional[Path] = None,
    ) -> None:
        self.dataset_path = dataset_path
        self.output_path = output_path
        self.num_samples = num_samples
        self.support_dataset_path = support_dataset_path or dataset_path
        self._no_change_captions: List[str] = [
            example["caption"] for example in FEW_SHOT_EXAMPLES["no_change"]
        ]
        self._no_change_index = 0

        self.api_key = self._load_api_key(env_path)
        genai.configure(api_key=self.api_key)
        self.model = genai.GenerativeModel(self.MODEL_NAME)
        self._support_parts = self._build_support_parts()

    @staticmethod
    def _load_api_key(env_path: Optional[Path]) -> str:
        """Load the Gemini API key, searching common ``.env`` locations."""

        candidate_paths: List[Path] = []
        if env_path:
            candidate_paths.append(Path(env_path))
        module_dir = Path(__file__).resolve().parent
        candidate_paths.extend(
            [
                Path.cwd() / ".env",
                module_dir / ".env",
                module_dir.parent / ".env",
            ]
        )

        seen = set()
        unique_candidates: List[Path] = []
        for candidate in candidate_paths:
            if candidate in seen:
                continue
            seen.add(candidate)
            unique_candidates.append(candidate)

        loaded_from: Optional[Path] = None
        for candidate in unique_candidates:
            if candidate.exists():
                load_dotenv(candidate)
                loaded_from = candidate
                break

        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            search_hint = ", ".join(str(path) for path in unique_candidates)
            raise ValueError(
                "GOOGLE_API_KEY not found. Set it as an environment variable or place it in a .env file "
                f"located at one of: {search_hint}"
            )

        if loaded_from:
            os.environ.setdefault("GEMINI_ENV_LOADED_FROM", str(loaded_from))

        return api_key

    def analyze_dataset(self) -> Dict[str, object]:
        """Iterate over A/B image pairs in ``self.dataset_path`` and call Gemini."""

        pairs = sorted(self._iter_image_pairs())
        if self.num_samples:
            pairs = pairs[: self.num_samples]

        captions_used: Dict[str, int] = {}
        results: List[DetectionResult] = []

        for pre_path, post_path in pairs:
            response = self._query_model(pre_path, post_path)
            detection = self._parse_response(response, pre_path.name)
            detection = self._assign_no_change_caption(detection)
            detection = self._ensure_category_caption(detection)
            detection = self._enforce_unique_caption(detection, captions_used)
            results.append(detection)

        summary = self._build_summary(results)
        payload = {"summary": summary, "results": [r.to_dict() for r in results]}

        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        with self.output_path.open("w", encoding="utf-8") as fp:
            json.dump(payload, fp, ensure_ascii=False, indent=2)

        return payload

    def _iter_image_pairs(self) -> Iterable[Tuple[Path, Path]]:
        """Yield aligned image pairs from ``A`` and ``B`` sub-directories."""

        a_dir = self.dataset_path / "A"
        b_dir = self.dataset_path / "B"
        if not a_dir.exists() or not b_dir.exists():
            raise FileNotFoundError(
                f"Expected 'A' and 'B' folders inside {self.dataset_path!s}, but one or both are missing."
            )

        for pre_path in sorted(a_dir.glob("*.png")):
            post_path = b_dir / pre_path.name
            if post_path.exists():
                yield pre_path, post_path

    def _build_support_parts(self) -> List[dict]:
        """Construct multimodal content parts that encode few-shot examples."""

        parts: List[dict] = [
            {
                "text": textwrap.dedent(
                    """
                    You are an expert in remote-sensing change detection.
                    Always respond with JSON containing the keys classification, category, caption,
                    confidence, and rationale. Classification must be either "change" or "no_change".
                    Here are labeled examples that demonstrate the expected reasoning style.
                    """
                ).strip()
            }
        ]

        for example in FEW_SHOT_EXAMPLES["no_change"]:
            parts.extend(self._serialize_support_example("no_change", example))
        for example in FEW_SHOT_EXAMPLES["change"]:
            parts.extend(self._serialize_support_example("change", example))

        parts.append(
            {
                "text": textwrap.dedent(
                    """
                    After the support set you will receive a query pair. Analyse it step-by-step
                    but output only JSON following the same schema. Do not include extra prose.
                    """
                ).strip()
            }
        )

        return parts

    def _serialize_support_example(self, classification: str, example: Dict[str, str]) -> List[dict]:
        """Serialize a single few-shot example into Gemini content parts."""

        pre_path, post_path = self._resolve_support_image_paths(example["image_pair"])
        payload = self._support_example_payload(classification, example)

        header = (
            f"Example ({classification}{('/' + example['category']) if classification == 'change' else ''},"
            f" {example['image_pair']}):"
        )

        return [
            {"text": header},
            {
                "inline_data": {
                    "mime_type": self._guess_mime_type(pre_path),
                    "data": self._encode_image(pre_path),
                }
            },
            {
                "inline_data": {
                    "mime_type": self._guess_mime_type(post_path),
                    "data": self._encode_image(post_path),
                }
            },
            {
                "text": f"Expected JSON: {json.dumps(payload, ensure_ascii=False)}",
            },
        ]

    def _resolve_support_image_paths(self, image_name: str) -> Tuple[Path, Path]:
        """Return the (pre, post) paths for a support example image."""

        pre_path = self.support_dataset_path / "A" / image_name
        post_path = self.support_dataset_path / "B" / image_name
        missing = [str(path) for path in (pre_path, post_path) if not path.exists()]
        if missing:
            raise FileNotFoundError(
                "Support example images are missing. Expected both pre and post files at: "
                + ", ".join(missing)
            )
        return pre_path, post_path

    @staticmethod
    def _support_example_payload(classification: str, example: Dict[str, str]) -> Dict[str, object]:
        """Return a canonical payload string for the few-shot prompt."""

        if classification == "change":
            rationale = (
                f"Significant {example['category']} alterations are visible: {example['caption']}"
            )
            category = example["category"]
            confidence = 0.95
        else:
            rationale = "No noticeable structural differences between the two images."
            category = None
            confidence = 0.9

        return {
            "classification": classification,
            "category": category,
            "caption": example["caption"],
            "confidence": confidence,
            "rationale": rationale,
        }

    def _query_model(self, pre_path: Path, post_path: Path) -> genai.types.content_types.GenerateContentResponse:
        """Call Gemini with the few-shot prompt and query image pair."""

        parts = [*self._support_parts]
        parts.extend(
            [
                {
                    "text": textwrap.dedent(
                        f"""
                        Now analyze the query pair.
                        Return JSON only, no prose.
                        Image identifier: {pre_path.name}.
                        Ensure captions mention one of these change categories when \"classification\" is \"change\":
                        {', '.join(sorted(CHANGE_CATEGORIES))}.
                        """
                    ).strip()
                },
                {
                    "inline_data": {
                        "mime_type": self._guess_mime_type(pre_path),
                        "data": self._encode_image(pre_path),
                    }
                },
                {
                    "inline_data": {
                        "mime_type": self._guess_mime_type(post_path),
                        "data": self._encode_image(post_path),
                    }
                },
            ]
        )

        return self.model.generate_content(parts, safety_settings=self._safety_settings(), stream=False)

    @staticmethod
    def _guess_mime_type(path: Path) -> str:
        if path.suffix.lower() in {".jpg", ".jpeg"}:
            return "image/jpeg"
        return "image/png"

    @staticmethod
    def _encode_image(path: Path) -> str:
        return base64.b64encode(path.read_bytes()).decode("utf-8")

    @staticmethod
    def _safety_settings() -> List[dict]:
        return [
            {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_NONE"},
            {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_NONE"},
            {"category": "HARM_CATEGORY_SEXUAL", "threshold": "BLOCK_NONE"},
            {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_NONE"},
            {"category": "HARM_CATEGORY_SELF_HARM", "threshold": "BLOCK_NONE"},
        ]

    def _parse_response(self, response: genai.types.content_types.GenerateContentResponse, image_name: str) -> DetectionResult:
        """Extract :class:`DetectionResult` from the Gemini response payload."""

        if not response.candidates:
            raise ValueError("Gemini returned no candidates")

        first_candidate = response.candidates[0]
        text_parts = []
        for part in first_candidate.content.parts:
            if getattr(part, "text", None):
                text_parts.append(part.text)
        if not text_parts:
            raise ValueError(f"Gemini response for {image_name} did not contain text content")
        text = "\n".join(text_parts).strip()
        try:
            payload = json.loads(text)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Gemini response for {image_name} was not valid JSON: {text}") from exc

        classification = payload.get("classification", "").lower()
        if classification not in {"change", "no_change"}:
            raise ValueError(f"Unexpected classification '{classification}' for {image_name}")

        category = payload.get("category")
        if classification == "change":
            if not category or category not in CHANGE_CATEGORIES:
                raise ValueError(
                    f"Gemini must provide a valid change category for {image_name}; got {category!r}"
                )
        else:
            category = None

        raw_confidence = payload.get("confidence", 0.0)
        try:
            confidence = float(raw_confidence)
        except (TypeError, ValueError):
            confidence = 0.0
        confidence = max(0.0, min(1.0, confidence))

        rationale = str(payload.get("rationale", "")).strip() or "Model did not provide a rationale."

        return DetectionResult(
            image_pair=image_name,
            classification=classification,
            caption=str(payload.get("caption", "")).strip(),
            confidence=confidence,
            rationale=rationale,
            category=category,
        )

    def _assign_no_change_caption(self, detection: DetectionResult) -> DetectionResult:
        if detection.classification != "no_change":
            return detection
        caption = self._next_no_change_caption()
        return replace(detection, caption=caption)

    def _next_no_change_caption(self) -> str:
        if not self._no_change_captions:
            return "No change detected in the scene."
        caption = self._no_change_captions[self._no_change_index % len(self._no_change_captions)]
        self._no_change_index += 1
        return caption

    @staticmethod
    def _ensure_category_caption(detection: DetectionResult) -> DetectionResult:
        if detection.classification != "change" or not detection.category:
            return detection
        if detection.category.lower() in detection.caption.lower():
            return detection
        fallback_caption = (
            f"{detection.caption} ({detection.category})"
            if detection.caption
            else f"{detection.category.capitalize()} change detected."
        )
        return replace(detection, caption=fallback_caption)

    def _enforce_unique_caption(
        self, detection: DetectionResult, captions_used: Dict[str, int]
    ) -> DetectionResult:
        """Guarantee that change captions stay unique across outputs."""

        if not detection.caption:
            return detection

        if detection.classification == "no_change":
            return detection

        key = detection.caption.lower().strip()
        count = captions_used.get(key, 0)
        captions_used[key] = count + 1
        if count == 0:
            return detection

        base_caption = f"{detection.category.capitalize()} change detected in {Path(detection.image_pair).stem}"  # type: ignore[arg-type]
        unique_caption = base_caption
        suffix = 1
        while unique_caption.lower() in captions_used:
            suffix += 1
            unique_caption = f"{base_caption} ({suffix})"
        captions_used[unique_caption.lower()] = 1
        return replace(detection, caption=unique_caption)

    def _build_summary(self, results: Iterable[DetectionResult]) -> Dict[str, object]:
        total = 0
        change = 0
        no_change = 0
        for result in results:
            total += 1
            if result.classification == "change":
                change += 1
            else:
                no_change += 1

        return {
            "total_images": total,
            "change_count": change,
            "no_change_count": no_change,
        }


def _prime_environment(env_path: Optional[str]) -> None:
    """Load environment variables from common ``.env`` locations."""

    candidate_paths: List[Path] = []
    if env_path:
        candidate_paths.append(Path(env_path))

    candidate_paths.extend(
        [
            Path.cwd() / ".env",
            Path(__file__).resolve().parent / ".env",
        ]
    )

    for candidate in candidate_paths:
        if candidate.exists():
            load_dotenv(candidate, override=False)


def _resolve_dataset_path(dataset_path: Optional[str]) -> Path:
    """Resolve the dataset directory, falling back to ``LEVIR_CC_DATASET``."""

    if dataset_path:
        resolved = Path(dataset_path).expanduser()
    else:
        env_value = os.getenv("LEVIR_CC_DATASET")
        if not env_value:
            raise ValueError(
                "Dataset path not provided. Pass a positional argument or set LEVIR_CC_DATASET in your environment/.env file."
            )
        resolved = Path(env_value).expanduser()

    if not resolved.exists():
        raise FileNotFoundError(
            f"Dataset path {resolved!s} does not exist. Check the value provided or the LEVIR_CC_DATASET environment variable."
        )

    return resolved


def _resolve_support_dataset_path(
    support_dataset_path: Optional[str], dataset_path: Path
) -> Path:
    """Determine which dataset should supply the few-shot support images."""

    if support_dataset_path:
        resolved = Path(support_dataset_path).expanduser()
    else:
        env_value = os.getenv("LEVIR_CC_SUPPORT_DATASET")
        if env_value:
            resolved = Path(env_value).expanduser()
        else:
            resolved = dataset_path

    if not resolved.exists():
        raise FileNotFoundError(
            f"Support dataset path {resolved!s} does not exist. Update --support-dataset-path or LEVIR_CC_SUPPORT_DATASET."
        )

    return resolved


def run_cli(
    dataset_path: Optional[str] = None,
    output_dir: str = "results/few_shot_gemini",
    num_samples: Optional[int] = None,
    output_name: str = "gemini-fewshots-result1.json",
    env_path: Optional[str] = None,
    support_dataset_path: Optional[str] = None,
) -> Dict[str, object]:
    """Convenience CLI entry point used by ``python -m`` invocation."""

    _prime_environment(env_path)
    resolved_dataset_path = _resolve_dataset_path(dataset_path)
    resolved_support_path = _resolve_support_dataset_path(
        support_dataset_path, resolved_dataset_path
    )

    detector = GeminiFewShotChangeDetector(
        dataset_path=resolved_dataset_path,
        output_path=Path(output_dir) / output_name,
        num_samples=num_samples,
        env_path=Path(env_path) if env_path else None,
        support_dataset_path=resolved_support_path,
    )
    return detector.analyze_dataset()


if __name__ == "__main__":  # pragma: no cover - manual execution entry point
    import argparse

    parser = argparse.ArgumentParser(description="Few-shot change detection with Gemini Vision")
    parser.add_argument(
        "dataset_path",
        nargs="?",
        default=None,
        help=(
            "Optional path to the LEVIR-CC dataset containing 'A' and 'B' sub-folders. "
            "If omitted, the value is read from the LEVIR_CC_DATASET environment variable."
        ),
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=None,
        help="Limit the number of image pairs processed (useful for smoke tests).",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="results/few_shot_gemini",
        help="Directory where the JSON report will be stored.",
    )
    parser.add_argument(
        "--output-name",
        type=str,
        default="gemini-fewshots-result1.json",
        help="Filename for the JSON report (within --output-dir).",
    )
    parser.add_argument(
        "--env-path",
        type=str,
        default=None,
        help="Optional path to a .env file containing GOOGLE_API_KEY.",
    )
    parser.add_argument(
        "--support-dataset-path",
        type=str,
        default=None,
        help="Optional dataset path for few-shot support examples (defaults to dataset_path).",
    )

    args = parser.parse_args()
    payload = run_cli(
        dataset_path=args.dataset_path,
        output_dir=args.output_dir,
        num_samples=args.num_samples,
        output_name=args.output_name,
        env_path=args.env_path,
        support_dataset_path=args.support_dataset_path,
    )

    print(json.dumps(payload["summary"], indent=2))
