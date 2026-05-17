"""
Meta-feature extraction for property listing descriptions.

This script uses an Ollama model to classify property descriptions into a
predefined set of meta feature categories, then saves the enriched dataset
with one-hot encoded category columns.
"""

import argparse
import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List

import ollama
import pandas as pd
from tqdm import tqdm

from meta_categories import META_FEATURE_CATEGORIES


@dataclass
class MetaFeatureExtractorConfig:
    """Configuration values for the meta-feature extractor."""

    model_name: str = "llama3.1"
    city: str = "gold_coast"
    save_every: int = 50
    max_desc_len: int = 1500

    input_path: str = field(init=False)
    output_path: str = field(init=False)
    checkpoint_path: str = field(init=False)

    def __post_init__(self):
        self.input_path = f"sample_data/{self.city}.csv"
        self.output_path = f"output_data/{self.city}_{self.model_name}_final.csv"
        self.checkpoint_path = f"output_data/{self.city}_{self.model_name}_checkpoint.csv"


def setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def ensure_output_dirs(output_path: str) -> None:
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)


def load_input_data(input_path: str) -> pd.DataFrame:
    df = pd.read_csv(input_path)
    df["LISTING_ID"] = df["LISTING_ID"].astype(str)
    return df


def warm_up_model(model_name: str) -> None:
    logging.info("Warming up Ollama model: %s", model_name)
    ollama.chat(model=model_name, messages=[{"role": "user", "content": "warmup"}])


def build_prompt(listing_id: str, description: str) -> str:
    return f"""
Classify the property description into one or more of these categories:
{META_FEATURE_CATEGORIES}

Description:
{description}

Return ONLY JSON:
{{
  "LISTING_ID": "{listing_id}",
  "categories": []
}}
"""


def parse_llm_response(content: str) -> Dict[str, Any]:
    """Parse the model output into JSON and normalize the returned categories."""
    data = json.loads(content)
    if not isinstance(data, dict):
        raise ValueError("Parsed response is not a JSON object")

    output = {
        "LISTING_ID": str(data.get("LISTING_ID", "")),
        "categories": []
    }

    categories = data.get("categories", [])
    if isinstance(categories, list):
        output["categories"] = [str(item).strip() for item in categories if isinstance(item, str)]

    return output


def classify_listing(model_name: str, listing_id: str, description: str, max_desc_len: int) -> Dict[str, Any]:
    """Classify a single listing description into meta feature categories."""
    description = (description or "")[:max_desc_len]
    prompt = build_prompt(listing_id, description)
    response = ollama.chat(
        model=model_name,
        messages=[{"role": "user", "content": prompt}],
        format="json",
        options={
            "temperature": 0,
            "num_ctx": 2048,
            "num_predict": 150,
        },
    )
    return parse_llm_response(str(response["message"]["content"]))


def save_checkpoint(results: List[Dict[str, Any]], checkpoint_path: str) -> None:
    logging.info("Saving checkpoint to %s", checkpoint_path)
    pd.DataFrame(results).to_csv(checkpoint_path, index=False)


def encode_meta_features(results: List[Dict[str, Any]]) -> pd.DataFrame:
    """Expand the list of categories into one-hot encoded meta-feature columns."""
    features_df = pd.DataFrame(results)
    for category in META_FEATURE_CATEGORIES:
        features_df[category] = features_df["categories"].apply(
            lambda values: 1 if isinstance(values, list) and category in values else 0
        )
    return features_df.drop(columns=["categories"])


def main() -> None:
    setup_logging()

    parser = argparse.ArgumentParser(description="Meta-feature extraction for property descriptions")
    parser.add_argument("--city", default="gold_coast", help="City name used to build the input file path")
    parser.add_argument("--model", default="llama3.1", help="Ollama model name")
    parser.add_argument("--save-every", type=int, default=50, help="Checkpoint frequency")
    parser.add_argument("--max-desc-len", type=int, default=1500, help="Maximum description length to feed the model")
    args = parser.parse_args()

    global config
    config = MetaFeatureExtractorConfig(
        model_name=args.model,
        city=args.city,
        save_every=args.save_every,
        max_desc_len=args.max_desc_len,
    )

    ensure_output_dirs(config.output_path)
    df = load_input_data(config.input_path)
    logging.info("Loaded %d rows from %s", len(df), config.input_path)

    processed_ids = set()
    results: List[Dict[str, Any]] = []

    checkpoint_path = config.checkpoint_path
    if Path(checkpoint_path).exists():
        checkpoint_df = pd.read_csv(checkpoint_path)
        processed_ids = set(checkpoint_df["LISTING_ID"].astype(str))
        results = checkpoint_df.to_dict("records")
        logging.info("Resuming from checkpoint. %d rows already processed.", len(processed_ids))

    warm_up_model(config.model_name)

    for _, row in tqdm(df.iterrows(), total=len(df), desc="Classifying listings"):
        listing_id = str(row["LISTING_ID"])
        if listing_id in processed_ids:
            continue

        description = str(row.get("DESCRIPTION", ""))
        try:
            classification = classify_listing(
                config.model_name,
                listing_id,
                description,
                config.max_desc_len,
            )
        except Exception as exc:
            logging.warning("Failed to classify listing %s: %s", listing_id, exc)
            classification = {"LISTING_ID": listing_id, "categories": []}

        results.append(classification)
        processed_ids.add(listing_id)

        if len(results) % config.save_every == 0:
            save_checkpoint(results, checkpoint_path)

    checkpoint_df = encode_meta_features(results)
    df_final = df.merge(checkpoint_df, on="LISTING_ID", how="left")
    df_final.to_csv(config.output_path, index=False)
    logging.info("Meta-features generated and saved to %s", config.output_path)


if __name__ == "__main__":
    main()
