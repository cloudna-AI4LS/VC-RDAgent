#!/usr/bin/env python3
"""
Extract mappings between disorders and their disorder types from Orphanet
classification XML files and write the aggregated data to a JSON file.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import csv
from typing import Dict, Iterable, List, Optional, Set, Tuple
import xml.etree.ElementTree as ET


def _preferred_text(elements: Iterable[ET.Element], preferred_lang: str = "en") -> str:
    """
    Return the text matching the preferred language from a collection of <Name> elements.
    Fall back to the first non-empty text if no preferred language is present.
    """
    preferred = next(
        (elem.text.strip() for elem in elements if elem.attrib.get("lang") == preferred_lang and elem.text),
        None,
    )
    if preferred:
        return preferred

    fallback = next((elem.text.strip() for elem in elements if elem.text), "")
    return fallback


def _normalize_classification_name(name: str) -> str:
    prefix = "Orphanet classification of "
    if name.startswith(prefix):
        return name[len(prefix) :].strip()
    return name


def _extract_disorder(disorder_elem: ET.Element) -> Optional[Dict[str, Optional[str]]]:
    """
    Collect the core fields for a disorder element.
    """
    orpha_code = (disorder_elem.findtext("OrphaCode") or "").strip()
    if not orpha_code:
        return None

    name = _preferred_text(disorder_elem.findall("Name"))

    disorder_type_elem = disorder_elem.find("DisorderType")
    disorder_type_id = disorder_type_elem.attrib.get("id") if disorder_type_elem is not None else None
    disorder_type_name = (
        _preferred_text(disorder_type_elem.findall("Name")) if disorder_type_elem is not None else ""
    )

    return {
        "orpha_code": orpha_code,
        "name": name,
        "disorder_type_id": disorder_type_id,
        "disorder_type_name": disorder_type_name,
    }


def extract_from_xml(xml_path: Path) -> List[Dict[str, Optional[str]]]:
    """
    Parse a classification XML file and return disorder records annotated with classification metadata.
    """
    tree = ET.parse(xml_path)
    root = tree.getroot()

    records: List[Dict[str, Optional[str]]] = []

    classification_list = root.find("ClassificationList")
    if classification_list is None:
        return records

    def visit_node(
        node: ET.Element,
        category_stack: List[Dict[str, Optional[str]]],
        classification_id: Optional[str],
        classification_name: str,
        classification_orpha_number: str,
    ) -> None:
        disorder_elem = node.find("Disorder")
        if disorder_elem is None:
            return

        record = _extract_disorder(disorder_elem)
        if not record:
            return

        nearest_category = category_stack[-1] if category_stack else None
        record.update(
            {
                "nearest_category_orpha_code": nearest_category.get("orpha_code") if nearest_category else None,
                "nearest_category_name": nearest_category.get("name") if nearest_category else None,
                "classification_id": classification_id,
                "classification_name": classification_name,
                "classification_orpha_number": classification_orpha_number,
            }
        )
        records.append(record)

        is_category = (record.get("disorder_type_name") or "").lower() == "category"
        if is_category:
            category_stack.append(
                {"orpha_code": record["orpha_code"], "name": record.get("name")}
            )

        child_list = node.find("ClassificationNodeChildList")
        if child_list is not None:
            for child in child_list.findall("ClassificationNode"):
                visit_node(child, category_stack, classification_id, classification_name, classification_orpha_number)

        if is_category:
            category_stack.pop()

    for classification in classification_list.findall("Classification"):
        classification_id = classification.attrib.get("id")
        classification_name = _normalize_classification_name(_preferred_text(classification.findall("Name")))
        classification_orpha_number = (classification.findtext("OrphaNumber") or "").strip()

        root_list = classification.find("ClassificationNodeRootList")
        if root_list is None:
            continue

        for node in root_list.findall("ClassificationNode"):
            visit_node(node, [], classification_id, classification_name, classification_orpha_number)

    return records


def build_mapping(xml_files: Iterable[Path]) -> Dict[str, Dict[str, object]]:
    """
    Aggregate disorder records across XML files into a mapping keyed by OrphaCode.
    """
    mapping: Dict[str, Dict[str, object]] = {}

    for xml_file in xml_files:
        try:
            records = extract_from_xml(xml_file)
        except ET.ParseError as exc:
            raise RuntimeError(f"Failed to parse '{xml_file}': {exc}") from exc

        for record in records:
            orpha_code = record["orpha_code"]
            entry = mapping.setdefault(
                orpha_code,
                {
                    "orpha_code": orpha_code,
                    "names": set(),  # type: Set[str]
                    "disorder_type_ids": set(),  # type: Set[str]
                    "disorder_type_names": set(),  # type: Set[str]
                    "classifications": set(),  # type: Set[Tuple[str, str]]
                    "source_files": set(),  # type: Set[str]
                    "nearest_categories": set(),  # type: Set[Tuple[Optional[str], Optional[str]]]
                },
            )

            name = record.get("name")
            if name:
                entry["names"].add(name)

            disorder_type_id = record.get("disorder_type_id")
            if disorder_type_id:
                entry["disorder_type_ids"].add(disorder_type_id)

            disorder_type_name = record.get("disorder_type_name")
            if disorder_type_name:
                entry["disorder_type_names"].add(disorder_type_name)

            classification_id = record.get("classification_id")
            classification_name = record.get("classification_name")
            if classification_id or classification_name:
                entry["classifications"].add((classification_id or "", classification_name or ""))

            entry["source_files"].add(xml_file.name)
            nearest_category_orpha_code = record.get("nearest_category_orpha_code")
            nearest_category_name = record.get("nearest_category_name")
            if nearest_category_orpha_code or nearest_category_name:
                entry["nearest_categories"].add(
                    (
                        nearest_category_orpha_code or None,
                        nearest_category_name or None,
                    )
                )

    # Convert set fields to sorted lists for JSON serialization.
    for entry in mapping.values():
        entry["names"] = sorted(entry["names"])
        entry["disorder_type_ids"] = sorted(entry["disorder_type_ids"])
        entry["disorder_type_names"] = sorted(entry["disorder_type_names"])
        entry["classifications"] = sorted(entry["classifications"])
        entry["source_files"] = sorted(entry["source_files"])
        entry["nearest_categories"] = [
            {
                "orpha_code": code,
                "name": name,
            }
            for code, name in sorted(entry["nearest_categories"], key=lambda item: (item[0] or "", item[1] or ""))
        ]

    return dict(sorted(mapping.items(), key=lambda item: int(item[0]) if item[0].isdigit() else item[0]))


def _classification_to_label(classification: Tuple[str, str]) -> str:
    classification_id, classification_name = classification
    name = (classification_name or "").strip()
    identifier = (classification_id or "").strip()

    if name and identifier:
        return f"{name}"
    if name:
        return name
    return identifier


def write_categories_csv(mapping: Dict[str, Dict[str, object]], output_path: Path) -> None:
    """
    Write a categorization CSV mirroring the legacy format, with Category populated
    from classification information (joined by '|').
    """
    rows = []
    for index, entry in enumerate(mapping.values()):
        orpha_code = entry.get("orpha_code", "")
        disorder_name = entry.get("names", [])
        representative_name = disorder_name[0] if disorder_name else ""

        classifications = entry.get("classifications", [])
        category_values = [
            label for label in (_classification_to_label(classification) for classification in classifications) if label
        ]
        category = "|".join(category_values)

        rows.append(
            {
                "": index,
                "OrphaNumber": orpha_code,
                "Disorder_Name": representative_name,
                "Category": category,
            }
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as csv_file:
        fieldnames = ["", "OrphaNumber", "Disorder_Name", "Category"]
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract disorder to disorder-type mappings from Orphanet classification XML files."
    )
    default_base_dir = Path(__file__).resolve().parent
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=default_base_dir / "Classifications of rare diseases",
        help="Directory containing classification XML files.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=default_base_dir / "disease_type_mapping.json",
        help="Path for the output JSON file.",
    )
    parser.add_argument(
        "--categories-output",
        type=Path,
        default=default_base_dir / "categorization_of_orphanet_diseases.csv",
        help="Path for the output CSV categorization file.",
    )
    parser.add_argument(
        "--ensure-ascii",
        action="store_true",
        help="Force ASCII-only JSON output (default keeps UTF-8 characters).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    input_dir = args.input_dir
    if not input_dir.is_dir():
        raise FileNotFoundError(f"Input directory not found: {input_dir}")

    xml_files = sorted(path for path in input_dir.glob("*.xml") if path.is_file())
    if not xml_files:
        raise FileNotFoundError(f"No XML files found in: {input_dir}")

    mapping = build_mapping(xml_files)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as json_file:
        json.dump(
            {
                "metadata": {
                    "source_directory": str(input_dir),
                    "file_count": len(xml_files),
                    "record_count": len(mapping),
                },
                "data": mapping,
            },
            json_file,
            indent=2,
            ensure_ascii=args.ensure_ascii,
        )

    print(f"Saved {len(mapping)} disorder records to '{args.output}'")
    write_categories_csv(mapping, args.categories_output)
    print(f"Wrote categorization CSV to '{args.categories_output}'")


if __name__ == "__main__":
    main()

