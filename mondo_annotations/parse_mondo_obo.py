#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Parse MONDO OBO file.

- MONDO ID equivalence (synonym EXACT, xref MONDO:equivalentTo)
- Name and synonyms per ID
- Rare-disease flag (subset: rare)
- Parents (is_a)
"""

import re
import json
from typing import Dict, List, Set
from collections import defaultdict


def parse_mondo_obo(file_path: str) -> Dict:
    """
    Parse MONDO OBO file.

    Args:
        file_path: Path to OBO file.

    Returns:
        Dict of disease info (id, name, synonyms, equivalent_ids, is_rare, parents).
    """
    diseases = {}
    current_disease = None

    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()

            # New [Term] block
            if line == '[Term]':
                if current_disease and not current_disease.get('skip') and current_disease.get('id'):
                    diseases[current_disease['id']] = current_disease
                current_disease = {
                    'id': None,
                    'name': None,
                    'synonyms': [],
                    'equivalent_ids': set(),
                    'is_rare': False,
                    'parents': [],
                    'skip': False  # skip non-MONDO (e.g. identifiers.org/hgnc/...)
                }
                continue

            if current_disease is None:
                continue

            # id
            if line.startswith('id:'):
                current_disease['id'] = line.split(':', 1)[1].strip()
                if not current_disease['id'].startswith('MONDO:'):
                    current_disease['skip'] = True
                    current_disease = None

            # name
            elif line.startswith('name:') and current_disease:
                current_disease['name'] = line.split(':', 1)[1].strip()

            # synonym: "name" TYPE [sources]
            elif line.startswith('synonym:') and current_disease:
                match = re.match(r'synonym:\s*"([^"]+)"\s+(\w+)', line)
                if match:
                    current_disease['synonyms'].append({
                        'name': match.group(1),
                        'type': match.group(2)
                    })

            # xref: ID {source=...}; treat as equivalent if source="MONDO:equivalentTo"
            elif line.startswith('xref:'):
                match = re.match(r'xref:\s*([^\s{]+)', line)
                if match and current_disease and 'source="MONDO:equivalentTo"' in line:
                    current_disease['equivalent_ids'].add(match.group(1))

            # subset: rare / rare_grouping / xxx_rare
            elif line.startswith('subset:') and current_disease:
                subset_value = line.split(':', 1)[1].strip()
                subset_name = subset_value.split()[0] if subset_value else ''
                if subset_name == 'rare' or subset_name == 'rare_grouping' or subset_name.endswith('_rare'):
                    current_disease['is_rare'] = True

            # is_a: MONDO:0002816 ! adrenal cortex disorder
            elif line.startswith('is_a:') and current_disease:
                match = re.match(r'is_a:\s*([^\s!]+)', line)
                if match:
                    current_disease['parents'].append(match.group(1))

    if current_disease and not current_disease.get('skip') and current_disease.get('id'):
        diseases[current_disease['id']] = current_disease

    for disease_id, disease_info in diseases.items():
        disease_info['equivalent_ids'] = list(disease_info['equivalent_ids'])

    return diseases


def generate_statistics(diseases: Dict) -> Dict:
    """
    Generate summary statistics.

    Args:
        diseases: Dict of disease info.

    Returns:
        Stats dict (total_diseases, rare_diseases, etc.).
    """
    stats = {
        'total_diseases': len(diseases),
        'rare_diseases': sum(1 for d in diseases.values() if d['is_rare']),
        'diseases_with_equivalents': sum(1 for d in diseases.values() if d['equivalent_ids']),
        'diseases_with_parents': sum(1 for d in diseases.values() if d['parents']),
        'total_equivalent_mappings': sum(len(d['equivalent_ids']) for d in diseases.values())
    }
    return stats


def save_results(diseases: Dict, stats: Dict, output_prefix: str):
    """
    Save parse results.

    Args:
        diseases: Dict of disease info.
        stats: Statistics dict.
        output_prefix: Output path prefix.
    """
    with open(f'{output_prefix}_full.json', 'w', encoding='utf-8') as f:
        json.dump(diseases, f, ensure_ascii=False, indent=2)
    with open(f'{output_prefix}_statistics.json', 'w', encoding='utf-8') as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)
    print(f"\nDone. Output:")
    print(f"  - Full: {output_prefix}_full.json")
    print(f"  - Stats: {output_prefix}_statistics.json")


def print_statistics(stats: Dict):
    """Print statistics summary."""
    print("\n" + "="*60)
    print("Statistics")
    print("="*60)
    print(f"Total diseases: {stats['total_diseases']:,}")
    print(f"Rare diseases: {stats['rare_diseases']:,}")
    print(f"Diseases with equivalent IDs: {stats['diseases_with_equivalents']:,}")
    print(f"Diseases with parents: {stats['diseases_with_parents']:,}")
    print(f"Total equivalent mappings: {stats['total_equivalent_mappings']:,}")
    print("="*60)


def main():
    """Main entry."""
    import os

    script_dir = os.path.dirname(os.path.abspath(__file__))
    input_file = os.path.join(script_dir, 'mondo-rare.obo')
    output_prefix = os.path.join(script_dir, 'mondo_parsed')

    print(f"Parsing: {input_file}")
    print("This may take a while...")

    diseases = parse_mondo_obo(input_file)
    stats = generate_statistics(diseases)
    print_statistics(stats)
    save_results(diseases, stats, output_prefix)

    print("\n" + "="*60)
    print("Sample (first 3 diseases):")
    print("="*60)
    for i, (mondo_id, disease_info) in enumerate(list(diseases.items())[:3]):
        print(f"\n{i+1}. {mondo_id}")
        print(f"   Name: {disease_info['name']}")
        print(f"   Synonyms: {len(disease_info['synonyms'])}")
        if disease_info['synonyms']:
            print(f"   Example: {disease_info['synonyms'][0]['name']} ({disease_info['synonyms'][0]['type']})")
        print(f"   Equivalent IDs: {len(disease_info['equivalent_ids'])}")
        if disease_info['equivalent_ids']:
            print(f"   Example: {', '.join(list(disease_info['equivalent_ids'])[:3])}")
        print(f"   Is rare: {'yes' if disease_info['is_rare'] else 'no'}")
        print(f"   Parents: {len(disease_info['parents'])}")
        if disease_info['parents']:
            print(f"   Example: {', '.join(disease_info['parents'])}")


if __name__ == '__main__':
    main()

