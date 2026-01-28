#!/usr/bin/env python3
"""
Extract phenotype embeddings from final_bio_embeddings.csv
and save to phe2embedding_recomputed.json in the same format as phe2embedding.json
"""

import argparse
import csv
import json
import sys
from pathlib import Path

def process_csv_to_json(csv_path, output_path):
    """
    Process CSV file and extract phenotype embeddings (rows where id starts with 'HP:')
    Save to JSON file in the same format as phe2embedding.json
    """
    phe2embedding = {}
    
    print(f"Reading CSV file: {csv_path}")
    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.reader(f)
        # Skip header
        header = next(reader)
        print(f"CSV header: {header}")
        
        count = 0
        phenotype_count = 0
        
        for row in reader:
            count += 1
            if count % 10000 == 0:
                print(f"Processed {count} rows, found {phenotype_count} phenotypes...")
            
            if len(row) < 3:
                continue
            
            node_type = row[0].strip()
            phenotype_id = row[1].strip()
            embedding_str = row[2].strip()
            
            # Check if this is a phenotype (id starts with 'HP:')
            if phenotype_id.startswith('HP:'):
                try:
                    # Parse embedding string (comma-separated floats) to list of floats
                    embedding = [float(x) for x in embedding_str.split(',')]
                    phe2embedding[phenotype_id] = embedding
                    phenotype_count += 1
                except ValueError as e:
                    print(f"Warning: Could not parse embedding for {phenotype_id}: {e}")
                    continue
    
    print(f"\nTotal rows processed: {count}")
    print(f"Total phenotypes found: {phenotype_count}")
    
    # Save to JSON file
    print(f"\nSaving to JSON file: {output_path}")
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(phe2embedding, f, indent=4, ensure_ascii=False)
    
    print(f"Done! Saved {len(phe2embedding)} phenotype embeddings to {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract phenotype embeddings from CSV file")
    parser.add_argument("csv_path", type=str, help="Path to the CSV file containing embeddings")
    parser.add_argument("-o", "--output", type=str, default=None, 
                        help="Path to output JSON file (default: phe2embedding_recomputed.json in same directory as CSV)")
    
    args = parser.parse_args()
    
    # Set paths
    csv_path = Path(args.csv_path)
    if args.output:
        output_path = Path(args.output)
    else:
        output_path = "phe2embedding_recomputed.json"
    
    if not csv_path.exists():
        print(f"Error: CSV file not found at {csv_path}")
        sys.exit(1)
    
    process_csv_to_json(csv_path, output_path)

