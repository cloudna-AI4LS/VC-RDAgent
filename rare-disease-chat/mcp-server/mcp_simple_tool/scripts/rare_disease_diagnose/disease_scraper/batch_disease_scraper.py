#!/usr/bin/env python3
"""
Batch Disease Description Scraper

This script reads disease IDs from a JSONL file and uses the HPO scraper
to get disease descriptions, then saves the results to a JSON file.

The script supports resuming interrupted scraping sessions by skipping
already processed disease IDs found in the output file.

Usage:
    python batch_disease_scraper.py [options]

Examples:
    # Basic usage - will skip already processed diseases
    python batch_disease_scraper.py --input phenotype_disease_case_library.jsonl --output disease_descriptions.json
    
    # Reprocess all diseases (ignore existing results)
    python batch_disease_scraper.py --input data.jsonl --output results.json --no-skip-existing
    
    # Process with custom delay and retry settings
    python batch_disease_scraper.py --input data.jsonl --delay 2 --max-retries 5
"""

import json
import argparse
import sys
import os
from collections import defaultdict
from datetime import datetime
import time

# Import the disease scraper
from disease_scraper import get_disease_description

def load_disease_ids_from_jsonl(file_path):
    """
    Load disease IDs from JSONL file
    
    Args:
        file_path (str): Path to the JSONL file
        
    Returns:
        set: Set of unique disease IDs
    """
    disease_ids = set()
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                try:
                    data = json.loads(line.strip())
                    if 'RareDisease' in data and data['RareDisease']:
                        for disease_id in data['RareDisease']:
                            if disease_id:  # Skip empty strings
                                disease_id = disease_id.strip()  # Remove leading/trailing whitespace
                                if not disease_id.startswith('CCRD') and not disease_id.startswith('DECIPHER'):  # Skip CCRD IDs
                                    disease_ids.add(disease_id)
                except json.JSONDecodeError as e:
                    print(f"Warning: Invalid JSON on line {line_num}: {e}")
                    continue
                except Exception as e:
                    print(f"Warning: Error processing line {line_num}: {e}")
                    continue
    except FileNotFoundError:
        print(f"Error: File not found: {file_path}")
        return None
    except Exception as e:
        print(f"Error reading file: {e}")
        return None
    
    return disease_ids

def load_existing_results(output_file):
    """
    Load existing results from output file
    
    Args:
        output_file (str): Path to the output JSON file
        
    Returns:
        tuple: (existing_results, existing_failed_ids, existing_stats)
    """
    if not os.path.exists(output_file):
        return {}, [], {}
    
    try:
        with open(output_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        results = data.get('results', {})
        failed_ids = data.get('failed_ids', [])
        stats = data.get('statistics', {})
        
        print(f"Found existing output file: {output_file}")
        print(f"  - Already processed: {len(results)} diseases")
        print(f"  - Previously failed: {len(failed_ids)} diseases")
        
        return results, failed_ids, stats
        
    except Exception as e:
        print(f"Warning: Could not load existing results from {output_file}: {e}")
        return {}, [], {}

def scrape_disease_descriptions(disease_ids, delay=1, max_retries=3, output_file=None, existing_results=None, existing_failed_ids=None, existing_stats=None, skip_existing=True):
    """
    Scrape disease descriptions for a list of disease IDs
    
    Args:
        disease_ids (set): Set of disease IDs to scrape
        delay (int): Delay between requests in seconds
        max_retries (int): Maximum number of retries for failed requests
        output_file (str): Output file path for real-time saving
        existing_results (dict): Previously scraped results
        existing_failed_ids (list): Previously failed disease IDs
        existing_stats (dict): Previous statistics
        skip_existing (bool): Whether to skip already processed disease IDs
        
    Returns:
        dict: Dictionary with results and statistics
    """
    # Initialize with existing data if provided
    if existing_results is None:
        existing_results = {}
    if existing_failed_ids is None:
        existing_failed_ids = []
    if existing_stats is None:
        existing_stats = {}
    
    # Filter out already processed disease IDs if skip_existing is True
    if skip_existing:
        already_processed = set(existing_results.keys()) | set(existing_failed_ids)
        remaining_ids = disease_ids - already_processed
        skipped_count = len(already_processed)
        
        if skipped_count > 0:
            print(f"Skipping {skipped_count} already processed disease IDs")
            print(f"Remaining to process: {len(remaining_ids)} disease IDs")
        
        disease_ids = remaining_ids
    
    # Initialize results with existing data
    results = existing_results.copy()
    failed_ids = existing_failed_ids.copy()
    
    # Initialize statistics
    stats = {
        'total': len(disease_ids) + len(existing_results) + len(existing_failed_ids),
        'successful': len(existing_results),
        'failed': len(existing_failed_ids),
        'skipped': len(existing_results) + len(existing_failed_ids) if skip_existing else 0,
        'new_processed': 0,
        'start_time': existing_stats.get('start_time', datetime.now().isoformat()),
        'end_time': None,
        'last_update': datetime.now().isoformat()
    }
    
    # Initialize output file if provided
    if output_file:
        initial_data = {
            'results': results,
            'failed_ids': failed_ids,
            'statistics': stats
        }
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(initial_data, f, ensure_ascii=False, indent=2)
        print(f"Updated output file: {output_file}")
    
    # Check if there are any new disease IDs to process
    if not disease_ids:
        print("No new disease IDs to process. All disease IDs have already been processed.")
        stats['end_time'] = datetime.now().isoformat()
        return {
            'results': results,
            'failed_ids': failed_ids,
            'statistics': stats
        }
    
    print(f"Starting to scrape {len(disease_ids)} new disease descriptions...")
    print(f"Delay between requests: {delay} seconds")
    print("-" * 60)
    
    for i, disease_id in enumerate(sorted(disease_ids), 1):
        print(f"[{i}/{len(disease_ids)}] Processing: {disease_id}")
        
        retry_count = 0
        success = False
        
        while retry_count < max_retries and not success:
            try:
                result = get_disease_description(disease_id)
                
                if result['status'] == 'success':
                    results[disease_id] = result
                    stats['successful'] += 1
                    stats['new_processed'] += 1
                    success = True
                    print(f"  ✓ Success: {result.get('disease_name', 'Unknown')}")
                elif result['status'] == 'no_content':
                    # Successfully accessed but no content - this is not a failure
                    results[disease_id] = result
                    stats['successful'] += 1
                    stats['new_processed'] += 1
                    success = True
                    print(f"  ✓ No content: {result.get('disease_name', 'Unknown')} (no description on HPO)")
                else:  # status == 'failed'
                    retry_count += 1
                    if retry_count < max_retries:
                        print(f"  ⚠ Retry {retry_count}/{max_retries}: {result.get('error', 'Unknown error')}")
                        time.sleep(delay * 2)  # Longer delay for retries
                    else:
                        failed_ids.append(disease_id)
                        stats['failed'] += 1
                        stats['new_processed'] += 1
                        print(f"  ✗ Failed: {result.get('error', 'Unknown error')} after {max_retries} attempts")
                        
            except Exception as e:
                retry_count += 1
                if retry_count < max_retries:
                    print(f"  ⚠ Retry {retry_count}/{max_retries}: Error - {e}")
                    time.sleep(delay * 2)
                else:
                    failed_ids.append(disease_id)
                    stats['failed'] += 1
                    stats['new_processed'] += 1
                    print(f"  ✗ Failed: {e}")
        
        # Update timestamp and save immediately if output file is provided
        if output_file:
            stats['last_update'] = datetime.now().isoformat()
            current_data = {
                'results': results,
                'failed_ids': failed_ids,
                'statistics': stats
            }
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(current_data, f, ensure_ascii=False, indent=2)
            print(f"  → Saved to {output_file}")
        
        # Add delay between requests to be respectful to the server
        if i < len(disease_ids):
            time.sleep(delay)
    
    stats['end_time'] = datetime.now().isoformat()
    
    return {
        'results': results,
        'failed_ids': failed_ids,
        'statistics': stats
    }

def save_results(data, output_file):
    """
    Save results to JSON file
    
    Args:
        data (dict): Results data to save
        output_file (str): Output file path
    """
    try:
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        print(f"\nResults saved to: {output_file}")
    except Exception as e:
        print(f"Error saving results: {e}")

def print_summary(data):
    """
    Print summary statistics
    
    Args:
        data (dict): Results data
    """
    stats = data['statistics']
    results = data['results']
    failed_ids = data['failed_ids']
    
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Total disease IDs in dataset: {stats['total']}")
    print(f"Already processed (skipped): {stats.get('skipped', 0)}")
    print(f"Newly processed: {stats.get('new_processed', 0)}")
    print(f"Total successful: {stats['successful']}")
    print(f"Total failed: {stats['failed']}")
    
    if stats['total'] > 0:
        print(f"Overall success rate: {(stats['successful']/stats['total']*100):.1f}%")
    
    if stats.get('new_processed', 0) > 0:
        new_successful = stats['successful'] - stats.get('skipped', 0)
        print(f"New processing success rate: {(new_successful/stats['new_processed']*100):.1f}%")
    
    print(f"Start time: {stats['start_time']}")
    print(f"End time: {stats['end_time']}")
    
    # Show first 10 disease IDs (all processed IDs, regardless of success/failure)
    all_processed_ids = set(results.keys()) | set(failed_ids)
    if all_processed_ids:
        print(f"\nFirst 10 processed disease IDs (alphabetical order):")
        for i, disease_id in enumerate(sorted(all_processed_ids)[:10], 1):
            status = "✓" if disease_id in results else "✗"
            disease_name = results.get(disease_id, {}).get('disease_name', 'Unknown') if disease_id in results else 'Failed'
            print(f"  {i:2d}. {status} {disease_id}: {disease_name}")
        if len(all_processed_ids) > 10:
            print(f"  ... and {len(all_processed_ids) - 10} more")
    
    # Show failed IDs summary
    if failed_ids:
        print(f"\nFailed disease IDs ({len(failed_ids)}):")
        for disease_id in sorted(failed_ids):
            print(f"  - {disease_id}")

def main():
    """Main function"""
    parser = argparse.ArgumentParser(
        description='Batch scrape disease descriptions from HPO website',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage - will skip already processed diseases
  python batch_disease_scraper.py --input phenotype_disease_case_library.jsonl
  
  # Reprocess all diseases (ignore existing results)
  python batch_disease_scraper.py --input data.jsonl --output results.json --no-skip-existing
  
  # Process with custom delay and retry settings
  python batch_disease_scraper.py --input data.jsonl --delay 2 --max-retries 5 --verbose
  
  # Test with limited number of diseases
  python batch_disease_scraper.py --input data.jsonl --limit 10
        """
    )
    
    parser.add_argument('--input', '-i',
                       default='phenotype_disease_case_library.jsonl',
                       help='Input JSONL file path (default: phenotype_disease_case_library.jsonl)')
    
    parser.add_argument('--output', '-o',
                       default='disease_descriptions_batch.json',
                       help='Output JSON file path (default: disease_descriptions_batch.json)')
    
    parser.add_argument('--delay', '-d',
                       type=int,
                       default=1,
                       help='Delay between requests in seconds (default: 1)')
    
    parser.add_argument('--max-retries', '-r',
                       type=int,
                       default=3,
                       help='Maximum number of retries for failed requests (default: 3)')
    
    parser.add_argument('--limit', '-l',
                       type=int,
                       help='Limit number of disease IDs to process (for testing)')
    
    parser.add_argument('--verbose', '-v',
                       action='store_true',
                       help='Verbose output')
    
    parser.add_argument('--no-skip-existing',
                       action='store_true',
                       help='Do not skip already processed disease IDs (reprocess all)')
    
    args = parser.parse_args()
    
    # Check if input file exists
    if not os.path.exists(args.input):
        print(f"Error: Input file not found: {args.input}")
        sys.exit(1)
    
    print("Batch Disease Description Scraper")
    print("=" * 40)
    print(f"Input file: {args.input}")
    print(f"Output file: {args.output}")
    print(f"Delay: {args.delay} seconds")
    print(f"Max retries: {args.max_retries}")
    print(f"Skip existing: {not args.no_skip_existing}")
    if args.limit:
        print(f"Limit: {args.limit} disease IDs")
    print()
    
    # Load disease IDs
    print("Loading disease IDs from input file...")
    all_disease_ids = load_disease_ids_from_jsonl(args.input)
    
    if not all_disease_ids:
        print("No disease IDs found in input file")
        sys.exit(1)
    
    print(f"Found {len(all_disease_ids)} unique disease IDs")
    
    # Load existing results if output file exists and we're not reprocessing all
    existing_results, existing_failed_ids, existing_stats = {}, [], {}
    if not args.no_skip_existing:
        existing_results, existing_failed_ids, existing_stats = load_existing_results(args.output)
    
    # Calculate how many will actually be processed
    if args.limit:
        # When limit is specified, get the first N IDs from the entire dataset
        limited_ids = set(sorted(all_disease_ids)[:args.limit])
        already_processed = set(existing_results.keys()) | set(existing_failed_ids)
        remaining_in_limited = limited_ids - already_processed
        
        print(f"Limited to {len(limited_ids)} disease IDs (first {args.limit} from entire dataset)")
        print(f"Already processed: {len(limited_ids) - len(remaining_in_limited)}")
        print(f"Remaining to process: {len(remaining_in_limited)}")
        
        if len(remaining_in_limited) == 0:
            print(f"All {len(limited_ids)} limited disease IDs have already been processed.")
            print("No new processing needed.")
            sys.exit(0)
        
        disease_ids = remaining_in_limited
        to_process = len(remaining_in_limited)
        skip_existing = False  # Don't skip in scrape_disease_descriptions since we already filtered
    else:
        skip_existing = not args.no_skip_existing
        if skip_existing:
            already_processed = set(existing_results.keys()) | set(existing_failed_ids)
            remaining_ids = all_disease_ids - already_processed
            to_process = len(remaining_ids)
            disease_ids = remaining_ids
        else:
            disease_ids = all_disease_ids
            to_process = len(disease_ids)
    
    # Confirm before starting
    if not args.verbose and to_process > 0:
        response = input(f"\nProceed with scraping {to_process} disease descriptions? (y/N): ")
        if response.lower() != 'y':
            print("Operation cancelled")
            sys.exit(0)
    
    # Scrape descriptions with real-time saving
    data = scrape_disease_descriptions(
        disease_ids, 
        args.delay, 
        args.max_retries, 
        args.output,
        existing_results,
        existing_failed_ids,
        existing_stats,
        skip_existing
    )
    
    # Final save (in case real-time saving was not used)
    if not args.output:
        save_results(data, 'disease_descriptions_batch.json')
    
    # Print summary
    print_summary(data)

if __name__ == "__main__":
    main()