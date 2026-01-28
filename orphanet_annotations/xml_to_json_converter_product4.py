#!/usr/bin/env python3
"""
Orphanet XML to JSON Converter
Converts the Orphanet HPO disorder annotations XML file to JSON format
"""

import xml.etree.ElementTree as ET
import json
import sys
from typing import Dict, List, Any
import argparse
from pathlib import Path

def parse_hpo_frequency(frequency_elem) -> Dict[str, str]:
    """Parse HPO frequency information"""
    if frequency_elem is not None:
        name_elem = frequency_elem.find('Name[@lang="en"]')
        if name_elem is not None:
            return {"name": name_elem.text, "lang": "en"}
    return {}

def parse_hpo_association(association_elem) -> Dict[str, Any]:
    """Parse a single HPO disorder association"""
    association = {}
    
    # Parse HPO information
    hpo_elem = association_elem.find('HPO')
    if hpo_elem is not None:
        hpo_id_elem = hpo_elem.find('HPOId')
        hpo_term_elem = hpo_elem.find('HPOTerm')
        
        association['hpo'] = {
            'id': hpo_id_elem.text if hpo_id_elem is not None else None,
            'term': hpo_term_elem.text if hpo_term_elem is not None else None
        }
    
    # Parse frequency information
    frequency_elem = association_elem.find('HPOFrequency')
    association['frequency'] = parse_hpo_frequency(frequency_elem)
    
    # Parse diagnostic criteria
    diagnostic_elem = association_elem.find('DiagnosticCriteria')
    association['diagnostic_criteria'] = diagnostic_elem.text if diagnostic_elem is not None and diagnostic_elem.text else None
    
    return association

def parse_disorder(disorder_elem) -> Dict[str, Any]:
    """Parse disorder information"""
    disorder = {}
    
    # Parse OrphaCode
    orpha_code_elem = disorder_elem.find('OrphaCode')
    disorder['orpha_code'] = orpha_code_elem.text if orpha_code_elem is not None else None
    
    # Parse ExpertLink
    expert_link_elem = disorder_elem.find('ExpertLink[@lang="en"]')
    disorder['expert_link'] = expert_link_elem.text if expert_link_elem is not None else None
    
    # Parse Name
    name_elem = disorder_elem.find('Name[@lang="en"]')
    disorder['name'] = name_elem.text if name_elem is not None else None
    
    # Parse DisorderType
    disorder_type_elem = disorder_elem.find('DisorderType')
    if disorder_type_elem is not None:
        type_name_elem = disorder_type_elem.find('Name[@lang="en"]')
        disorder['disorder_type'] = {
            'id': disorder_type_elem.get('id'),
            'name': type_name_elem.text if type_name_elem is not None else None
        }
    
    # Parse DisorderGroup
    disorder_group_elem = disorder_elem.find('DisorderGroup')
    if disorder_group_elem is not None:
        group_name_elem = disorder_group_elem.find('Name[@lang="en"]')
        disorder['disorder_group'] = {
            'id': disorder_group_elem.get('id'),
            'name': group_name_elem.text if group_name_elem is not None else None
        }
    
    # Parse HPO Disorder Associations
    hpo_associations_elem = disorder_elem.find('HPODisorderAssociationList')
    disorder['hpo_associations'] = []
    
    if hpo_associations_elem is not None:
        association_count = int(hpo_associations_elem.get('count', 0))
        disorder['hpo_association_count'] = association_count
        
        for association_elem in hpo_associations_elem.findall('HPODisorderAssociation'):
            association = parse_hpo_association(association_elem)
            disorder['hpo_associations'].append(association)
    
    return disorder

def parse_hpo_disorder_set_status(status_elem) -> Dict[str, Any]:
    """Parse a single HPO disorder set status entry"""
    status = {}
    
    # Parse Disorder
    disorder_elem = status_elem.find('Disorder')
    if disorder_elem is not None:
        status['disorder'] = parse_disorder(disorder_elem)
    
    # Parse other fields
    source_elem = status_elem.find('Source')
    status['source'] = source_elem.text if source_elem is not None else None
    
    validation_status_elem = status_elem.find('ValidationStatus')
    status['validation_status'] = validation_status_elem.text if validation_status_elem is not None else None
    
    online_elem = status_elem.find('Online')
    status['online'] = online_elem.text if online_elem is not None else None
    
    validation_date_elem = status_elem.find('ValidationDate')
    status['validation_date'] = validation_date_elem.text if validation_date_elem is not None else None
    
    return status

def convert_xml_to_json(xml_file_path: str, output_file_path: str) -> None:
    """Convert Orphanet XML file to JSON format"""
    print(f"Reading XML file: {xml_file_path}")
    
    try:
        # Parse XML file
        tree = ET.parse(xml_file_path)
        root = tree.getroot()
        
        # Initialize result structure
        result = {
            'metadata': {
                'date': root.get('date'),
                'version': root.get('version'),
                'copyright': root.get('copyright'),
                'dbserver': root.get('dbserver')
            },
            'availability': {},
            'hpo_disorder_set_status_list': {
                'count': 0,
                'entries': []
            }
        }
        
        # Parse Availability
        availability_elem = root.find('Availability')
        if availability_elem is not None:
            licence_elem = availability_elem.find('Licence')
            if licence_elem is not None:
                full_name_elem = licence_elem.find('FullName[@lang="en"]')
                short_id_elem = licence_elem.find('ShortIdentifier')
                legal_code_elem = licence_elem.find('LegalCode')
                
                result['availability']['licence'] = {
                    'full_name': full_name_elem.text if full_name_elem is not None else None,
                    'short_identifier': short_id_elem.text if short_id_elem is not None else None,
                    'legal_code': legal_code_elem.text if legal_code_elem is not None else None
                }
        
        # Parse HPO Disorder Set Status List
        hpo_list_elem = root.find('HPODisorderSetStatusList')
        if hpo_list_elem is not None:
            result['hpo_disorder_set_status_list']['count'] = int(hpo_list_elem.get('count', 0))
            
            print(f"Processing {result['hpo_disorder_set_status_list']['count']} disorder entries...")
            
            for i, status_elem in enumerate(hpo_list_elem.findall('HPODisorderSetStatus')):
                if i % 100 == 0:
                    print(f"Processed {i} entries...")
                
                status = parse_hpo_disorder_set_status(status_elem)
                result['hpo_disorder_set_status_list']['entries'].append(status)
        
        print(f"Writing JSON file: {output_file_path}")
        
        # Write JSON file
        with open(output_file_path, 'w', encoding='utf-8') as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
        
        print(f"Conversion completed successfully!")
        print(f"Total entries processed: {len(result['hpo_disorder_set_status_list']['entries'])}")
        
    except ET.ParseError as e:
        print(f"XML parsing error: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"Error during conversion: {e}")
        sys.exit(1)

def main():
    parser = argparse.ArgumentParser(description='Convert Orphanet XML to JSON')
    parser.add_argument('--input', '-i', 
                       default='en_product4.xml',
                       help='Input XML file path')
    parser.add_argument('--output', '-o',
                       default='en_product4.json',
                       help='Output JSON file path')
    
    args = parser.parse_args()
    
    # Check if input file exists
    if not Path(args.input).exists():
        print(f"Error: Input file {args.input} does not exist")
        sys.exit(1)
    
    # Create output directory if it doesn't exist
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    convert_xml_to_json(args.input, args.output)

if __name__ == "__main__":
    main()
