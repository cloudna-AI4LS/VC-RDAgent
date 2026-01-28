#!/usr/bin/env python3
"""
Orphanet XML to JSON Converter for Product1 (Disorder List)
Converts the Orphanet disorder list XML file to JSON format
"""

import xml.etree.ElementTree as ET
import json
import sys
from typing import Dict, List, Any
import argparse
from pathlib import Path

def parse_disorder_flag(flag_elem) -> Dict[str, Any]:
    """Parse disorder flag information"""
    return {
        'id': flag_elem.get('id'),
        'value': flag_elem.find('Value').text if flag_elem.find('Value') is not None else None,
        'label': flag_elem.find('Label').text if flag_elem.find('Label') is not None else None
    }

def parse_synonym(synonym_elem) -> Dict[str, str]:
    """Parse synonym information"""
    return {
        'text': synonym_elem.text,
        'lang': synonym_elem.get('lang')
    }

def parse_external_reference(ref_elem) -> Dict[str, Any]:
    """Parse external reference information"""
    ref = {
        'id': ref_elem.get('id'),
        'source': ref_elem.find('Source').text if ref_elem.find('Source') is not None else None,
        'reference': ref_elem.find('Reference').text if ref_elem.find('Reference') is not None else None
    }
    
    # Parse mapping relation
    mapping_relation_elem = ref_elem.find('DisorderMappingRelation')
    if mapping_relation_elem is not None:
        name_elem = mapping_relation_elem.find('Name[@lang="en"]')
        ref['mapping_relation'] = {
            'id': mapping_relation_elem.get('id'),
            'name': name_elem.text if name_elem is not None else None
        }
    
    # Parse ICD relation
    icd_relation_elem = ref_elem.find('DisorderMappingICDRelation')
    if icd_relation_elem is not None:
        name_elem = icd_relation_elem.find('Name[@lang="en"]')
        ref['icd_relation'] = {
            'id': icd_relation_elem.get('id'),
            'name': name_elem.text if name_elem is not None else None
        }
    
    # Parse validation status
    validation_elem = ref_elem.find('DisorderMappingValidationStatus')
    if validation_elem is not None:
        name_elem = validation_elem.find('Name[@lang="en"]')
        ref['validation_status'] = {
            'id': validation_elem.get('id'),
            'name': name_elem.text if name_elem is not None else None
        }
    
    # Parse URL and URI
    url_elem = ref_elem.find('DisorderMappingICDRefUrl')
    ref['url'] = url_elem.text if url_elem is not None else None
    
    uri_elem = ref_elem.find('DisorderMappingICDRefUri')
    ref['uri'] = uri_elem.text if uri_elem is not None else None
    
    return ref

def parse_disorder(disorder_elem) -> Dict[str, Any]:
    """Parse disorder information"""
    disorder = {
        'id': disorder_elem.get('id')
    }
    
    # Parse OrphaCode
    orpha_code_elem = disorder_elem.find('OrphaCode')
    disorder['orpha_code'] = orpha_code_elem.text if orpha_code_elem is not None else None
    
    # Parse ExpertLink
    expert_link_elem = disorder_elem.find('ExpertLink[@lang="en"]')
    disorder['expert_link'] = expert_link_elem.text if expert_link_elem is not None else None
    
    # Parse Name
    name_elem = disorder_elem.find('Name[@lang="en"]')
    disorder['name'] = name_elem.text if name_elem is not None else None
    
    # Parse DisorderFlags
    flags_elem = disorder_elem.find('DisorderFlagList')
    disorder['flags'] = []
    if flags_elem is not None:
        flag_count = int(flags_elem.get('count', 0))
        disorder['flag_count'] = flag_count
        for flag_elem in flags_elem.findall('DisorderFlag'):
            flag = parse_disorder_flag(flag_elem)
            disorder['flags'].append(flag)
    
    # Parse Synonyms
    synonyms_elem = disorder_elem.find('SynonymList')
    disorder['synonyms'] = []
    if synonyms_elem is not None:
        synonym_count = int(synonyms_elem.get('count', 0))
        disorder['synonym_count'] = synonym_count
        for synonym_elem in synonyms_elem.findall('Synonym'):
            synonym = parse_synonym(synonym_elem)
            disorder['synonyms'].append(synonym)
    
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
    
    # Parse External References
    external_refs_elem = disorder_elem.find('ExternalReferenceList')
    disorder['external_references'] = []
    if external_refs_elem is not None:
        ref_count = int(external_refs_elem.get('count', 0))
        disorder['external_reference_count'] = ref_count
        for ref_elem in external_refs_elem.findall('ExternalReference'):
            ref = parse_external_reference(ref_elem)
            disorder['external_references'].append(ref)
    
    return disorder

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
            'disorder_list': {
                'count': 0,
                'disorders': []
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
        
        # Parse Disorder List
        disorder_list_elem = root.find('DisorderList')
        if disorder_list_elem is not None:
            result['disorder_list']['count'] = int(disorder_list_elem.get('count', 0))
            
            print(f"Processing {result['disorder_list']['count']} disorder entries...")
            
            for i, disorder_elem in enumerate(disorder_list_elem.findall('Disorder')):
                if i % 500 == 0:
                    print(f"Processed {i} entries...")
                
                disorder = parse_disorder(disorder_elem)
                result['disorder_list']['disorders'].append(disorder)
        
        print(f"Writing JSON file: {output_file_path}")
        
        # Write JSON file
        with open(output_file_path, 'w', encoding='utf-8') as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
        
        print(f"Conversion completed successfully!")
        print(f"Total entries processed: {len(result['disorder_list']['disorders'])}")
        
    except ET.ParseError as e:
        print(f"XML parsing error: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"Error during conversion: {e}")
        sys.exit(1)

def main():
    parser = argparse.ArgumentParser(description='Convert Orphanet Product1 XML to JSON')
    parser.add_argument('--input', '-i', 
                       default='en_product1.xml',
                       help='Input XML file path')
    parser.add_argument('--output', '-o',
                       default='en_product1.json',
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

