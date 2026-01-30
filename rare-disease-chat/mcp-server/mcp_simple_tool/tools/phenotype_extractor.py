#!/usr/bin/env python3
"""
MCP server for phenotype extraction functionality
"""
import os
import re
import json
import sys
from typing import Dict, List, Optional, Tuple, Any
from mcp import types
from FastHPOCR.HPOAnnotator import HPOAnnotator
from FastHPOCR.IndexHPO import IndexHPO
import contextlib
import io

# Add parent directory to path for importing other modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import vc_ranker.py
from scripts.rare_disease_diagnose.vc_ranker import EnsembleDiseaseRanker

# Base directory for resolving relative paths (based on file location)
BASE_DIR = os.path.dirname(__file__)

# Global variables for HPO data
hpo_annotator = None
hpo_data = {}
obo_file_path = None

# Global phenotype analyzer instance
_phenotype_analyzer = None
_initialization_failed = False

def _get_phenotype_analyzer(config_file: str = None):
    """Get phenotype analyzer instance (singleton pattern)"""
    global _phenotype_analyzer
    if _phenotype_analyzer is None:
        try:
            if config_file is None:
                config_file = os.path.join(BASE_DIR, '../scripts/rare_disease_diagnose/prompt_config_forKG.json')
            print("Initializing phenotype analyzer...")
            with contextlib.redirect_stdout(io.StringIO()):
                _phenotype_analyzer = EnsembleDiseaseRanker(config_file)
            print("Phenotype analyzer initialized successfully")
        except Exception as e:
            print(f"Failed to initialize phenotype analyzer: {e}")
            _initialization_failed = True
            return None
    return _phenotype_analyzer

def _initialize_annotator(index_path: Optional[str] = None):
    """Initialize the HPO annotator with the given index path."""
    global hpo_annotator
    try:
        # If already initialized, return
        if hpo_annotator is not None:
            return None
        
        if index_path is None or not os.path.exists(index_path):
            # Default paths (based on file location, not working directory)
            hpo_obo_path = os.path.join(BASE_DIR, '../data/hpo_annotations/hp.obo')
            index_output_path = os.path.join(BASE_DIR, '../data/hpo_annotations/')
            index_file_path = os.path.join(index_output_path, "hp.index")
            
            # Check if index file already exists
            if os.path.exists(index_file_path) or os.path.exists(f"{index_file_path}.gz"):
                # Prefer compressed version
                gz_index_file_path = f"{index_file_path}.gz"
                if os.path.exists(gz_index_file_path):
                    index_path = gz_index_file_path
                else:
                    index_path = index_file_path
            else:
                # If index file doesn't exist, create it first
                if not os.path.exists(hpo_obo_path):
                    return f"Error: HPO OBO file not found at {hpo_obo_path}. Please download hp.obo first."
                
                # Create index
                try:
                    
                    index_config = {
                        'rootConcepts': [
                            'HP:0000119', 'HP:0000152', 'HP:0000478', 'HP:0000598',
                            'HP:0000707', 'HP:0000769', 'HP:0000818', 'HP:0001197',
                            'HP:0001507', 'HP:0001574', 'HP:0001608', 'HP:0001626',
                            'HP:0001871', 'HP:0001939', 'HP:0002086', 'HP:0002664',
                            'HP:0002715', 'HP:0025031', 'HP:0025142', 'HP:0025354',
                            'HP:0033127', 'HP:0040064', 'HP:0045027'
                        ],
                        'allow3LetterAcronyms': True,
                        'includeTopLevelCategory': True,
                        'allowDuplicateEntries': True,
                        'compressIndex': True
                    }
                    
                    indexHPO = IndexHPO(hpo_obo_path, index_output_path, indexConfig=index_config)
                    indexHPO.index()
                    
                    # After creation, prefer compressed version
                    gz_index_file_path = f"{index_file_path}.gz"
                    if os.path.exists(gz_index_file_path):
                        index_path = gz_index_file_path
                    else:
                        index_path = index_file_path
                        
                except ImportError:
                    return "Error: FastHPOCR library not installed. Please install it using: pip install FastHPOCR"
        
        hpo_annotator = HPOAnnotator(index_path)
        return None  # Success
        
    except ImportError:
        return "Error: FastHPOCR library not installed. Please install it using: pip install FastHPOCR"
    except Exception as e:
        return f"Error initializing HPO annotator: {str(e)}"

def _load_hpo_data(obo_file_path: str):
    """Load HPO data from OBO file"""
    global hpo_data
    
    if hpo_data and obo_file_path == obo_file_path:
        return None  # Already loaded
    
    hpo_data = {}
    
    try:
        with open(obo_file_path, 'r', encoding='utf-8') as f:
            current_term = {}
            for line in f:
                line = line.strip()
                
                if line == '[Term]':
                    # Save previous term if we have one
                    if current_term and 'id' in current_term:
                        _process_term(current_term)
                    current_term = {}
                    
                elif line.startswith('id: HP:'):
                    current_term['id'] = line.split(': ')[1]
                    
                elif line.startswith('name: '):
                    current_term['name'] = line.split(': ', 1)[1]
                    
                elif line.startswith('is_a: '):
                    if 'is_a' not in current_term:
                        current_term['is_a'] = []
                    # Parse is_a line: "is_a: HP:0001252 ! Hypotonia"
                    is_a_content = line.split(': ', 1)[1]
                    if ' ! ' in is_a_content:
                        parent_id = is_a_content.split(' ! ', 1)[0]
                        current_term['is_a'].append(parent_id)
                    else:
                        current_term['is_a'].append(is_a_content)
                        
                elif line.startswith('synonym: '):
                    if 'synonyms' not in current_term:
                        current_term['synonyms'] = []
                    synonym_content = line.split(': ', 1)[1]
                    if '"' in synonym_content:
                        start_quote = synonym_content.find('"')
                        end_quote = synonym_content.rfind('"')
                        if start_quote != -1 and end_quote != -1 and start_quote < end_quote:
                            current_term['synonyms'].append(synonym_content[start_quote + 1:end_quote])
                    else:
                        current_term['synonyms'].append(synonym_content)
            
            # Don't forget the last term
            if current_term and 'id' in current_term:
                _process_term(current_term)
                
    except Exception as e:
        return f"Error loading HPO OBO file: {str(e)}"
    
    return None

def _process_term(term: dict):
    """Process a single HPO term and store in hpo_data"""
    global hpo_data
    hpo_id = term['id']
    hpo_data[hpo_id] = {
        'id': hpo_id,
        'name': term.get('name', ''),
        'is_a': term.get('is_a', []),
        'synonyms': term.get('synonyms', [])
    }

def _extract_hpo_ids_from_text(text: str) -> List[Tuple[int, int, str]]:
    """Extract HPO IDs from text and return (start, end, hpo_id) tuples"""
    # Pattern to match HP: followed by 7 digits
    pattern = r'HP:\d{7}'
    matches = []
    
    for match in re.finditer(pattern, text):
        start = match.start()
        end = match.end()
        hpo_id = match.group()
        matches.append((start, end, hpo_id))
    
    return matches

def _get_parent_info(hpo_id: str) -> str:
    """Get top-level parent information for an HPO ID"""
    global hpo_data
    
    if hpo_id not in hpo_data:
        return ""
    
    # Define top-level HPO categories
    top_level_categories = {
        'HP:0000119', 'HP:0000152', 'HP:0000478', 'HP:0000598',
        'HP:0000707', 'HP:0000769', 'HP:0000818', 'HP:0001197',
        'HP:0001507', 'HP:0001574', 'HP:0001608', 'HP:0001626',
        'HP:0001871', 'HP:0001939', 'HP:0002086', 'HP:0002664',
        'HP:0002715', 'HP:0025031', 'HP:0025142', 'HP:0025354',
        'HP:0033127', 'HP:0040064', 'HP:0045027'
    }
    
    # Find top-level parents by traversing up the hierarchy
    top_level_parents = set()
    visited = set()
    
    def find_top_level_parents(current_id):
        if current_id in visited:
            return
        visited.add(current_id)
        
        if current_id in top_level_categories:
            top_level_parents.add(current_id)
            return
        
        if current_id in hpo_data:
            parents = hpo_data[current_id].get('is_a', [])
            for parent_id in parents:
                find_top_level_parents(parent_id)
    
    find_top_level_parents(hpo_id)
    
    if not top_level_parents:
        return ""
    
    # Get parent names
    parent_names = []
    for parent_id in top_level_parents:
        if parent_id in hpo_data:
            parent_name = hpo_data[parent_id]['name']
            parent_names.append(f"{parent_id} ({parent_name})")
        else:
            parent_names.append(parent_id)
    
    return " | ".join(parent_names)

def _generate_phenotype_details_structured(hpo_ids: List[str]) -> Dict:
    """Generate structured phenotype details for given HPO IDs"""
    try:
        phenotype_analyzer = _get_phenotype_analyzer()
        if not phenotype_analyzer:
            return {}
        
        phenotype_details_structured = {}
        
        for hpo_id in hpo_ids:
            phenotype_info = phenotype_analyzer.generator.get_phenotype_info_from_kg(hpo_id)
            phenotype_name = phenotype_info.get("standard_name", "")
            # phenotype_name = phenotype_analyzer.generator.get_phenotype_name(hpo_id)
            if phenotype_name:
                synonyms = phenotype_info.get("synonyms", [])
                definition = phenotype_info.get("description", "")
                is_a_names = phenotype_analyzer.generator.get_phenotype_ancestors_from_kg(hpo_id, max_depth=1)
                disease_count = phenotype_info.get("associations", 0)
                # synonyms = phenotype_analyzer.generator.get_phenotype_synonyms(hpo_id)
                # definition = phenotype_analyzer.generator.get_phenotype_definition(hpo_id)
                # is_a_names = phenotype_analyzer.generator.get_phenotype_is_a_names(hpo_id)
                # disease_count = len(phenotype_analyzer.generator.phenotype_to_diseases.get(hpo_id, set()))
                if disease_count == 0:
                    disease_count = 1
                # phenotype_abnormal_category_id = phenotype_analyzer.generator.get_phenotype_abnormal_category(hpo_id)
                phenotype_abnormal_category_id = phenotype_analyzer.generator.get_phenotype_abnormal_category_from_kg(hpo_id)
                if isinstance(phenotype_abnormal_category_id, list):
                    phenotype_abnormal_categorys = []
                    for category_id in phenotype_abnormal_category_id:
                        # phenotype_abnormal_categorys.append(phenotype_analyzer.generator.get_phenotype_name(category_id))
                        phenotype_abnormal_categorys.append(phenotype_analyzer.generator.get_phenotype_info_from_kg(category_id).get("standard_name", ""))
                    phenotype_abnormal_category = " | ".join(phenotype_abnormal_categorys)
                else:
                    # phenotype_abnormal_category = phenotype_analyzer.generator.get_phenotype_name(phenotype_abnormal_category_id) if phenotype_abnormal_category_id else "Unknown category"
                    phenotype_abnormal_category = phenotype_analyzer.generator.get_phenotype_info_from_kg(phenotype_abnormal_category_id).get("standard_name", "") if phenotype_abnormal_category_id else "Unknown category"
                
                phenotype_details_structured[phenotype_name] = {
                    "hpo_id": hpo_id,
                    "phenotype abnormal category": phenotype_abnormal_category,
                    "synonyms": synonyms if synonyms else [],
                    "parent categories": is_a_names if is_a_names else [],
                    # "number of associated diseases": disease_count,
                    "detailed information": definition if definition else ""
                }
        
        return phenotype_details_structured
    except Exception as e:
        print(f"Error generating phenotype details: {e}")
        return {}

def phenotype_extractor(query: str, index_path: Optional[str] = None, include_categories: bool = True, only_init: bool = False) -> str:
    """Synchronous function: execute phenotype extraction, return JSON string (for other modules to call)"""
    global hpo_annotator, hpo_data
    
    try:
        if index_path is None:
            index_path = os.path.join(BASE_DIR, '../data/hpo_annotations/hp.index')

        # Initialize annotator if not already done
        if hpo_annotator is None:
            init_result = _initialize_annotator(index_path)
            if init_result is not None:  # If initialization failed, return the error
                return json.dumps({
                    "success": False,
                    "error": init_result
                }, ensure_ascii=False, indent=2)

        if only_init:
            phenotype_analyzer = _get_phenotype_analyzer()
            result = {
                "init": True,
            }
            return json.dumps(result, ensure_ascii=False, indent=2)

        # Check if input is provided
        if not query.strip():
            return json.dumps({
                "success": False,
                "error": "No text provided for phenotype extraction."
            }, ensure_ascii=False, indent=2)
        
        # First, extract HPO IDs from text and get their information
        hpo_obo_path = os.path.join(BASE_DIR, '../data/hpo_annotations/hp.obo')
        hpo_ids = []
        # Load HPO data if not already loaded
        if not hpo_data:
            error = _load_hpo_data(hpo_obo_path)
            if error:
                return json.dumps({
                    "success": False,
                    "error": error
                }, ensure_ascii=False, indent=2)
        
        # Extract HPO IDs from text
        hpo_matches = _extract_hpo_ids_from_text(query)
        
        # Process HPO ID matches
        for start, end, hpo_id in hpo_matches:
            hpo_ids.append(hpo_id)
        
        # Then, use hpo_annotator to extract phenotypes from text
        annotations = hpo_annotator.annotate(query)
        
        # Extract HPO IDs from annotations
        lines = []
        for annotationObject in annotations:
            if include_categories:
                lines.append(annotationObject.toStringWithCategories())
            else:
                lines.append(annotationObject.toString())
        
        # Extract IDs from lines with format [start:end]	HP:ID	term	description
        extracted_ids = []
        for line in lines:
            # Pattern to match [start:end]	HP:ID	term	description
            # Split by tab and get the second element (index 1) which should be the HP:ID
            parts = line.split('\t')
            if len(parts) >= 2 and parts[1].startswith('HP:'):
                extracted_ids.append(parts[1])
        
        # Add extracted IDs to hpo_ids list
        hpo_ids.extend(extracted_ids)
        
        # Remove duplicates
        hpo_ids = list(set(hpo_ids))
        # Sort hpo_ids
        hpo_ids.sort()

        if not hpo_ids:
            return json.dumps({
                "success": False,
                "error": "No valid phenotypes or HPO IDs detected"
            }, ensure_ascii=False, indent=2)
        
        # Generate structured phenotype details
        phenotype_details_structured = _generate_phenotype_details_structured(hpo_ids)
        # Extract all HPO IDs from phenotype_details_structured, sort and join as string as key
        hpo_ids_str = []
        for phenotype_name, details in phenotype_details_structured.items():
            hpo_id = details.get("hpo_id")
            if hpo_id:
                hpo_ids_str.append(hpo_id)
        hpo_ids_str.sort()
        hpo_ids_str_str = ",".join(hpo_ids_str)
        
        result = {
            "extracted_phenotypes": {
                hpo_ids_str_str: phenotype_details_structured
            }
        }
        
        return json.dumps(result, ensure_ascii=False, indent=2)

    except Exception as e:
        return json.dumps({
            "success": False,
            "error": f"Error during phenotype extraction: {str(e)}"
        }, ensure_ascii=False, indent=2)

def get_tool_definition() -> types.Tool:
    """Return tool definition"""
    return types.Tool(
        name="phenotype-extractor",
        description="Support extracting phenotype, symptom, HPO IDs (e.g., HP:0000123, HP:0000124) from user query and mapping them to specific phenotypes and phenotype descriptions.",
        inputSchema={
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Text query containing phenotypes, symptoms, or HPO IDs to extract"
                },
                "index_path": {
                    "type": "string",
                    "description": "Optional path to HPO index file. If not provided, uses default path."
                },
                "include_categories": {
                    "type": "boolean",
                    "description": "Whether to include categories in the annotation output. Default is true.",
                    "default": True
                },
                "only_init": {
                    "type": "boolean",
                    "description": "If true, only initialize the phenotype analyzer without processing the query. Default is false.",
                    "default": False
                }
            },
            "required": ["query"]
        }
    )


async def call_tool(context, arguments: Dict[str, Any]) -> List[types.ContentBlock]:
    """Execute phenotype extraction tool"""
    try:
        query = arguments.get("query", "")
        index_path = arguments.get("index_path")
        include_categories = arguments.get("include_categories", True)
        only_init = arguments.get("only_init", False)
        
        result = phenotype_extractor(query, index_path, include_categories, only_init)
        
        return [types.TextContent(
            type="text",
            text=result
        )]
        
    except Exception as e:
        error_msg = {
            "success": False,
            "error": f"Error during phenotype extraction: {str(e)}"
        }
        return [types.TextContent(
            type="text",
            text=json.dumps(error_msg, ensure_ascii=False, indent=2)
        )]
