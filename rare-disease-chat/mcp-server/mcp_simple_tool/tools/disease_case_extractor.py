#!/usr/bin/env python3
"""
MCP server for disease case extraction functionality
Integrates vc_ranker.py and phenotype_extractor_tool functionality
Extracts phenotype information from user input and returns formatted disease case information
"""

import os
import json
import sys
from typing import Dict, List, Optional, Any
from mcp import types

# Redirect stdout to suppress print statements inside process_sample
import contextlib
import io


# Add parent directory to path for importing other modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import related modules
from tools.phenotype_extractor import phenotype_extractor, _get_phenotype_analyzer

# Base directory for resolving relative paths (based on file location)
BASE_DIR = os.path.dirname(__file__)

# Global variables
_disease_case_extractor = None
_initialization_failed = False


def _get_disease_case_extractor(config_file: str = None):
    """Get disease case extractor instance (singleton pattern)"""
    global _disease_case_extractor
    if _disease_case_extractor is None:
        try:
            if config_file is None:
                config_file = os.path.join(BASE_DIR, '../scripts/rare_disease_diagnose/prompt_config_forKG.json')
            # print("Initializing disease case extractor...")
            _disease_case_extractor = DiseaseCaseExtractor(config_file)
            # print("Disease case extractor initialized successfully")
        except Exception as e:
            print(f"Failed to initialize disease case extractor: {e}")
            _initialization_failed = True
            return None
    return _disease_case_extractor


class DiseaseCaseExtractor:
    """Disease case extraction tool class"""
    
    def __init__(self, config_file: str = None):
        """Initialize disease case extractor"""
        # print("Initializing disease case extractor...")
        
        if config_file is None:
            config_file = os.path.join(BASE_DIR, '../scripts/rare_disease_diagnose/prompt_config_forKG.json')
        
        # Use already initialized phenotype analyzer instance to avoid duplicate initialization
        self.ranker = _get_phenotype_analyzer(config_file)
        if not self.ranker:
            raise Exception("Failed to get phenotype analyzer instance")
        
        # print("Disease case extractor initialization completed")
    
    def extract_phenotypes_from_text(self, text: str) -> List[str]:
        """Extract phenotype information from text"""
        try:
            # Call phenotype_extractor tool
            result = phenotype_extractor(text, include_categories=True)
            result_data = json.loads(result)
            
            if "success" in result_data and not result_data["success"]:
                print(f"Phenotype extraction failed: {result_data.get('error', 'Unknown error')}")
                return []
            
            # Extract HPO IDs from results
            extracted_phenotypes = result_data.get("extracted_phenotypes", {})
            hpo_ids = []
            
            for phenotype_name, details in extracted_phenotypes.items():
                hpo_id = details.get("hpo_id")
                if hpo_id:
                    hpo_ids.append(hpo_id)
            
            # print(f"Extracted {len(hpo_ids)} phenotypes from text")
            return hpo_ids, extracted_phenotypes
            
        except Exception as e:
            print(f"Error during phenotype extraction: {e}")
            return []
    
    def generate_disease_cases(self, phenotypes: List[str], top_k: int = 100, final_top_k: int = 50) -> Dict[str, Any]:
        """Generate disease case information"""
        try:
            # print(f"Generating disease cases, input phenotypes: {len(phenotypes)}")
            if len(phenotypes) == 0:
                return {
                    "success": False,
                    "error": "No valid phenotypes extracted from input text"
                }
            elif len(phenotypes) == 1:
                # Single phenotype case: directly get related diseases from phenotype-disease relationship database
                hpo_id = phenotypes[0]
                
                # Get all diseases associated with this phenotype
                # associated_disease_ids = self.ranker.generator.phenotype_to_diseases.get(hpo_id, set())
                diseases_from_kg = self.ranker.generator.get_diseases_from_hpo_id_from_kg(hpo_id)
                associated_disease_ids = associated_disease_ids = [
                                                                    did
                                                                    for v in diseases_from_kg.values()
                                                                    for did in v.get('disease_id', [])
                                                                ]
                
                # Sort by phenotype frequency in corresponding diseases
                # Build disease-frequency pair list
                disease_frequency_pairs = []
                disease_info = {}
                for disease_id in associated_disease_ids:
                    # Get frequency information of this phenotype in this disease
                    # freq_key = (hpo_id, disease_id)
                    # freq_info = self.ranker.generator.phenotype_disease_frequency.get(freq_key, {})
                    # frequency_numeric = freq_info.get('frequency_numeric', -1)
                    # disease_frequency_pairs.append((disease_id, frequency_numeric))

                    disease_exp_info = self.ranker.generator.get_disease_exp_info_from_kg(disease_id)
                    disease_info[disease_id] = disease_exp_info
                    associated_phenotypes = disease_exp_info.get('phenotypes', [])
                    frequency_numerics = disease_exp_info.get('phenotype_max_frequencies', [])
                    for phenotype, frequency_numeric in zip(associated_phenotypes, frequency_numerics):
                        if phenotype == hpo_id:
                            disease_frequency_pairs.append((disease_id, frequency_numeric))
                            break
                
                # Sort by frequency in descending order (highest frequency first)
                # frequency_numeric: 1.0 (100%) > 0.8 (80%) > ... > 0 (unknown/very rare)
                disease_frequency_pairs.sort(key=lambda x: x[1], reverse=True)
                
                if not associated_disease_ids:
                    return {
                        "success": False,
                        "error": f"No diseases found associated with phenotype {hpo_id}"
                    }
                
                # Build disease case list (limited to final_top_k, using sorted disease list)
                disease_cases = []
                for disease_id, frequency_numeric in disease_frequency_pairs[:final_top_k]:
                    # Get disease name
                    # disease_name = self.ranker.generator.get_disease_name(disease_id)
                    disease_name = disease_info[disease_id].get('standard_name', "")
                    if not disease_name:
                        disease_name = disease_id
                    
                    # Get disease type and description
                    # disease_type = self.ranker.generator.disease_types.get(disease_id, "")
                    # disease_description = self.ranker.generator.get_disease_description(disease_id)
                    disease_type = disease_info[disease_id].get('disease_type', "")
                    disease_description = disease_info[disease_id].get('description', "")
                    if disease_description == "":
                        disease_description = "[Information is missing; please infer based on your memory.]"
                    
                    # Get phenotype information and frequency information for this disease
                    # phenotype_name = self.ranker.generator.get_phenotype_name(hpo_id)
                    phenotype_name = self.ranker.generator.get_phenotype_info_from_kg(hpo_id).get('standard_name', "")
                    phenotype_info = f"{phenotype_name} ({hpo_id})" if phenotype_name else hpo_id
                    
                    # Get frequency text description
                    # freq_key = (hpo_id, disease_id)
                    # freq_info = self.ranker.generator.phenotype_disease_frequency.get(freq_key, {})
                    # frequency_text = freq_info.get('frequency', 'Unknown')
                    
                    disease_cases.append({
                        "disease_id": disease_id,
                        "disease_name": disease_name,
                        "disease_type": disease_type,
                        "disease_description": disease_description,
                        "associated_phenotypes": [phenotype_info],
                        "frequency": frequency_numeric
                    })
                
                # Build formatted return result
                disease_cases_json = {}
                for i, disease_info in enumerate(disease_cases, 1):
                    disease_cases_json[f"Case {i}"] = {
                        "Disease name": disease_info["disease_name"],
                        "Disease id": disease_info["disease_id"],
                        "Disease category": disease_info["disease_type"],
                        "Disease description": disease_info["disease_description"],
                        "Phenotype frequency": disease_info["frequency"],
                    }
                
                return {
                    "disease_cases": disease_cases_json,
                }
            
            # Redirect both stdout and stderr to suppress all output
            with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                # Call ensemble ranker's process_sample method
                result = self.ranker.process_sample(
                    phenotypes=phenotypes,
                    diseases=['OMIM:000000'],  # Set a virtual disease ID to avoid no disease ID situation
                    top_k=top_k,
                    final_top_k=final_top_k,
                    generate_prompt=False,  # Don't generate prompt, only get disease cases
                )

            # print(f"DEBUG: Disease case generation result: {result}")

            if "error" in result:
                return {
                    "success": False,
                    "error": result["error"]
                }
            
            # Format result
            with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                formatted_result = self._format_disease_cases(result, final_top_k)
            
            return formatted_result
            
        except Exception as e:
            print(f"Error during disease case generation: {e}")
            return {
                "success": False,
                "error": f"Disease case generation failed: {str(e)}"
            }
    
    def _format_disease_cases(self, result: Dict[str, Any], final_top_k: int) -> Dict[str, Any]:
        """Format disease case information, build detailed case JSON list"""
        try:
            # Get final ranking results
            final_rankings = result.get("final_rankings", [])
            
            # Build detailed disease case JSON list
            disease_cases_json = {}
            
            for i, disease_info in enumerate(final_rankings[:final_top_k], 1):
                disease_name = disease_info.get("disease_name", "")
                disease_id = disease_info.get("disease_id", "")
                
                # Handle disease_id as list (from rank_diseases_by_phenotype_associations_from_kg_local)
                disease_id_for_lookup = disease_id
                disease_id_for_display = disease_id
                if isinstance(disease_id, list):
                    # Use first element for internal lookups
                    disease_id_for_lookup = disease_id[0] if disease_id else ""
                    # Join all IDs with semicolon for display
                    disease_id_for_display = "; ".join(disease_id) if disease_id else ""
                
                # Get disease type and description
                disease_exp_info = self.ranker.generator.get_disease_exp_info_from_kg(disease_id_for_lookup)
                disease_type = disease_exp_info.get('disease_type', "")
                disease_description = disease_exp_info.get('description', "")
                
                if disease_description == "":
                    disease_description = "[Information is missing; please infer based on your memory.]"
                
                # Get related phenotypes
                associated_phenotypes = disease_info.get("associated_phenotypes", [])
                
                # Build case information
                disease_cases_json[f"Case {i}"] = {
                    "Disease name": disease_name,
                    "Disease id": disease_id_for_display,
                    "Disease category": disease_type,
                    "Disease description": disease_description,
                    # "Matching phenotypes": associated_phenotypes,
                    # "Z statistic": disease_info.get("z_statistic", 0.0),
                    # "Final rank": i
                }
            
            # Build complete return result
            formatted_result = {
                "disease_cases": disease_cases_json,
                # "extraction_methods": self.ranker.extraction_methods
            }
            
            return formatted_result
            
        except Exception as e:
            print(f"Error formatting disease case information: {e}")
            return {
                "success": False,
                "error": f"Formatting failed: {str(e)}"
            }
    
    def extract_and_generate_cases(self, text: str, top_k: int = 100, final_top_k: int = 50) -> Dict[str, Any]:
        """Complete process: extract phenotypes from text and generate disease cases"""
        try:
            # print("Starting complete disease case extraction process...")
            
            # Step 1: Extract phenotypes from text
            _, extracted_phenotypes = self.extract_phenotypes_from_text(text)
            
            if not extracted_phenotypes:
                return {
                    "success": False,
                    "error": f"No valid phenotypes extracted from input: {text}"
                }
            
            # Step 2: Extract disease cases separately for each independent symptom set (hpo_ids_str key)
            # extracted_phenotypes format: {hpo_ids_str: phenotype_details_structured, ...}
            all_disease_cases = {}
            all_extracted_phenotypes = {}
            
            for hpo_ids_str, phenotype_details_structured in extracted_phenotypes.items():
                # Extract HPO IDs list for this symptom set from phenotype_details_structured
                hpo_ids = []
                if isinstance(phenotype_details_structured, dict):
                    for phenotype_name, details in phenotype_details_structured.items():
                        hpo_id = details.get("hpo_id")
                        if hpo_id and hpo_id not in hpo_ids:
                            hpo_ids.append(hpo_id)
                
                if not hpo_ids:
                    continue
                
                # Generate disease cases separately for this symptom set
                case_result = self.generate_disease_cases(hpo_ids, top_k, final_top_k)
                
                if "error" in case_result:
                    # If extraction fails for a symptom set, log error but continue processing other sets
                    print(f"Error generating cases for symptom set {hpo_ids_str}: {case_result.get('error')}")
                    continue
                
                # Store results under corresponding hpo_ids_str key
                all_disease_cases[hpo_ids_str] = case_result.get("disease_cases", {})
                all_extracted_phenotypes[hpo_ids_str] = phenotype_details_structured
            
            if not all_disease_cases:
                return {
                    "success": False,
                    "error": "Failed to generate disease cases for any symptom set"
                }
            
            # Step 3: Build final result
            result = {
                "extracted_disease_cases": all_disease_cases,
                "extracted_phenotypes": all_extracted_phenotypes
            }
            
            # print(f"Disease case extraction completed, generated cases for {len(all_disease_cases)} symptom sets")
            return result
            
        except Exception as e:
            print(f"Error during complete process execution: {e}")
            return {
                "success": False,
                "error": f"Complete process execution failed: {str(e)}"
            }
    
def extract_disease_cases(query: str, top_k: int = 100, final_top_k: int = 50) -> str:
    """
    Disease Case Extractor Tool: Extract phenotypes from query and generate disease cases using ensemble methods.
    
    Args:
        query: Input query containing phenotype descriptions
        top_k: Number of diseases extracted by each method (default: 100)
        final_top_k: Number of final disease cases to return (default: 50)
    
    Returns:
        JSON string containing disease cases and extraction results
    """
    global _initialization_failed
    
    if _initialization_failed:
        return json.dumps({
            "success": False,
            "error": "Disease case extractor initialization failed"
        }, ensure_ascii=False, indent=2)
    
    # Check if input is provided
    if not query.strip():
        return json.dumps({
            "success": False,
            "error": "No query provided for disease case extraction"
        }, ensure_ascii=False, indent=2)
    
    try:
        # Get disease case extractor instance
        extractor = _get_disease_case_extractor()
        if not extractor:
            return json.dumps({
                "success": False,
                "error": "Failed to initialize disease case extractor"
            }, ensure_ascii=False, indent=2)
        
        # Extract phenotypes and generate disease cases
        result = extractor.extract_and_generate_cases(
            text=query,
            top_k=top_k,
            final_top_k=final_top_k
        )

        # Check if there are any errors
        if "error" in result:
            return json.dumps(result, ensure_ascii=False, indent=2)

        # Reorganize result format
        # result structure: {
        #     "extracted_disease_cases": {hpo_ids_str: disease_cases, ...},
        #     "extracted_phenotypes": {hpo_ids_str: phenotype_details_structured, ...}
        # }
        # Need to convert to: {
        #     "extracted_disease_cases": {
        #         hpo_ids_str: {
        #             "extracted_phenotypes": phenotype_details_structured,
        #             "disease_cases": disease_cases
        #         }, ...
        #     }
        # }
        extracted_disease_cases = result.get("extracted_disease_cases", {})
        extracted_phenotypes = result.get("extracted_phenotypes", {})
        
        formatted_result = {
            "extracted_disease_cases": {}
        }
        
        # Organize data for each symptom set (hpo_ids_str key)
        for hpo_ids_str in extracted_disease_cases.keys():
            formatted_result["extracted_disease_cases"][hpo_ids_str] = {
                "extracted_phenotypes": extracted_phenotypes.get(hpo_ids_str, {}),
                "disease_cases": extracted_disease_cases.get(hpo_ids_str, {})
            }
        
        return json.dumps(formatted_result, ensure_ascii=False, indent=2)
        
    except Exception as e:
        return json.dumps({
            "success": False,
            "error": f"Error during disease case extraction: {str(e)}"
        }, ensure_ascii=False, indent=2)

def get_tool_definition() -> types.Tool:
    """Return tool definition"""
    return types.Tool(
        name="disease-case-extractor",
        description="Extract phenotypes from query and generate disease cases using ensemble methods.",
        inputSchema={
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Input query containing phenotype descriptions"
                },
                "top_k": {
                    "type": "integer",
                    "description": "Number of diseases extracted by each method (default: 100)",
                    "default": 100
                },
                "final_top_k": {
                    "type": "integer",
                    "description": "Number of final disease cases to return (default: 50)",
                    "default": 50
                }
            },
            "required": ["query"]
        }
    )

async def call_tool(context, arguments: Dict[str, Any]) -> List[types.ContentBlock]:
    """Execute disease case extraction tool"""
    try:
        query = arguments.get("query", "")
        top_k = arguments.get("top_k", 100)
        final_top_k = arguments.get("final_top_k", 50)
        
        result = extract_disease_cases(query, top_k, final_top_k)
        
        return [types.TextContent(
            type="text",
            text=result
        )]
        
    except Exception as e:
        error_msg = {
            "success": False,
            "error": f"Error during disease case extraction: {str(e)}"
        }
        return [types.TextContent(
            type="text",
            text=json.dumps(error_msg, ensure_ascii=False, indent=2)
        )]
