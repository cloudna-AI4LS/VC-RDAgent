#!/usr/bin/env python3
"""
MCP server tool to extract disease names from free text, normalize them to standard IDs,
and enrich with detailed information using the existing disease case extractor's dataset.
"""

import os
import re
import json
import sys
import contextlib
import io
from typing import Dict, List, Any, Optional
from mcp import types
from langchain.chat_models import init_chat_model
from langchain_core.messages import HumanMessage, SystemMessage

# Add parent directory to path to import top-level modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import dependencies for normalization and details enrichment
from tools.disease_case_extractor import _get_disease_case_extractor
from tools.phenotype_extractor import _get_phenotype_analyzer

# Import load_config function
try:
    from scripts.rare_disease_diagnose.query_kg import load_config
except ImportError as e:
    print(f"Warning: Could not import load_config: {e}")
    def load_config(config_file: str = "config.json") -> dict:
        """Fallback load_config function"""
        try:
            with open(config_file, 'r', encoding='utf-8') as f:
                config = json.load(f)
            # Replace {base_path} parameter in all string values
            if 'base_path' in config:
                base_path = config['base_path']
                for key, value in config.items():
                    if isinstance(value, str) and '{base_path}' in value:
                        config[key] = value.format(base_path=base_path)
                    elif isinstance(value, dict):
                        for subkey, subvalue in value.items():
                            if isinstance(subvalue, str) and '{base_path}' in subvalue:
                                config[key][subkey] = subvalue.format(base_path=base_path)
            return config
        except Exception:
            return {}

# Base directory for resolving relative paths (based on file location)
BASE_DIR = os.path.dirname(__file__)

# Load configuration and create model instance
config_file = os.path.join(os.path.dirname(BASE_DIR), 'scripts/rare_disease_diagnose/prompt_config_forKG.json')
config = load_config(config_file)

# Get model configuration from config file
model_config = config.get('model_config', {})

# Build init_chat_model arguments
init_args = {
    'model': model_config.get('model', 'Qwen/Qwen3-8B'),
    'base_url': model_config.get('base_url', 'http://192.168.0.127:8000/v1'),
    'api_key': model_config.get('api_key', 'EMPTY'),
    'temperature': model_config.get('temperature', 0.0),
    'top_p': model_config.get('top_p', 0.95),
    'streaming': False,
}

# Only add model_provider if it's not empty, otherwise use default 'openai' for OpenAI-compatible APIs
model_provider = model_config.get('model_provider', '').strip()
if model_provider:
    init_args['model_provider'] = model_provider
else:
    # Default to 'openai' if base_url looks like OpenAI-compatible API
    init_args['model_provider'] = 'openai'

model = init_chat_model(**init_args)

def _extract_disease_names_with_model(text: str) -> List[str]:
    """Extract complete disease names from raw text using the same model
    initialization approach as phenotype_to_disease_agent_langchain.py.

    Input: raw user text string
    Output: strictly parsed disease name list (deduplicated, no trimming)
    """
    
    system = SystemMessage(content=(
        "You extract ONLY explicitly mentioned disease/condition names or disease IDs (e.g., OMIM:123456, ORPHA:123456, MONDO:123456) from the user query. "
        "DO NOT infer or suggest diseases based on the user query."
        "If no disease/condition names or disease IDs are explicitly mentioned in the user query, return empty list. "
        "Output STRICT JSON only with exact schema: {\"diseases\": [\"...\"]}. "
        "No extra text, no markdown, no explanations."
    ))
    user = HumanMessage(content=(
        "Query:\n" + text.strip()
    ))

    try:
        resp = model.invoke([system, user])
        # print(f"DEBUG: resp: {resp}")
        raw = getattr(resp, "content", "")
        # Remove content inside <think> tags if present
        raw = re.sub(r'<think>.*?</think>', '', raw, flags=re.DOTALL)
        raw = raw.strip()
        
        # print(f"DEBUG: raw: {raw}")

        # Extract JSON object {"diseases": [...]} from raw text
        # Find all matches and use the last one (usually more accurate)
        json_matches = re.findall(r'\{"diseases"\s*:\s*\[.*?\]\s*\}', raw, re.DOTALL)
        if json_matches:
            raw = json_matches[-1]
        
        # print(f"DEBUG: extracted raw: {raw}")

        # print(f"DEBUG: raw: {raw}")
        data = json.loads(raw)

        diseases = data.get("diseases", []) if isinstance(data, dict) else []
    except Exception:
        diseases = []

    # Clean, deduplicate, and return
    seen = set()
    result: List[str] = []
    for name in diseases:
        if not isinstance(name, str):
            continue
        clean = name.strip()
        key = clean.lower()
        if clean and key not in seen:
            seen.add(key)
            result.append(clean)

    return result


def _split_disease_names_for_list(disease_names: List[str]) -> List[str]:
    out: List[str] = []
    for dn in disease_names:
        if ';' not in dn:
            out.append(dn.strip())
            continue
        parts = [p.strip() for p in dn.split(';') if p.strip()]
        for p in parts:
            if p not in out:
                out.append(p)
    return out


def _extract_key_parts(name_lower: str) -> List[str]:
    common_words = {
        'syndrome', 'disease', 'disorder', 'condition', 'anomaly', 'defect',
        'malformation', 'abnormality', 'deficiency', 'insufficiency',
        'type', 'form', 'variant', 'subtype', 'class', 'category',
        'autosomal', 'dominant', 'recessive', 'x-linked', 'y-linked',
        'inherited', 'genetic', 'congenital', 'hereditary', 'familial',
        'rare', 'common', 'frequent', 'occasional', 'very', 'extremely',
        'with', 'of', 'and', 'or',
    }
    tokens = re.split(r"[^a-z0-9\-]+", name_lower)
    return [t for t in tokens if t and t not in common_words and len(t) > 1]


def _check_key_parts_match(key_parts: List[str], names_lower: List[str]) -> List[str]:
    matched: List[str] = []
    for cand in names_lower:
        ok = True
        for kp in key_parts:
            if kp not in cand:
                ok = False
                break
        if ok:
            matched.append(cand)
    return matched


def _fast_filter_key_parts_from_hpoa(disease_name: str, gen: Any) -> List[Dict[str, Any]]:
    candidates: List[Dict[str, Any]] = []
    name_lower = disease_name.lower()
    key_parts = _extract_key_parts(name_lower)
    max_candidates = 100
    count = 0
    
    # try:
    #     hpoa_diseases = set(getattr(gen, 'disease_to_phenotypes', {}).keys())
    # except Exception:
    #     hpoa_diseases = set()
    # if not hpoa_diseases:
    #     return []
    
    # Read local disease ID to name mapping table for fast querying
    # Build a dictionary mapping disease_id (key) to {standard_name, synonyms list}
    disease_ids_names = {}
    with open(os.path.join(BASE_DIR, '../data/disease_annotations/disease_ids_names.json'), 'r') as f:
        disease_ids_names = json.load(f)
    
    for did in disease_ids_names:
        if count >= max_candidates:
            break
        try:
            std = disease_ids_names[did].get('standard_name', "")
            std_lower = (std or did).lower()
            syns = []
            try:
                syns = disease_ids_names[did].get('synonyms', [])
            except Exception:
                syns = []
            syns_lower = [s.lower() for s in syns if isinstance(s, str)]
            all_names_lower = [std_lower] + syns_lower
            matched = _check_key_parts_match(key_parts, all_names_lower)
            if matched:
                candidates.append({
                    'disease_id': did,
                    'standard_name': std or did,
                    'matched_names': matched,
                })
                count += 1
        except Exception:
            continue
    return candidates


def _normalize_disease_names(names: List[str]) -> Dict[str, Dict[str, Any]]:
    """Two-stage disease matching: replicate _find_best_matching_disease_two_stage
    logic but using phenotype_extractor's analyzer.generator.
    """
    # Handle None or empty input
    if not names:
        return {}
    
    try:
        analyzer = _get_phenotype_analyzer()
    except Exception:
        analyzer = None
    if analyzer is None:
        return {}

    gen = analyzer.generator

    normalized: Dict[str, Dict[str, Any]] = {}
    for input_name in names:
        if isinstance(input_name, str) and re.match(r"^(OMIM|ORPHA|MONDO):[A-Za-z0-9_.-]+$", input_name.strip(), flags=re.IGNORECASE):
            try:
                did = input_name.strip()
                # Normalize prefix to original case as provided; generator should accept canonical IDs
                # std_name = gen.get_disease_name(did)
                std_name = gen.get_disease_info_from_kg(did).get('standard_name', "")
                # If same input_name already exists, keep the one with higher similarity
                if input_name not in normalized or 1.0 >= normalized[input_name].get('similarity', 0.0):
                    normalized[input_name] = {
                        'standard_name': std_name or did,
                        'disease_id': did,
                        'similarity': 1.0,
                    }
                continue
            except Exception:
                # If direct lookup fails, fall back to matching flow below
                pass

        try:
            candidates = _fast_filter_key_parts_from_hpoa(input_name, gen)
            # print(f"DEBUG: candidates: {candidates}")
            if not candidates:
                continue
            best_match = None
            best_similarity = 0.0
            for c in candidates:
                cand_names = _split_disease_names_for_list(c.get('matched_names', []))
                max_sim = 0.0
                best_cand_name = c.get('standard_name', '')
                for cand_name in cand_names:
                    try:
                        # Suppress potential tqdm progress bars emitted to stdout/stderr
                        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                            sim = gen.calculate_semantic_similarity(input_name.lower(), cand_name)
                    except Exception:
                        sim = 0.0
                    if sim > max_sim:
                        max_sim = sim
                        best_cand_name = cand_name
                if max_sim > best_similarity:
                    best_similarity = max_sim
                    best_match = {**c, 'similarity': max_sim, 'matched_candidate_name': best_cand_name}

            # Keep consistent with original: ignore low similarity (threshold default 0.8)
            threshold = float(os.getenv('DISEASE_NAME_SIM_THRESHOLD', '0.8'))
            if best_match and best_similarity >= threshold:
                # If same input_name already exists, keep the one with higher similarity
                if input_name not in normalized or float(best_similarity) > normalized[input_name].get('similarity', 0.0):
                    normalized[input_name] = {
                        'standard_name': best_match.get('standard_name', input_name),
                        'disease_id': best_match.get('disease_id', 'UNKNOWN'),
                        'similarity': float(best_similarity),
                    }
        except Exception as e:
            # print(f"DEBUG: Exception when processing '{input_name}': {type(e).__name__}: {str(e)}")
            continue

    return normalized


def _enrich_with_details(standardized: Dict[str, Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    """Fetch disease details using ranker.generator from disease_case_extractor."""
    extractor = _get_disease_case_extractor()
    if extractor is None:
        return standardized

    gen = extractor.ranker.generator
    enriched: Dict[str, Dict[str, Any]] = {}
    for extracted_name, item in standardized.items():
        did = item.get("disease_id")
        # Get disease information from knowledge graph
        disease_exp_info = gen.get_disease_exp_info_from_kg(did)
        
        try:
            # disease_name = gen.get_disease_name(did)
            # disease_type = gen.disease_types.get(did, "")
            # description = gen.get_disease_description(did)
            disease_name = disease_exp_info.get('standard_name', "")
            disease_type = disease_exp_info.get('disease_type', "")
            description = disease_exp_info.get('description', "")
            
            # Get all disease aliases/synonyms
            aliases = []
            try:
                # aliases = gen.get_disease_all_names(did)
                aliases = disease_exp_info.get('synonyms', [])
                # Filter out the standard name from aliases
                if disease_name and isinstance(aliases, list):
                    aliases = [a for a in aliases if a != disease_name]
            except Exception:
                aliases = []
            
            # Get all associated phenotypes/symptoms grouped by frequency
            phenotypes_by_frequency = {
                'Obligate': [],  # Obligate; 100%
                'Very frequent': [],  # Very frequent; 80%-99%
                'Frequent': [],  # Frequent; 30%-79%
                'Occasional': [],  # Occasional; 5%-29%
                'Very rare': [],  # Very rare; 1%-4%
                'Excluded': [],  # Excluded; 0%
                'Unknown': []  # Unknown frequency (frequency_numeric == -1)
            }
            
            try:
                # if hasattr(gen, 'disease_to_phenotypes') and did in gen.disease_to_phenotypes:
                if disease_exp_info:
                    # hpo_ids = gen.disease_to_phenotypes.get(did, set())
                    hpo_ids = disease_exp_info.get('phenotypes', [])
                    frequency_numerics = disease_exp_info.get('phenotype_max_frequencies', [])
                    # Convert to list if it's a set
                    if isinstance(hpo_ids, set):
                        hpo_ids = list(hpo_ids)
                    # Get phenotype names for each HPO ID and group by frequency
                    for idx, hpo_id in enumerate(hpo_ids):
                        try:
                            # phenotype_name = gen.get_phenotype_name(hpo_id)
                            phenotype_name = gen.get_phenotype_info_from_kg(hpo_id).get('standard_name', "")
                            phenotype_str = f"{phenotype_name} ({hpo_id})" if phenotype_name else hpo_id
                            
                            # Get frequency information
                            # freq_key = (hpo_id, did)
                            # freq_info = gen.phenotype_disease_frequency.get(freq_key, {})
                            # frequency_numeric = freq_info.get('frequency_numeric', -1)
                            
                            # Use enumerate index instead of index() to avoid issues with duplicate hpo_ids
                            # Check if frequency_numerics has enough elements
                            if idx < len(frequency_numerics):
                                frequency_numeric = frequency_numerics[idx]
                            else:
                                frequency_numeric = -1
                        
                            # Convert to float, handling None, empty string, etc.
                            if frequency_numeric is not None and frequency_numeric != "":
                                try:
                                    frequency_numeric = float(frequency_numeric)
                                except (ValueError, TypeError):
                                    frequency_numeric = -1
                            else:
                                frequency_numeric = -1
                            
                            # Categorize by frequency
                            if frequency_numeric == -1:
                                phenotypes_by_frequency['Unknown'].append(phenotype_str)
                            elif frequency_numeric == 0.0:
                                phenotypes_by_frequency['Excluded'].append(phenotype_str)
                            elif 0.01 <= frequency_numeric <= 0.04:
                                phenotypes_by_frequency['Very rare'].append(phenotype_str)
                            elif 0.05 <= frequency_numeric <= 0.29:
                                phenotypes_by_frequency['Occasional'].append(phenotype_str)
                            elif 0.3 <= frequency_numeric <= 0.79:
                                phenotypes_by_frequency['Frequent'].append(phenotype_str)
                            elif 0.8 <= frequency_numeric <= 0.99:
                                phenotypes_by_frequency['Very frequent'].append(phenotype_str)
                            elif frequency_numeric >= 1.0:
                                phenotypes_by_frequency['Obligate'].append(phenotype_str)
                            else:
                                # Fallback for any other values
                                phenotypes_by_frequency['Unknown'].append(phenotype_str)
                        except Exception:
                            # If can't get name or frequency, add to unknown
                            phenotypes_by_frequency['Unknown'].append(hpo_id)
            except Exception:
                phenotypes_by_frequency = {
                    'Obligate': ["error"],  # Obligate; 100%
                    'Very frequent': ["error"],  # Very frequent; 80%-99%
                    'Frequent': ["error"],  # Frequent; 30%-79%
                    'Occasional': ["error"],  # Occasional; 5%-29%
                    'Very rare': ["error"],  # Very rare; 1%-4%
                    'Excluded': ["error"],  # Excluded; 0%
                    'Unknown': ["error"]  # Unknown frequency (frequency_numeric == -1)
                    }
                pass
            
            if description == "":
                description = "[Information is missing; please infer based on your memory.]"
            enriched[extracted_name] = {
                **item,
                "disease_category": disease_type,
                "disease_description": description,
                "disease_aliases": aliases,
                "associated_phenotypes_by_frequency": phenotypes_by_frequency,
            }
        except Exception:
            enriched[extracted_name] = item
    return enriched


def disease_information_retrieval(query: str, use_model: bool = True, candidates: List[str] = None) -> str:
    """
    Disease Information Retrieval Tool: extract disease names from query, normalize to standard disease names,
    and enrich with detailed information.

    Args:
        query: input query
        use_model: whether to use the model to extract disease names (default True)
        candidates: optional list of disease name candidates

    Returns:
        JSON string containing extracted candidates, normalized results, and details.
    """

    if not query or not query.strip():
        return json.dumps({
            "success": False,
            "error": "No query provided"
        }, ensure_ascii=False, indent=2)

    try:
        # Step 1: extract candidate disease names
        if use_model:
            candidates = _extract_disease_names_with_model(query)
        elif candidates is None:
            # If use_model is False and no candidates provided, return error
            return json.dumps({
                "success": False,
                "error": "No candidates provided and use_model is False"
            }, ensure_ascii=False, indent=2)

        # Limit the number of candidates to 5
        if len(candidates) > 5:
            candidates = candidates[:5]

        # Step 2: normalize to disease IDs
        normalized = _normalize_disease_names(candidates)
        # Step 3: enrich with detailed information
        with contextlib.redirect_stdout(io.StringIO()):
            detailed = _enrich_with_details(normalized)

        return json.dumps({
            "extracted_diseases": detailed,
        }, ensure_ascii=False, indent=2)

    except Exception as e:
        return json.dumps({
            "success": False,
            "error": f"Extraction error: {str(e)}"
        }, ensure_ascii=False, indent=2)

def get_tool_definition() -> types.Tool:
    """Return tool definition"""
    return types.Tool(
        name="disease-information-retrieval",
        description="Extract disease names from query, normalize to standard disease names, and enrich with detailed information.",
        inputSchema={
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Input query containing disease names or descriptions"
                },
                "use_model": {
                    "type": "boolean",
                    "description": "Whether to use the model to extract disease names (default True)",
                    "default": True
                },
                "candidates": {
                    "type": "array",
                    "items": {
                        "type": "string"
                    },
                    "description": "Optional list of disease name candidates (used when use_model is False)"
                }
            },
            "required": ["query"]
        }
    )

async def call_tool(context, arguments: Dict[str, Any]) -> List[types.ContentBlock]:
    """Execute disease information retrieval tool"""
    try:
        query = arguments.get("query", "")
        use_model = arguments.get("use_model", True)
        candidates = arguments.get("candidates")
        
        result = disease_information_retrieval(query, use_model, candidates)
        
        return [types.TextContent(
            type="text",
            text=result
        )]
        
    except Exception as e:
        error_msg = {
            "success": False,
            "error": f"Error during disease information retrieval: {str(e)}"
        }
        return [types.TextContent(
            type="text",
            text=json.dumps(error_msg, ensure_ascii=False, indent=2)
        )]


