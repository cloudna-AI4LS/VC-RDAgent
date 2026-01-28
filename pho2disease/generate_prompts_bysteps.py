#!/usr/bin/env python3
"""
Generate phenotype-to-disease prompts from input phenotype-disease JSON.

- Input: JSON array of [phenotypes, diseases] per sample, e.g.
  [[["HP:0001","HP:0002"], ["OMIM:123456"]], ...]
- Supports 2-step and 3-step prompt schemes.

Config (prompt_config.json):
  base_path:  project root; paths can use {base_path}.
  input_file: path to the phenotype-disease JSON.

Usage:

  # 1. Config-only (paths from prompt_config.json)
  python generate_prompts_bysteps.py --config prompt_config.json

  # 2. Overlap, 2-step, no true samples
  python generate_prompts_bysteps.py --input_file /path/to/sample.json \\
    --case_extraction_method overlap --prompt_steps 2 --top_k 50 \\
    --output_file output_2steps_overlap.json

  # 3. Embedding, 2-step (needs embedding_file, ic_file, case_library in config or CLI)
  python generate_prompts_bysteps.py --input_file /path/to/sample.json \\
    --case_extraction_method embedding --prompt_steps 2 \\
    --output_file output_2steps_embedding.json

  # 4. Both (overlap + embedding), 2-step
  python generate_prompts_bysteps.py --input_file /path/to/sample.json \\
    --case_extraction_method both --prompt_steps 2 \\
    --output_file output_2steps_both.json

  # 5. Overlap + true samples (case_library as phenotype-disease source; current sample excluded)
  python generate_prompts_bysteps.py --input_file /path/to/sample.json \\
    --case_extraction_method overlap --prompt_steps 2 --top_k 50 \\
    --use_samples --case_library /path/to/true_samples.jsonl \\
    --output_file output_2steps_overlap.json

  # 6. Pick GPU and sample indices
  python generate_prompts_bysteps.py --config prompt_config.json \\
    --gpu_id 0 --sample_indices 0,5,10 --output_file output_selected.json

  # 7. Save case library only (no prompts)
  #    Exports phenotype-disease case library from loaded KB (phenotype.hpoa, Orphanet, etc.) to
  #    general_cases/: phenotype_disease_case_library.jsonl, phenotype_disease_case_library_with_high_freq.jsonl,
  #    phenotype_disease_case_database.json, disease_ids_names.json, and graph CSVs. Then exits.
  python generate_prompts_bysteps.py --config prompt_config.json --save_case_library_only

Key parameters:
  case_extraction_method  overlap | embedding | both
  use_samples             use phenotype-disease from case_library (--case_library) for ranking; exclude current sample
  case_library            path to case library JSONL (true or virtual); required for embedding/both and when use_samples
  prompt_steps            2 or 3 (step structure: phenotype->disease; step1+case->disease; optional case->disease)
  top_k                   number of candidate diseases to include
  use_IC_weights          use Information Content for ranking
  save_case_library_only  only export case library (and DB, graph files) from KB to general_cases/; do not generate prompts
"""

import json
import sys
import os
from datetime import datetime
import pandas as pd
import random
import numpy as np
import torch
import time
import re
import argparse
import csv
import math
import requests
from collections import defaultdict
from typing import Any, List, Dict, Set, Optional, Tuple, Iterable
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForCausalLM
from analysis.disease_scraper.disease_scraper import get_disease_description as disease_description_scraper
from LR import (
    PhenotypeRankingEngine,
    load_minimal_ontology_from_hp_json,
    load_disease_models_from_hpoa,
)  

def parse_sample_indices(indices_str: str) -> List[int]:
    """
    Parse comma-separated sample indices string into a list of integers.
    Supports both individual indices and ranges.
    
    Examples:
        "0,5,10" -> [0, 5, 10]
        "0-5,10-15" -> [0, 1, 2, 3, 4, 5, 10, 11, 12, 13, 14, 15]
        "0,5-7,10" -> [0, 5, 6, 7, 10]
    """
    if not indices_str or indices_str.strip() == "":
        return []
    
    indices = []
    parts = indices_str.split(',')
    
    for part in parts:
        part = part.strip()
        if '-' in part:
            # Handle range (e.g., "0-5")
            try:
                start, end = map(int, part.split('-'))
                indices.extend(range(start, end + 1))
            except ValueError:
                raise ValueError(f"Invalid range format: {part}")
        else:
            # Handle single index
            try:
                indices.append(int(part))
            except ValueError:
                raise ValueError(f"Invalid index format: {part}")
    
    # Remove duplicates and sort
    return sorted(list(set(indices)))

def find_file_path(possible_paths: List[str]) -> Optional[str]:
    """Find the first existing file from a list of possible paths"""
    for path in possible_paths:
        if os.path.exists(path):
            return path
    return None

def load_disease_phenotype_mapping(jsonl_file_path: str) -> Dict[str, Set[str]]:
    """
    Load disease-phenotype association DB
    
    Read disease-phenotype associations from JSONL, return disease_id -> set of phenotype_ids.
    Each disease may appear in multiple rows; all associations are merged.
    
    Args:
        jsonl_file_path: path to JSONL; each line has:
            - "RareDisease": list of disease IDs (e.g. ["OMIM:602152"] or ["ORPHA:140976"])
            - "Phenotype": list of phenotype IDs (e.g. ["HP:0000107", "HP:0000364", ...])
            - "Department": department (optional, may be null)
    
    Returns:
        Dict[str, Set[str]]: dict, key=disease ID, value=set of associated phenotype IDs
        
    Example:
        >>> mapping = load_disease_phenotype_mapping("phenotype_disease_case_library_expanded.jsonl")
        >>> print(mapping.get("OMIM:602152"))
        {'HP:0000107', 'HP:0000364', 'HP:0000556', ...}
    """
    disease_phenotype_mapping: Dict[str, Set[str]] = defaultdict(set)
    
    # try to find file path
    file_path = find_file_path([jsonl_file_path])
    if not file_path:
        print(f"Warning: File not found: {jsonl_file_path}")
        return dict(disease_phenotype_mapping)
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:  # skip empty line
                    continue
                
                try:
                    data = json.loads(line)
                    
                    # get disease ID list
                    rare_diseases = data.get('RareDisease', [])
                    if not rare_diseases:
                        continue
                    
                    # get phenotype ID list
                    phenotypes = data.get('Phenotype', [])
                    if not phenotypes:
                        continue
                    
                    # add associated phenotypes for each disease
                    for disease_id in rare_diseases:
                        disease_id = disease_id.strip()  # strip
                        if disease_id.startswith("OMIM:") or disease_id.startswith("ORPHA:"):  # ensure valid
                            # add all phenotypes to disease set (auto dedup)
                            disease_phenotype_mapping[disease_id].update(phenotypes)
                            
                except json.JSONDecodeError as e:
                    print(f"Warning: Invalid JSON at line {line_num} in {file_path}: {e}")
                    continue
                except Exception as e:
                    print(f"Warning: Error processing line {line_num} in {file_path}: {e}")
                    continue
        
        print(f"Loaded disease-phenotype mapping from {file_path}: {len(disease_phenotype_mapping)} diseases, "
              f"{sum(len(phenotypes) for phenotypes in disease_phenotype_mapping.values())} total phenotype associations")
        
    except FileNotFoundError:
        print(f"Error: File not found: {file_path}")
        return {}
    except Exception as e:
        print(f"Error reading file {file_path}: {e}")
        return {}
    
    # convert defaultdict to dict for return
    return dict(disease_phenotype_mapping)

def set_gpu_id(gpu_id: int):
    """Set GPU ID for CUDA operations"""
    if gpu_id is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
        print(f"Set GPU ID to: {gpu_id}")
    else:
        print("No GPU ID specified, using default CUDA settings")

class KnowledgeGraphQuery:
    """
    Knowledge Graph Query client for querying knowledge graph API.
    Supports multiple search types: simple, boolean, fuzzy, network, and relationship searches.
    """
    def __init__(self, api_url: str = "http://192.168.0.9:5008/nebulasearch/", 
                 space: str = "RDPkg", timeout: int = 30):
        """
        Initialize Knowledge Graph Query client.
        
        Args:
            api_url: Knowledge graph API URL
            space: Default knowledge graph space
            timeout: Request timeout in seconds
        """
        self.kg_api_url = api_url
        self.kg_api_space = space
        self.kg_api_timeout = timeout
        self.kg_api_cache = {}  # Cache for API query results

    def search(self, search_type: str, ent1: str = None, prop1: str = None, label1: str = None,
               ent2: str = None, prop2: str = None, label2: str = None,
               boolean_op: str = None, space: str = None, use_cache: bool = True) -> Optional[Dict]:
        """
        Unified search method for knowledge graph queries.
        
        Args:
            search_type: Search type - "simplesearch", "boolsearch", "fuzzysearch", "netsearch", or "relsearch"
            ent1: Entity 1 identifier
            prop1: Property 1 (optional)
            label1: Label 1 (optional)
            ent2: Entity 2 identifier (optional, for boolsearch and relsearch)
            prop2: Property 2 (optional, for boolsearch and relsearch)
            label2: Label 2 (optional, for boolsearch and relsearch)
            boolean_op: Boolean operator ("and" or "or", for boolsearch only)
            space: Knowledge graph space (defaults to self.kg_api_space)
            use_cache: Whether to use cached results, default True
            
        Returns:
            Optional[Dict]: API response as dictionary, or None if query fails
        """

        # define var names directly to avoid arg mismatch
        _serchtype = search_type
        _space = space if space is not None else self.kg_api_space
        _ent1 = ent1 if ent1 is not None else ""
        _prop1 = prop1 if prop1 is not None else ""
        _label1 = label1 if label1 is not None else ""
        _ent2 = ent2 if ent2 is not None else ""
        _prop2 = prop2 if prop2 is not None else ""
        _label2 = label2 if label2 is not None else ""
        _boolean = boolean_op if boolean_op is not None else ""
        
        payload = {
            "serchtype": _serchtype,
            "space": _space,
            "ent1": _ent1,
            "prop1": _prop1,
            "label1": _label1,
            "ent2": _ent2,
            "prop2": _prop2,
            "label2": _label2,
            "Boolean": _boolean
        }
        
        print(f"Payload: {payload}")
        
        return self.query_knowledge_graph(payload, use_cache=use_cache)
    
    def query_knowledge_graph(self, payload: Dict, use_cache: bool = True) -> Optional[Dict]:
        """
        Query knowledge graph API with a given payload (low-level method).
        
        Args:
            payload (Dict): Request payload dictionary containing API parameters
            use_cache (bool): Whether to use cached results, default True
            
        Returns:
            Optional[Dict]: API response as dictionary, or None if query fails
        """
        # Create a copy of payload to avoid modifying the original
        request_payload = payload.copy()
        
        # Ensure space is set if not provided in payload
        if "space" not in request_payload:
            request_payload["space"] = self.kg_api_space
        
        # Generate cache key from final payload
        cache_key = str(sorted(request_payload.items()))
        if use_cache and cache_key in self.kg_api_cache:
            return self.kg_api_cache[cache_key]
        
        try:
            # Make API request
            response = requests.post(
                self.kg_api_url,
                json=request_payload,
                headers={
                    "accept": "application/json",
                    "Content-Type": "application/json"
                },
                timeout=self.kg_api_timeout
            )
            
            # Check if request was successful
            if response.status_code == 200:
                result = response.json()
                # Cache the result
                if use_cache:
                    self.kg_api_cache[cache_key] = result
                return result
            else:
                print(f"Knowledge graph API returned status code {response.status_code}: {response.text}")
                return None
                
        except requests.exceptions.Timeout:
            print(f"Knowledge graph API request timed out after {self.kg_api_timeout} seconds")
            return None
        except requests.exceptions.RequestException as e:
            print(f"Error querying knowledge graph API: {str(e)}")
            return None
        except Exception as e:
            print(f"Unexpected error in knowledge graph query: {str(e)}")
            return None

class LocalKnowledgeGraphQuery:
    """Helper class for local KG data operations."""
    
    # class-level cache
    _kg_data_cache = None
    _kg_data_loaded = False
    
    @classmethod
    def load_kg_data(cls, config: Dict) -> Dict:
        """
        Load local KG data from CSV and JSON files. Only loads once using class attribute cache.
        
        Args:
            config: Configuration dictionary containing kg_dir and base_path
            
        Returns:
            Dictionary containing all KG data structures
        """
        # check if already loaded
        if cls._kg_data_loaded and cls._kg_data_cache is not None:
            return cls._kg_data_cache
        
        # read kg_dir from config
        kg_dir = config.get("kg_dir", "")
        if not kg_dir:
            raise ValueError("kg_dir must be specified in config")
        
        # resolve {base_path} in config
        base_path = config.get("base_path", "")
        if base_path:
            kg_dir = kg_dir.replace("{base_path}", base_path)
        
        kg_dir = os.path.abspath(kg_dir)
        
        # init data structures
        phenotype_to_diseases = defaultdict(set)
        disease_to_phenotypes = defaultdict(set)
        disease_phenotype_counts = defaultdict(int)
        hpo_is_a = defaultdict(list)  # phenotype_id -> list of parent_ids
        parent_to_children = defaultdict(list)  # parent_id -> list of child_ids
        ic_dict = {}
        disease_names = {}
        disease_name_to_ids = defaultdict(set)
        disease_mapping_with_synonyms = defaultdict(list)
        disease_descriptions = {}  # disease_id -> description
        disease_types = {}  # disease_id -> disease_type
        phenotype_disease_frequency = {}  # (phenotype_id, disease_id) -> frequency_info
        phe2embedding = {}
        disease_id_mapping = {}  # aggregated_id -> list of public_ids
        
        # 1. load disease nodes
        disease_nodes_file = os.path.join(kg_dir, "disease_nodes.csv")
        if os.path.exists(disease_nodes_file):
            with open(disease_nodes_file, 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    disease_id = row['ID'].strip()
                    standard_name = row.get('standard_name', '').strip()
                    synonyms_str = row.get('synonyms', '').strip()
                    description = row.get('description', '').strip()
                    disease_type = row.get('disease_type', '').strip()
                    
                    if disease_id and standard_name:
                        disease_names[disease_id] = standard_name
                        disease_name_lower = standard_name.lower().strip()
                        disease_name_to_ids[disease_name_lower].add(disease_id)
                        
                        # store disease description
                        if description and description != "No disease description found.":
                            disease_descriptions[disease_id] = description
                        
                        # store disease type
                        if disease_type:
                            disease_types[disease_id] = disease_type
                        
                        # synonyms: add standard name to disease_mapping_with_synonyms even if no synonyms
                        if synonyms_str:
                            synonyms = [s.strip() for s in synonyms_str.split(';') if s.strip()]
                            disease_mapping_with_synonyms[disease_id] = [standard_name] + synonyms
                        else:
                            # include standard name even when no synonyms
                            disease_mapping_with_synonyms[disease_id] = [standard_name]
        
        # 2. load disease to public ID mapping
        disease_to_public_file = os.path.join(kg_dir, "disease_to_publicDisease_edges.csv")
        if os.path.exists(disease_to_public_file):
            with open(disease_to_public_file, 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    aggregated_id = row['sourceID'].strip()
                    public_id = row['targetID'].strip()
                    if aggregated_id and public_id:
                        if aggregated_id not in disease_id_mapping:
                            disease_id_mapping[aggregated_id] = []
                        disease_id_mapping[aggregated_id].append(public_id)
        
        # 3. load disease-phenotype edges (with frequency)
        disease_phenotype_file = os.path.join(kg_dir, "disease_to_phenotype_edges.csv")
        if os.path.exists(disease_phenotype_file):
            with open(disease_phenotype_file, 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    disease_id = row['sourceID'].strip()
                    phenotype_id = row['targetID'].strip()
                    frequency = row.get('frequency', '').strip()
                    frequency_max = row.get('frequency_max', '').strip()
                    
                    if disease_id and phenotype_id:
                        # handle aggregated_id to public_id mapping
                        public_ids = disease_id_mapping.get(disease_id, [disease_id])
                        
                        for public_id in public_ids:
                            phenotype_to_diseases[phenotype_id].add(public_id)
                            disease_to_phenotypes[public_id].add(phenotype_id)
                            
                            # store frequency info
                            freq_value = None
                            if frequency_max:
                                try:
                                    freq_value = float(frequency_max)
                                except:
                                    pass
                            
                            if freq_value is not None:
                                phenotype_disease_frequency[(phenotype_id, public_id)] = {
                                    'frequency': frequency_max,
                                    'frequency_type': 'numeric'
                                }
                        
                        # update count (once per aggregated_id)
                        disease_phenotype_counts[disease_id] += 1
        
        # 4. load phenotype nodes (IC and embedding)
        phenotype_nodes_file = os.path.join(kg_dir, "phenotype_nodes.csv")
        if os.path.exists(phenotype_nodes_file):
            with open(phenotype_nodes_file, 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    phenotype_id = row['ID'].strip()
                    ic_str = row.get('IC', '').strip()
                    embedding_str = row.get('embedding', '').strip()
                    
                    if phenotype_id:
                        # load IC values
                        if ic_str:
                            try:
                                ic_dict[phenotype_id] = float(ic_str)
                            except:
                                pass
                        
                        # load embedding
                        if embedding_str:
                            try:
                                embedding = json.loads(embedding_str)
                                if isinstance(embedding, list):
                                    phe2embedding[phenotype_id] = np.array(embedding, dtype=np.float32)
                            except:
                                pass
        
        # 5. load IC dict if exists
        ic_dict_file = os.path.join(kg_dir, "ic_dict_recomputed.json")
        if os.path.exists(ic_dict_file):
            with open(ic_dict_file, 'r', encoding='utf-8') as f:
                ic_dict_from_file = json.load(f)
                # merge IC (file values take precedence)
                for k, v in ic_dict_from_file.items():
                    try:
                        ic_dict[k] = float(v)
                    except:
                        pass
        
        # 6. load phenotype hierarchy (is_a)
        phenotype_phenotype_file = os.path.join(kg_dir, "phenotype_to_phenotype_edges.csv")
        if os.path.exists(phenotype_phenotype_file):
            with open(phenotype_phenotype_file, 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    source_id = row['sourceID'].strip()
                    target_id = row['targetID'].strip()
                    relationship = row.get('relationship', '').strip()
                    
                    if source_id and target_id and relationship == 'is_a':
                        # source_id is_a target_id: target is parent of source
                        hpo_is_a[source_id].append(target_id)
                        parent_to_children[target_id].append(source_id)
        
        # build reverse: public_id -> aggregated_id
        public_to_aggregated = {}
        for aggregated_id, public_ids in disease_id_mapping.items():
            for public_id in public_ids:
                public_to_aggregated[public_id] = aggregated_id
        
        # cache data
        cache = {
            'phenotype_to_diseases': phenotype_to_diseases,
            'disease_to_phenotypes': disease_to_phenotypes,
            'disease_phenotype_counts': disease_phenotype_counts,
            'hpo_is_a': hpo_is_a,
            'parent_to_children': parent_to_children,
            'ic_dict': ic_dict,
            'disease_names': disease_names,
            'disease_name_to_ids': disease_name_to_ids,
            'disease_mapping_with_synonyms': disease_mapping_with_synonyms,
            'disease_descriptions': disease_descriptions,
            'disease_types': disease_types,
            'phenotype_disease_frequency': phenotype_disease_frequency,
            'phe2embedding': phe2embedding,
            'disease_id_mapping': disease_id_mapping,  # aggregated_id -> list of public_ids
            'public_to_aggregated': public_to_aggregated  # public_id -> aggregated_id
        }
        
        # mark loaded and cache
        cls._kg_data_loaded = True
        cls._kg_data_cache = cache
        
        return cache
    
    def __init__(self, kg_data: Dict):
        """
        Initialize helper with KG data.
        
        Args:
            kg_data: Dictionary containing all KG data structures
        """
        self.hpo_is_a = kg_data['hpo_is_a']
        self.parent_to_children = kg_data['parent_to_children']
        self.ic_dict = kg_data['ic_dict']
        self.disease_names = kg_data['disease_names']
        self.disease_name_to_ids = kg_data['disease_name_to_ids']
        self.disease_mapping_with_synonyms = kg_data['disease_mapping_with_synonyms']
        self.disease_descriptions = kg_data['disease_descriptions']
        self.disease_types = kg_data['disease_types']
        self.phenotype_disease_frequency = kg_data['phenotype_disease_frequency']
        self.phe2embedding = kg_data['phe2embedding']
        self.disease_to_phenotypes = kg_data['disease_to_phenotypes']
        self.public_to_aggregated = kg_data.get('public_to_aggregated', {})
        self.disease_id_mapping = kg_data.get('disease_id_mapping', {})
    
    def get_all_parent_phenotypes(self, phenotype_id: str, visited: set = None, max_depth: int = 100) -> set:
        """Get all parent phenotypes recursively"""
        if visited is None:
            visited = set()
        if phenotype_id in visited or max_depth <= 0:
            return set()
        visited.add(phenotype_id)
        all_parents = set()
        direct_parents = self.hpo_is_a.get(phenotype_id, [])
        all_parents.update(direct_parents)
        for parent in direct_parents:
            all_parents.update(self.get_all_parent_phenotypes(parent, visited.copy(), max_depth - 1))
        return all_parents
    
    def get_disease_name(self, disease_id: str) -> str:
        """Get disease name from disease ID"""
        return self.disease_names.get(disease_id, disease_id)
    
    def get_disease_all_names(self, disease_id: str) -> List[str]:
        """Get all names (including synonyms) for a disease from disease ID"""
        synonyms = self.disease_mapping_with_synonyms.get(disease_id, [])
        if not synonyms and disease_id in self.disease_names:
            return [self.disease_names[disease_id]]
        # dedup, keep order (case-insensitive)
        seen_lower = {}
        result = []
        for name in synonyms:
            name_lower = name.lower()
            if name_lower not in seen_lower:
                seen_lower[name_lower] = name
                result.append(name)
        return result
    
    def get_frequency_info(self, phenotype_id: str, disease_id: str) -> dict:
        """Get frequency information for a specific phenotype-disease pair"""
        return self.phenotype_disease_frequency.get((phenotype_id, disease_id))
    
    def is_hp_with_only_child(self, parent_phenotype: str) -> str:
        """Check if a parent phenotype has only one child, return the child if so"""
        children = self.parent_to_children.get(parent_phenotype, [])
        return children[0] if len(children) == 1 else None
    
    def calculate_case_similarity(self, patient_phenotypes: List[str], disease_id: str) -> float:
        """Calculate cosine similarity between patient phenotypes and a specific disease using phenotype embeddings"""
        if not self.phe2embedding or not self.ic_dict:
            return 0.0
        
        # weighted embedding for patient phenotypes
        patient_embeddings = []
        patient_ic_values = []
        for phe in patient_phenotypes:
            if phe in self.phe2embedding and phe in self.ic_dict:
                patient_embeddings.append(self.phe2embedding[phe])
                patient_ic_values.append(self.ic_dict[phe])
        
        if not patient_embeddings:
            return 0.0
        
        patient_embeddings = np.array(patient_embeddings)
        patient_ic_values = np.array(patient_ic_values)
        
        # normalize IC as weight
        if patient_ic_values.sum() > 0:
            weights = patient_ic_values / patient_ic_values.sum()
        else:
            weights = np.ones(len(patient_ic_values)) / len(patient_ic_values)
        
        patient_embedding = np.average(patient_embeddings, axis=0, weights=weights)
        patient_embedding = patient_embedding / (np.linalg.norm(patient_embedding) + 1e-10)
        
        # weighted embedding for disease phenotypes
        disease_phenotypes = self.disease_to_phenotypes.get(disease_id, set())
        disease_embeddings = []
        disease_ic_values = []
        for phe in disease_phenotypes:
            if phe in self.phe2embedding and phe in self.ic_dict:
                disease_embeddings.append(self.phe2embedding[phe])
                disease_ic_values.append(self.ic_dict[phe])
        
        if not disease_embeddings:
            return 0.0
        
        disease_embeddings = np.array(disease_embeddings)
        disease_ic_values = np.array(disease_ic_values)
        
        if disease_ic_values.sum() > 0:
            weights = disease_ic_values / disease_ic_values.sum()
        else:
            weights = np.ones(len(disease_ic_values)) / len(disease_ic_values)
        
        disease_embedding = np.average(disease_embeddings, axis=0, weights=weights)
        disease_embedding = disease_embedding / (np.linalg.norm(disease_embedding) + 1e-10)
        
        # compute cosine similarity
        try:
            similarity = float(np.dot(patient_embedding, disease_embedding))
            if not np.isfinite(similarity):
                return 0.0
            return similarity
        except:
            return 0.0
    
    def convert_hpo_frequency_to_description(self, hpo_id: str) -> str:
        """Convert HPO frequency ID to description"""
        hpo_freq_map = {
            'HP:0040285': 'Excluded',
            'HP:0040284': 'Very rare',
            'HP:0040283': 'Occasional',
            'HP:0040282': 'Frequent',
            'HP:0040281': 'Very frequent',
            'HP:0040280': 'Obligate'
        }
        return hpo_freq_map.get(hpo_id, hpo_id)
    
    def get_phenotype_abnormal_category(self, hpo_id: str) -> List[str]:
        """
        Get all phenotype abnormal categories for a given HPO ID.
        Returns all direct children of HP:0000118 (Phenotypic abnormality) that the phenotype belongs to.
        Uses local KG data (hpo_is_a hierarchy).
        
        Args:
            hpo_id (str): The HPO ID to find the categories for
            
        Returns:
            List[str]: List of HPO IDs of the abnormal categories, or empty list if not found
        """
        # Direct children of HP:0000118 (Phenotypic abnormality)
        hp_0000118_subcategories = {
            'HP:0000119',  # Abnormality of the genitourinary system
            'HP:0000152',  # Abnormality of head or neck
            'HP:0000478',  # Abnormality of the eye
            'HP:0000598',  # Abnormality of the ear
            'HP:0000707',  # Abnormality of the nervous system
            'HP:0000769',  # Abnormality of the breast
            'HP:0000818',  # Abnormality of the endocrine system
            'HP:0001197',  # Abnormality of prenatal development or birth
            'HP:0001507',  # Growth abnormality
            'HP:0001574',  # Abnormality of the integument
            'HP:0001608',  # Abnormality of the voice
            'HP:0001626',  # Abnormality of the cardiovascular system
            'HP:0001871',  # Abnormality of blood and blood-forming tissues
            'HP:0001939',  # Abnormality of metabolism/homeostasis
            'HP:0002086',  # Abnormality of the respiratory system
            'HP:0002664',  # Neoplasm
            'HP:0002715',  # Abnormality of the immune system
            'HP:0025031',  # Abnormality of the digestive system
            'HP:0025142',  # Constitutional symptom
            'HP:0025354',  # Abnormal cellular phenotype
            'HP:0033127',  # Abnormality of the musculoskeletal system
            'HP:0040064',  # Abnormality of limbs
            'HP:0045027'   # Abnormality of the thoracic cavity
        }
        
        # Use BFS to traverse up the hierarchy and collect all categories
        queue = [hpo_id]
        visited = set()
        found_categories = []
        
        while queue:
            current_id = queue.pop(0)
            if current_id in visited:
                continue
            visited.add(current_id)
            
            # Check if current phenotype is one of the direct subcategories
            if current_id in hp_0000118_subcategories:
                found_categories.append(current_id)
            
            # Get parent phenotypes (is_a relationships)
            parents = self.hpo_is_a.get(current_id, [])
            for parent_id in parents:
                if parent_id not in visited:
                    queue.append(parent_id)
        
        return found_categories
    
    def get_disease_exp_info_from_local(self, disease_id: str) -> Dict:
        """
        Get expanded disease information from local KG data.
        Similar to get_disease_exp_info_from_kg but uses local data.
        
        Args:
            disease_id: Disease ID (e.g., "OMIM:253800", "ORPHA:123456", or aggregated ID like "D:1")
            
        Returns:
            Dict containing disease information similar to get_disease_exp_info_from_kg format
        """
        # first try disease_id (may be aggregated_id)
        aggregated_id = disease_id
        public_ids = []
        
        # if disease_id is public_id, find aggregated_id
        if disease_id in self.public_to_aggregated:
            aggregated_id = self.public_to_aggregated[disease_id]
            public_ids = self.disease_id_mapping.get(aggregated_id, [disease_id])
        elif disease_id in self.disease_id_mapping:
            # disease_id is aggregated_id
            public_ids = self.disease_id_mapping[disease_id]
        else:
            # neither; try direct lookup
            public_ids = [disease_id]
        
        # get disease info via aggregated_id
        result = {
            'disease_id': public_ids if public_ids else [disease_id],  # return public_id list
            'standard_name': self.disease_names.get(aggregated_id, ""),
            'synonyms': self.disease_mapping_with_synonyms.get(aggregated_id, []),
            'disease_type': self.disease_types.get(aggregated_id, ""),
            'description': self.disease_descriptions.get(aggregated_id, ""),
            'is_rare': '',
            'link': '',
            'aggregated_disease_id': aggregated_id,
            'relationship_type': 'disease_exact' if aggregated_id != disease_id else '',
            'phenotypes': list(self.disease_to_phenotypes.get(public_ids[0] if public_ids else disease_id, set())),
            'phenotype_max_frequencies': []
        }
        
        # get phenotype freq (use public_id)
        target_disease_id = public_ids[0] if public_ids else disease_id
        phenotypes = result['phenotypes']
        phenotype_max_frequencies = []
        for phenotype_id in phenotypes:
            freq_info = self.phenotype_disease_frequency.get((phenotype_id, target_disease_id))
            if freq_info:
                frequency_max = freq_info.get('frequency', '')
                phenotype_max_frequencies.append(frequency_max)
            else:
                phenotype_max_frequencies.append('')
        result['phenotype_max_frequencies'] = phenotype_max_frequencies
        
        return result
    
class PhenotypeToDiseasePromptGenerator:
    def __init__(self, disease_mapping_file: str = None, phenotype_hpoa_file: str = None, 
                 phenotype_to_genes_file: str = None, genes_to_phenotype_file: str = None,
                 genes_to_disease_file: str = None,
                 embedding_file: str = None, ic_file: str = None, case_library: str = None,
                 disease_descriptions_file: str = None,
                 sentence_transformer_model: str = "FremyCompany/BioLORD-2023", config: Dict = None):
        """Initialize Phenotype to Disease Prompt Generator"""
        self.disease_mapping_file = disease_mapping_file
        self.phenotype_hpoa_file = phenotype_hpoa_file
        self.phenotype_to_genes_file = phenotype_to_genes_file
        self.genes_to_phenotype_file = genes_to_phenotype_file
        self.genes_to_disease_file = genes_to_disease_file
        self.embedding_file = embedding_file
        self.ic_file = ic_file
        self.case_library = case_library
        self.disease_descriptions_file = disease_descriptions_file
        self.config = config or {}
        
        # Initialize disease descriptions dictionary
        self.disease_descriptions = {}
        self.rare_disease_types = {}
        
        # Initialize Qwen model for semantic comparison
        self.qwen_model = None
        self.qwen_tokenizer = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.semantic_comparison_cache = {}  # Cache for semantic comparison results
        # self._init_qwen_model() # TODO: optimize, current similarity method not ideal
        
        # Load sentence transformer for semantic similarity
        print(f"Loading sentence transformer model: {sentence_transformer_model}")
        # Set cache directory for sentence transformer models from config
        cache_dir = self.config.get("sentence_transformer_cache_dir", "../../model_weight")
        cache_dir = os.path.abspath(cache_dir)
        print(f"Sentence transformer cache directory: {cache_dir}")
        self.sentence_model = SentenceTransformer(sentence_transformer_model, cache_folder=cache_dir)
        
        print(f"Case extraction files:")
        print(f"  Embedding file: {embedding_file}")
        print(f"  IC file: {ic_file}")
        print(f"  Case library: {case_library}")
        
        # Load embedding and IC data if files are provided
        self.phe2embedding = {}
        self.ic_dict = {}
        self.external_samples = []
        self._cached_sample_matrices = []
        self._cached_sample_embeddings = []
        
        if embedding_file and ic_file:
            print("Loading embedding data...")
            self._load_embedding_data()
        
        if case_library:
            print("Loading case library...")
            self._load_external_samples()
        
        print(f"Initialization complete:")
        print(f"  Embedding data loaded: {len(self.phe2embedding)} phenotypes")
        print(f"  IC data loaded: {len(self.ic_dict)} phenotypes")
        print(f"  External samples loaded: {len(self.external_samples)} samples")
        
        self.disease_names = {}
        self.phenotype_names = {}
        self.hpo_frequency_descriptions = {}  # HPO ID -> frequency description mapping
        self.hpo_frequency_description_to_id = {
            'Excluded': 'HP:0040285',
            'Very rare': 'HP:0040284',
            'Rare': 'HP:0040284',
            'Occasional': 'HP:0040283',
            'Frequent': 'HP:0040282',
            'Very frequent': 'HP:0040281',
            'Obligate': 'HP:0040280',
        }

        self.hpo2freq_dict = None # HPO ID -> frequency weight list [low, medium, high]
        self.word2freq_dict = None # word -> frequency weight
        # Add new data structures for HPO synonyms, definitions, and comments
        self.hpo_synonyms = {}  # HPO ID -> list of synonyms
        self.hpo_definitions = {}  # HPO ID -> definition
        self.hpo_comments = {}  # HPO ID -> comment
        self.hpo_is_a = {}  # HPO ID -> list of parent terms (is_a relationships)
        self.hpo_alt_ids = {}  # primary ID -> list of alt_ids
        
        # Add disease synonyms data structure (similar to phenotype_to_disease_prediction_bysteps.py)
        self.disease_mapping_with_synonyms = {}  # disease_id -> list of disease names/synonyms

        self.phenotype_to_diseases = defaultdict(set)
        # Add new data structures for disease ranking
        self.disease_to_phenotypes = defaultdict(set)  # disease_id -> set of phenotype_ids
        self.disease_phenotype_counts = defaultdict(int)  # disease_id -> count of phenotype associations

        # Add data structures for phenotype-disease mappings from gene files
        self.phenotype_to_diseases_from_genes = defaultdict(set)  # phenotype_id -> set of disease_ids
        self.disease_to_phenotypes_from_genes = defaultdict(set)  # disease_id -> set of phenotype_ids
        
        # Add data structures for gene-phenotype-disease mappings
        # (disease_id, phenotype_id) -> set of (ncbi_gene_id, gene_symbol)
        self.disease_phenotype_to_genes = defaultdict(set)  # (disease_id, phenotype_id) -> set of (ncbi_gene_id, gene_symbol)
        self.phenotype_to_genes = defaultdict(set)  # phenotype_id -> set of (ncbi_gene_id, gene_symbol)
        self.gene_to_phenotypes = defaultdict(set)  # (ncbi_gene_id, gene_symbol) -> set of phenotype_ids
        
        # Add data structures for gene-disease mappings
        self.gene_to_diseases = defaultdict(set)  # (ncbi_gene_id, gene_symbol) -> set of disease_ids
        self.disease_to_genes = defaultdict(set)  # disease_id -> set of (ncbi_gene_id, gene_symbol)

        # Add data structures for phenotype-disease mappings from trueSamples.jsonl
        self.phenotype_to_diseases_from_trueSamples = defaultdict(set)  # phenotype_id -> set of disease_ids
        self.disease_to_phenotypes_from_trueSamples = defaultdict(set)  # disease_id -> set of phenotype_ids

        # Add data structures for frequency annotations
        self.phenotype_disease_frequency = {}  # (phenotype_id, disease_id) -> frequency_info
        self.disease_phenotype_frequency = defaultdict(dict)  # disease_id -> {phenotype_id: frequency_info}
        # Add new data structure for disease name to disease IDs mapping (case-insensitive)
        self.disease_name_to_ids = defaultdict(set)  # disease_name_lower -> set of disease_ids
        
        # Add data structure for disease type mapping
        self.disease_types = {}  # disease_id -> disease_type/category
        
        # Initialize dictionary to store OMIM disease IDs from phenotype.hpoa
        self.rare_disease_in_hpoa = set()  # Set of OMIM disease IDs found in phenotype.hpoa
        
        # Initialize LR ranking engine (lazy loading)
        self.lr_ranking_engine = None
        
        # Initialize knowledge graph query client
        kg_api_url = self.config.get("kg_api_url", "http://192.168.0.9:5008/nebulasearch/")
        kg_api_space = self.config.get("kg_api_space", "RDPkg")
        kg_api_timeout = self.config.get("kg_api_timeout", 30)
        self.kg_query = KnowledgeGraphQuery(api_url=kg_api_url, space=kg_api_space, timeout=kg_api_timeout)
        
        # Initialize local KG data cache (lazy loading, only loads once per instance)
        self._local_kg_data_cache = None
        self._local_kg_data_loaded = False
        
        
        self.load_disease_names()
        # Note: phenotype_disease_mappings will be loaded after OBO file is loaded
        if phenotype_to_genes_file:
            self.load_phenotype_to_diseases_from_genes_file()
        if genes_to_phenotype_file:
            self.load_genes_to_phenotype_diseases_file()
        if genes_to_disease_file:
            self.load_genes_to_disease_file()
        
        # # Load disease-phenotype mapping from expanded case library and merge into self.disease_to_phenotypes
        # expanded_library_path = self.config.get("expanded_case_library_file")
        # if expanded_library_path:
        #     expanded_mapping = load_disease_phenotype_mapping(expanded_library_path)
        #     # Merge into self.disease_to_phenotypes (merge phenotypes for each disease)
        #     for disease_id, phenotypes in expanded_mapping.items():
        #         self.disease_to_phenotypes[disease_id].update(phenotypes)
        #         # Also update phenotype_to_diseases for reverse mapping
        #         for phenotype_id in phenotypes:
        #             self.phenotype_to_diseases[phenotype_id].add(disease_id)
        
    def _init_qwen_model(self):
        """Initialize Qwen model for semantic comparison"""
        try:
            # Check if model exists in HuggingFace cache format
            cache_dir = os.path.abspath("../../model_weight")
            print(f"Model cache directory: {cache_dir}")

            model_name = "Qwen/Qwen3-8B"
            text_model_dir = os.path.join(cache_dir, f"models--{model_name.replace('/', '--')}")
            print(f"Checking for cached model at: {text_model_dir}")
            
            if os.path.exists(text_model_dir):
                print(f"Found cached model at: {text_model_dir}")
                # Use the original model name, transformers will find it in cache
                text_model_to_use = model_name
            else:
                print(f"Cached model not found at: {text_model_dir}")
                print(f"Will download model to: {cache_dir}")
                text_model_to_use = model_name

            self.qwen_tokenizer = AutoTokenizer.from_pretrained(
                text_model_to_use, 
                trust_remote_code=True,
                cache_dir=cache_dir
            )
            self.qwen_model = AutoModelForCausalLM.from_pretrained(
                text_model_to_use, 
                cache_dir=cache_dir,
                torch_dtype=torch.float16,
                device_map="auto",
                trust_remote_code=True
            )
            print("Qwen model loaded successfully")
        except Exception as e:
            print(f"Warning: Failed to load Qwen model: {e}")
            self.qwen_model = None
            self.qwen_tokenizer = None

    # TODO: optimize, current similarity method not ideal
    def _check_semantic_similar_by_embedding(self, phenotype: str, phenotype_with_parent: str) -> bool:   
        """Check if two phenotypes have similar semantic meanings using embedding method"""
        if not self.phe2embedding or not self.ic_dict or not self.external_samples:
            return True  # If model not available, don't filter
        
        # Calculate current sample embedding
        try:
            phenotype_embedding = self.phe2embedding[self._extract_hp_code(phenotype)]
            phenotype_with_parent_embedding = self.phe2embedding[self._extract_hp_code(phenotype_with_parent)]
        except Exception as e:
            return True  # If embedding not available, don't filter

        cosine_similarity = np.dot(phenotype_embedding, phenotype_with_parent_embedding) / (np.linalg.norm(phenotype_embedding) * np.linalg.norm(phenotype_with_parent_embedding))
        
        if cosine_similarity > 0.6:
            print(f"phenotype {self.phenotype_names[phenotype]} and phenotype {self.phenotype_names[phenotype_with_parent]} are similar in meaning")


        return cosine_similarity > 0.6

    # TODO: optimize, current similarity method not ideal
    def _check_semantic_opposite(self, phenotype: str, phenotype_with_parent: str) -> bool:
        """Check if two phenotypes have opposite semantic meanings using Qwen model"""
        if self.qwen_model is None or self.qwen_tokenizer is None:
            return False  # If model not available, don't filter 
        
        # Create cache key for this comparison
        cache_key = tuple(sorted([phenotype, phenotype_with_parent]))
        
        # Check if result is already cached
        if cache_key in self.semantic_comparison_cache:
            return self.semantic_comparison_cache[cache_key]
        
        try:
            print(f"Checking if phenotype {self.phenotype_names[phenotype]} and phenotype {self.phenotype_names[phenotype_with_parent]} are opposite in meaning")
            # Create prompt for semantic comparison
            prompt = f"""
Compare the semantic meanings of these two medical phenotypes and determine if they are opposite in meaning. Only answer "Yes" or "No". 
1. {phenotype} 
2. {phenotype_with_parent} 
Are these phenotypes semantically opposite? Answer only "Yes" or "No":
"""
            # Prepare messages
            messages = [{"role": "user", "content": prompt}]
            
            # Apply chat template
            text = self.qwen_tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False
            )
            
            # Tokenize input
            model_inputs = self.qwen_tokenizer([text], return_tensors="pt").to(self.device)
            
            # Generate response
            with torch.no_grad():
                generated_ids = self.qwen_model.generate(
                    **model_inputs,
                    max_new_tokens=10,  # Short response expected
                    temperature=0.6,    # Low temperature for consistent output
                    top_p=0.95,
                    top_k=20,
                    do_sample=True,
                    pad_token_id=self.qwen_tokenizer.eos_token_id,
                    eos_token_id=self.qwen_tokenizer.eos_token_id
                )
            
            # Decode output
            output_ids = generated_ids[0][len(model_inputs.input_ids[0]):].tolist()
            if 151668 in output_ids:
                index = len(output_ids) - output_ids[::-1].index(151668)
            else:
                index = 0
            response = self.qwen_tokenizer.decode(output_ids[index:], skip_special_tokens=True).strip("\n")
            
            # Extract Yes/No from response
            print(f"Response: {response}")
            response_lower = response.lower()
            if "yes" in response_lower:
                result = True
            elif "no" in response_lower:
                result = False
            else:
                # If unclear response, default to similar
                result = False
            
            # Cache the result
            self.semantic_comparison_cache[cache_key] = result
            return result
                
        except Exception as e:
            print(f"Error in semantic comparison: {e}")
            return False  # Default to similar if error occurs

    def _load_embedding_data(self):
        """Load embedding and IC data"""
        print("Loading embedding and IC data...")
        
        try:
            # Try to find embedding file
            embedding_path = find_file_path([self.embedding_file] if self.embedding_file else [])
            if embedding_path:
                with open(embedding_path, 'r', encoding='utf-8-sig') as f:
                    self.phe2embedding = json.load(f)
                print(f"Loaded phenotype embeddings from: {embedding_path}")
            else:
                print(f"Error: Embedding file not found: {self.embedding_file}")
                print("Please provide a valid embedding file path using --embedding_file")
                raise FileNotFoundError(f"Embedding file not found: {self.embedding_file}")
                
            # Try to find IC file
            ic_path = find_file_path([self.ic_file] if self.ic_file else [])
            if ic_path:
                with open(ic_path, 'r', encoding='utf-8-sig') as f:
                    self.ic_dict = json.load(f)
                print(f"Loaded IC dictionary from: {ic_path}")
            else:
                print(f"Error: IC file not found: {self.ic_file}")
                print("Please provide a valid IC file path using --ic_file")
                raise FileNotFoundError(f"IC file not found: {self.ic_file}")
                
        except Exception as e:
            print(f"Error loading embedding data: {e}")
            raise
    
    def _load_trueSamples(self, exclude_sample_id: int = None):
        """Load trueSamples data from JSONL file"""
        
        # Load trueSamples data if needed
        if not exclude_sample_id:
            print(f"Loading trueSamples data from {self.case_library}...")
        
        # read trueSamples.jsonl, load phenotype-disease associations
        trueSamples_file = self.case_library
        with open(trueSamples_file, 'r', encoding='utf-8') as f:
            for line in f:
                data = json.loads(line)
                phenotype_ids = data['Phenotype']
                disease_ids = data['RareDisease']
                source = data['Department']
                
                if exclude_sample_id and source.split('_')[-1] == str(exclude_sample_id):
                    # print(f"source: {source}, exclude_sample_id: {exclude_sample_id}")
                    continue

                for phenotype_id in phenotype_ids:
                    if phenotype_id not in self.phenotype_to_diseases_from_trueSamples:
                        self.phenotype_to_diseases_from_trueSamples[phenotype_id] = set()
                    self.phenotype_to_diseases_from_trueSamples[phenotype_id].update(disease_ids)

                for disease_id in disease_ids:
                    if disease_id not in self.disease_to_phenotypes_from_trueSamples:
                        self.disease_to_phenotypes_from_trueSamples[disease_id] = set()
                    self.disease_to_phenotypes_from_trueSamples[disease_id].update(phenotype_ids)
    
    def _load_external_samples(self):
        """Load external samples from JSONL file"""
        """Load external case DB from file (JSONL)."""
        print(f"Loading case library from: {self.case_library}")
        
        # Try to find case library file
        samples_path = find_file_path([self.case_library] if self.case_library else [])
        if not samples_path:
            print(f"Error: Case library file not found: {self.case_library}")
            print("Please provide a valid case library path using --case_library")
            return
            # raise FileNotFoundError(f"Case library file not found: {self.case_library}")
        
        try:
            with open(samples_path, 'r', encoding='utf-8') as f:
                for line_num, line in enumerate(f, 1):
                    if line.strip():
                        try:
                            sample = json.loads(line.strip())
                            self.external_samples.append(sample)
                        except json.JSONDecodeError as e:
                            print(f"Warning: Invalid JSON at line {line_num}: {e}")
                            continue
            
            print(f"Loaded {len(self.external_samples)} external samples from: {samples_path}")
            
        except Exception as e:
            print(f"Error loading external samples: {e}")
            raise

        # Prepare cached embedding matrices for external samples
        self._prepare_external_sample_cache()


    # get info from KG
    # function: given hpo_id, get phenotype info from KG; include:
    # standard_name，synonyms，description，comment，associations，IC，embedding
    def get_phenotype_info_from_kg(self, hpo_id: str) -> Dict:
        """
        Get phenotype information from knowledge graph by HPO ID.
        Only queries KG, does not use local data.
        
        Args:
            hpo_id: HPO ID (e.g., "HP:0000001")
            
        Returns:
            Dict containing:
                - standard_name: Standard name of the phenotype
                - synonyms: List of synonyms (parsed from semicolon-separated string)
                - description: Description of the phenotype
                - comment: Comment about the phenotype
                - associations: Number of associations
                - IC: Information Content value
                - embedding: Embedding vector (as string)
        """
        result = {
            'standard_name': '',
            'synonyms': [],
            'description': '',
            'comment': '',
            'associations': 0,
            'IC': 0.0,
            'embedding': []
        }
        
        if not hpo_id:
            return result
        
        try:
            # Query knowledge graph for phenotype information using HPO ID
            kg_result = self.kg_query.search(
                search_type="simplesearch",
                ent1="phenotype",
                prop1="hpo_id",
                label1=hpo_id,
                use_cache=True
            )
            
            if kg_result is None or kg_result.get('hasError', False):
                return result
            
            # Parse the result according to the API response format
            phenotypes = []
            
            if isinstance(kg_result, dict) and 'data' in kg_result:
                data = kg_result['data']
                if isinstance(data, dict) and 'results' in data:
                    results_str = data['results']
                    if isinstance(results_str, str):
                        try:
                            phenotypes = json.loads(results_str)
                        except json.JSONDecodeError as e:
                            print(f"Error parsing JSON string from results for {hpo_id}: {e}")
                            return result
                    elif isinstance(results_str, list):
                        phenotypes = results_str
            
            # Extract information from the first matching phenotype
            if phenotypes and len(phenotypes) > 0:
                item = phenotypes[0]
                if isinstance(item, dict) and 'n' in item:
                    node = item['n']
                    if isinstance(node, dict) and 'properties' in node:
                        properties = node['properties']
                        
                        # Extract standard_name
                        if 'standard_name' in properties:
                            result['standard_name'] = str(properties['standard_name']).strip()
                        
                        # Extract synonyms (semicolon-separated string)
                        if 'synonyms' in properties:
                            synonyms = properties['synonyms']
                            if isinstance(synonyms, str):
                                if synonyms.strip():
                                    result['synonyms'] = [s.strip() for s in synonyms.split(';') if s.strip()]
                            elif isinstance(synonyms, list):
                                result['synonyms'] = [str(s).strip() for s in synonyms if s]
                        
                        # Extract description
                        if 'description' in properties:
                            result['description'] = str(properties['description']).strip()
                        
                        # Extract comment
                        if 'comment' in properties:
                            result['comment'] = str(properties['comment']).strip()
                        
                        # Extract associations
                        if 'associations' in properties:
                            try:
                                result['associations'] = int(properties['associations'])
                            except (ValueError, TypeError):
                                result['associations'] = 0
                        
                        # Extract IC (Information Content)
                        if 'IC' in properties:
                            try:
                                result['IC'] = float(properties['IC'])
                            except (ValueError, TypeError):
                                result['IC'] = 0.0
                        
                        # Extract embedding (parse JSON string to list)
                        if 'embedding' in properties:
                            embedding = properties['embedding']
                            if isinstance(embedding, str):
                                try:
                                    result['embedding'] = json.loads(embedding)
                                except json.JSONDecodeError:
                                    result['embedding'] = []
                            elif isinstance(embedding, list):
                                result['embedding'] = embedding
                            else:
                                result['embedding'] = []
            
        except Exception as e:
            print(f"Error getting phenotype info from KG for {hpo_id}: {e}")
            import traceback
            traceback.print_exc()
        
        return result

    def get_phenotype_ancestors_from_kg(self, hpo_id: str, max_depth: int = None) -> List[str]:
        """
        Get ancestors of a phenotype from knowledge graph by HPO ID.
        Only queries KG, does not use local data.
        
        Args:
            hpo_id: HPO ID (e.g., "HP:0000001")
            max_depth: Maximum depth to traverse. If None, gets all ancestors (default: None)
            
        Returns:
            List of ancestor HPO IDs
        """
        if not hpo_id:
            return []
        
        ancestors = set()  # Use set to avoid duplicates
        visited = set()  # Track visited nodes to avoid cycles
        
        def get_direct_parents(current_hpo_id: str, current_depth: int):
            """Recursively get parent phenotypes up to max_depth (or all if max_depth is None)"""
            # Check depth limit if max_depth is specified
            if max_depth is not None and current_depth > max_depth:
                return
            
            # Avoid cycles
            if current_hpo_id in visited:
                return
            
            visited.add(current_hpo_id)
            
            try:
                # Query knowledge graph for phenotype relationships
                # Using relsearch to find relationships where current_hpo_id is the destination
                kg_result = self.kg_query.search(
                    search_type="relsearch",
                    ent1="phenotype",
                    prop1="hpo_id",
                    label1=current_hpo_id,
                    ent2="phenotype",
                    prop2="",
                    label2="",
                    use_cache=True
                )
                
                if kg_result is None or kg_result.get('hasError', False):
                    return
                
                # print(kg_result)
                # Parse the result according to the API response format
                # Format: {'data': {'results': '[{"n": {...}, "r": {...}, "m": {...}}]'}, ...}
                relationships = []
                
                if isinstance(kg_result, dict) and 'data' in kg_result:
                    data = kg_result['data']
                    if isinstance(data, dict) and 'results' in data:
                        results_str = data['results']
                        if isinstance(results_str, str):
                            try:
                                relationships = json.loads(results_str)
                            except json.JSONDecodeError as e:
                                print(f"Error parsing JSON string from results for {current_hpo_id}: {e}")
                                return
                        elif isinstance(results_str, list):
                            relationships = results_str
                
                # Extract parent phenotypes from relationships
                # Filter for is_a relationships where current_hpo_id is the source (src)
                # The parent phenotype is in dst
                for item in relationships:
                    if not isinstance(item, dict):
                        continue
                    
                    # Get relationship information
                    relationship = item.get('r', {})
                    if not isinstance(relationship, dict):
                        continue
                    
                    # Check if relationship type is "is_a" and source matches current_hpo_id
                    # src is the current query id, dst is its parent phenotype id
                    rel_type = relationship.get('type', '')
                    rel_src = relationship.get('src', '')
                    
                    if rel_type == 'is_a' and rel_src == current_hpo_id:
                        # Get parent HPO ID from destination (dst)
                        parent_hpo_id = relationship.get('dst', '')
                        
                        # Also check the target node (m) for the parent HPO ID
                        if not parent_hpo_id:
                            target_node = item.get('m', {})
                            if isinstance(target_node, dict):
                                parent_hpo_id = target_node.get('vid', '')
                        
                        if parent_hpo_id and parent_hpo_id.startswith('HP:'):
                            ancestors.add(parent_hpo_id)
                            
                            # Recursively get parents of this parent
                            # If max_depth is None, continue until no more parents
                            # If max_depth is set, only continue if depth allows
                            if max_depth is None or current_depth < max_depth:
                                get_direct_parents(parent_hpo_id, current_depth + 1)
            
            except Exception as e:
                print(f"Error getting phenotype ancestors from KG for {current_hpo_id}: {e}")
                import traceback
                traceback.print_exc()
        
        # Start recursive traversal
        get_direct_parents(hpo_id, 1)
        
        return list(ancestors)

    def get_phenotype_descendants_from_kg(self, hpo_id: str, max_depth: int = None) -> List[str]:
        """
        Get descendants (children) of a phenotype from knowledge graph by HPO ID.
        Only queries KG, does not use local data.
        
        Args:
            hpo_id: HPO ID (e.g., "HP:0000001")
            max_depth: Maximum depth to traverse. If None, gets all descendants (default: None)
            
        Returns:
            List of descendant HPO IDs
        """
        if not hpo_id:
            return []
        
        descendants = set()  # Use set to avoid duplicates
        visited = set()  # Track visited nodes to avoid cycles
        
        def get_direct_children(current_hpo_id: str, current_depth: int):
            """Recursively get child phenotypes up to max_depth (or all if max_depth is None)"""
            # Check depth limit if max_depth is specified
            if max_depth is not None and current_depth > max_depth:
                return
            
            # Avoid cycles
            if current_hpo_id in visited:
                return
            
            visited.add(current_hpo_id)
            
            try:
                # Query knowledge graph for phenotype relationships
                # Using relsearch to find relationships where current_hpo_id is involved
                kg_result = self.kg_query.search(
                    search_type="relsearch",
                    ent1="phenotype",
                    prop1="hpo_id",
                    label1=current_hpo_id,
                    ent2="phenotype",
                    prop2="",
                    label2="",
                    use_cache=True
                )
                
                # if kg_result is None or kg_result.get('hasError', False):
                if kg_result is None:
                    return
                
                # Parse the result according to the API response format
                # Format: {'data': {'results': '[{"n": {...}, "r": {...}, "m": {...}}]'}, ...}
                relationships = []
                
                if isinstance(kg_result, dict) and 'data' in kg_result:
                    data = kg_result['data']
                    if isinstance(data, dict) and 'results' in data:
                        results_str = data['results']
                        if isinstance(results_str, str):
                            try:
                                relationships = json.loads(results_str)
                            except json.JSONDecodeError as e:
                                print(f"Error parsing JSON string from results for {current_hpo_id}: {e}")
                                return
                        elif isinstance(results_str, list):
                            relationships = results_str
                
                # Extract child phenotypes from relationships
                # Filter for is_a relationships where current_hpo_id is the destination (dst)
                # The child phenotype is in src
                for item in relationships:
                    if not isinstance(item, dict):
                        continue
                    
                    # Get relationship information
                    relationship = item.get('r', {})
                    if not isinstance(relationship, dict):
                        continue
                    
                    # Check if relationship type is "is_a" and destination matches current_hpo_id
                    # dst is the current query id, src is its child phenotype id
                    rel_type = relationship.get('type', '')
                    rel_dst = relationship.get('dst', '')
                    
                    if rel_type == 'is_a' and rel_dst == current_hpo_id:
                        # Get child HPO ID from source (src)
                        child_hpo_id = relationship.get('src', '')
                        
                        # Also check the source node (n) for the child HPO ID
                        if not child_hpo_id:
                            source_node = item.get('n', {})
                            if isinstance(source_node, dict):
                                child_hpo_id = source_node.get('vid', '')
                        
                        if child_hpo_id and child_hpo_id.startswith('HP:'):
                            descendants.add(child_hpo_id)
                            
                            # Recursively get children of this child
                            # If max_depth is None, continue until no more children
                            # If max_depth is set, only continue if depth allows
                            if max_depth is None or current_depth < max_depth:
                                get_direct_children(child_hpo_id, current_depth + 1)
            
            except Exception as e:
                print(f"Error getting phenotype descendants from KG for {current_hpo_id}: {e}")
                import traceback
                traceback.print_exc()
        
        # Start recursive traversal
        get_direct_children(hpo_id, 1)
        
        return list(descendants)

    def get_disease_info_from_kg(self, disease_id: str) -> Dict:
        """
        Get disease information from knowledge graph by disease ID.
        Only queries KG, does not use local data.
        
        Args:
            disease_id: Disease ID (e.g., "OMIM:253800", "ORPHA:123456")
            
        Returns:
            Dict containing:
                - disease_id: Disease ID
                - standard_name: Standard name of the disease
                - link: Link to the disease information
        """
        result = {
            'disease_id': '',
            'standard_name': '',
            'link': ''
        }
        
        if not disease_id:
            return result
        
        try:
            # Query knowledge graph for disease information using disease ID
            kg_result = self.kg_query.search(
                search_type="simplesearch",
                ent1="publicDisease",
                prop1="disease_id",
                label1=disease_id,
                use_cache=True
            )
            
            if kg_result is None or kg_result.get('hasError', False):
                return result
            
            # Parse the result according to the API response format
            # Format: {'data': {'results': '[{"n": {"vid": "...", "properties": {...}}}]'}, ...}
            diseases = []
            
            if isinstance(kg_result, dict) and 'data' in kg_result:
                data = kg_result['data']
                if isinstance(data, dict) and 'results' in data:
                    results_str = data['results']
                    if isinstance(results_str, str):
                        try:
                            diseases = json.loads(results_str)
                        except json.JSONDecodeError as e:
                            print(f"Error parsing JSON string from results for {disease_id}: {e}")
                            return result
                    elif isinstance(results_str, list):
                        diseases = results_str
            
            # Extract information from the first matching disease
            if diseases and len(diseases) > 0:
                item = diseases[0]
                if isinstance(item, dict) and 'n' in item:
                    node = item['n']
                    if isinstance(node, dict) and 'properties' in node:
                        properties = node['properties']
                        
                        # Extract disease_id
                        if 'disease_id' in properties:
                            result['disease_id'] = str(properties['disease_id']).strip()
                        
                        # Extract standard_name
                        if 'standard_name' in properties:
                            result['standard_name'] = str(properties['standard_name']).strip()
                        
                        # Extract link
                        if 'link' in properties:
                            result['link'] = str(properties['link']).strip()
            
        except Exception as e:
            print(f"Error getting disease info from KG for {disease_id}: {e}")
            import traceback
            traceback.print_exc()
        
        return result

    def get_disease_exp_info_from_kg(self, disease_id: str) -> Dict:
        """
        Get expanded disease information from knowledge graph by disease ID using relationship search.
        First tries to get from local KG data if available, otherwise queries online KG.
        
        Args:
            disease_id: Disease ID from publicDisease (e.g., "OMIM:253800", "ORPHA:123456")
            
        Returns:
            Dict containing:
                - disease_id: Disease ID from publicDisease (e.g., "OMIM:253800", "ORPHA:123456")
                - standard_name: Standard name of the disease
                - synonyms: List of synonyms (only extracted if relationship_type is "disease_exact")
                - disease_type: disease_type
                - description: Description of the disease (only extracted if relationship_type is "disease_exact")
                - is_rare: Whether the disease is rare (only extracted if relationship_type is "disease_exact")
                - link: Link to the disease information
                - aggregated_disease_id: The other ID in the relationship (the aggregated disease ID)
                - relationship_type: Type of the relationship (e.g., "disease_exact")
                - phenotypes: List of phenotypes associated with the disease (synced to relationship type)
                - phenotype_max_frequencies: List of corresponding maximum frequency information for phenotypes (synced to phenotypes)
                
        """
        
        result = {
            'disease_id': '',
            'standard_name': '',
            # extended info (when relationship_type is disease_exact)
            'synonyms': [],
            'disease_type':'',
            'description': '',
            'is_rare': '',
            'link': '',
            'aggregated_disease_id': '',  # aggregated disease ID (other end of edge)
            'relationship_type': '',  # relationship type
            'phenotypes': [],  # phenotype list
            'phenotype_max_frequencies': [],  # max freq per phenotype, 1:1 with phenotypes
        }
        
        if not disease_id:
            return result
        
        try:
            # Query knowledge graph using relationship search
            # From publicDisease to disease through relationship network
            kg_result = self.kg_query.search(
                search_type="relsearch",
                ent1="publicDisease",
                prop1="disease_id",
                label1=disease_id,
                ent2="disease",
                prop2="",
                label2="",
                use_cache=True
            )
            
            if kg_result is None or kg_result.get('hasError', False):
                return result
            
            # Parse the result according to the API response format
            # Format: {'data': {'results': '[{"n": {...}, "r": {...}, "m": {...}}]'}, ...}
            relationships = []
            
            if isinstance(kg_result, dict) and 'data' in kg_result:
                data = kg_result['data']
                if isinstance(data, dict) and 'results' in data:
                    results_str = data['results']
                    if isinstance(results_str, str):
                        try:
                            relationships = json.loads(results_str)
                        except json.JSONDecodeError as e:
                            print(f"Error parsing JSON string from results for {disease_id}: {e}")
                            return result
                    elif isinstance(results_str, list):
                        relationships = results_str
            
            # Extract information from the first matching relationship
            if relationships and len(relationships) > 0:
                item = relationships[0]
                
                # Extract publicDisease node information (n)
                if isinstance(item, dict) and 'n' in item:
                    node_n = item['n']
                    if isinstance(node_n, dict) and 'properties' in node_n:
                        properties_n = node_n['properties']
                        
                        if 'disease_id' in properties_n:
                            result['disease_id'] = str(properties_n['disease_id']).strip()
                        if 'standard_name' in properties_n:
                            result['standard_name'] = str(properties_n['standard_name']).strip()
                        if 'link' in properties_n:
                            result['link'] = str(properties_n['link']).strip()
                
                # Extract relationship information (r)
                relationship_type = ''
                aggregated_disease_id = ''
                
                if isinstance(item, dict) and 'r' in item:
                    rel = item['r']
                    if isinstance(rel, dict):
                        # Extract relationship type
                        if 'type' in rel:
                            relationship_type = str(rel['type']).strip()
                            result['relationship_type'] = relationship_type
                        
                        # Extract the other ID from the relationship (the aggregated disease ID)
                        # Check both src/dst and sourceID/targetID
                        if 'src' in rel and 'dst' in rel:
                            # If input disease_id matches dst, then src is the aggregated ID
                            # If input disease_id matches src, then dst is the aggregated ID
                            src_id = str(rel['src']).strip()
                            dst_id = str(rel['dst']).strip()
                            if disease_id == dst_id:
                                aggregated_disease_id = src_id
                            elif disease_id == src_id:
                                aggregated_disease_id = dst_id
                        
                        # Also check properties for sourceID/targetID
                        if 'properties' in rel and isinstance(rel['properties'], dict):
                            props_r = rel['properties']
                            source_id = props_r.get('sourceID', '').strip() if 'sourceID' in props_r else ''
                            target_id = props_r.get('targetID', '').strip() if 'targetID' in props_r else ''
                            
                            if not aggregated_disease_id:
                                if disease_id == target_id:
                                    aggregated_disease_id = source_id
                                elif disease_id == source_id:
                                    aggregated_disease_id = target_id
                
                # Set aggregated disease ID
                if aggregated_disease_id:
                    result['aggregated_disease_id'] = aggregated_disease_id
                
                # Extract disease node information (m) only if relationship type is "disease_exact"
                if relationship_type == "disease_exact" and isinstance(item, dict) and 'm' in item:
                    node_m = item['m']
                    if isinstance(node_m, dict) and 'properties' in node_m:
                        properties_m = node_m['properties']
                        
                        # Extract synonyms (semicolon-separated string)
                        if 'synonyms' in properties_m:
                            synonyms = properties_m['synonyms']
                            if isinstance(synonyms, str):
                                if synonyms.strip():
                                    result['synonyms'] = [s.strip() for s in synonyms.split(';') if s.strip()]
                            elif isinstance(synonyms, list):
                                result['synonyms'] = [str(s).strip() for s in synonyms if s]
                        
                        if 'standard_name' in properties_m:
                            result['standard_name'] = str(properties_m['standard_name']).strip()
                        if 'disease_type' in properties_m:
                            result['disease_type'] = str(properties_m['disease_type']).strip()
                        if 'description' in properties_m:
                            result['description'] = str(properties_m['description']).strip()
                        if 'is_rare' in properties_m:
                            result['is_rare'] = str(properties_m['is_rare']).strip()
                
                # Query disease to phenotype relationships to get phenotypes and frequencies
                if aggregated_disease_id:
                    try:
                        # Query knowledge graph for disease to phenotype relationships
                        phenotype_kg_result = self.kg_query.search(
                            search_type="relsearch",
                            ent1="disease",
                            prop1="disease_id",
                            label1=aggregated_disease_id,
                            ent2="phenotype",
                            prop2="",
                            label2="",
                            use_cache=True
                        )
                        
                        if phenotype_kg_result is not None:
                            phenotype_relationships = []
                            
                            if isinstance(phenotype_kg_result, dict) and 'data' in phenotype_kg_result:
                                data = phenotype_kg_result['data']
                                if isinstance(data, dict) and 'results' in data:
                                    results_str = data['results']
                                    if isinstance(results_str, str):
                                        try:
                                            phenotype_relationships = json.loads(results_str)
                                        except json.JSONDecodeError:
                                            pass
                                    elif isinstance(results_str, list):
                                        phenotype_relationships = results_str
                            
                            # Extract phenotypes and frequencies from relationships
                            phenotypes = []
                            phenotype_max_frequencies = []
                            
                            for rel_item in phenotype_relationships:
                                if not isinstance(rel_item, dict):
                                    continue
                                
                                # Extract phenotype node (could be n or m)
                                phenotype_node = None
                                source_node = rel_item.get('n', {})
                                target_node = rel_item.get('m', {})
                                
                                # Determine which node is phenotype
                                if isinstance(source_node, dict):
                                    tags = source_node.get('tags', [])
                                    if isinstance(tags, list) and 'phenotype' in tags:
                                        phenotype_node = source_node
                                
                                if phenotype_node is None and isinstance(target_node, dict):
                                    tags = target_node.get('tags', [])
                                    if isinstance(tags, list) and 'phenotype' in tags:
                                        phenotype_node = target_node
                                
                                # Extract phenotype ID and frequency from relationship
                                if phenotype_node is not None:
                                    props = phenotype_node.get('properties', {})
                                    if isinstance(props, dict) and 'hpo_id' in props:
                                        phenotype_id = str(props['hpo_id']).strip()
                                        
                                        # Extract frequency_max from relationship
                                        relationship = rel_item.get('r', {})
                                        frequency_max = ''
                                        if isinstance(relationship, dict):
                                            if 'properties' in relationship:
                                                rel_props = relationship['properties']
                                                if isinstance(rel_props, dict):
                                                    if 'frequency_max' in rel_props:
                                                        frequency_max = str(rel_props['frequency_max']).strip()
                                        
                                        if phenotype_id:
                                            phenotypes.append(phenotype_id)
                                            phenotype_max_frequencies.append(frequency_max if frequency_max else '')
                            
                            # Store phenotypes and frequencies maintaining correspondence
                            result['phenotypes'] = phenotypes
                            result['phenotype_max_frequencies'] = phenotype_max_frequencies
                    
                    except Exception as e:
                        print(f"Error getting phenotypes for disease {aggregated_disease_id}: {e}")
                        # Continue even if phenotype query fails
            
        except Exception as e:
            print(f"Error getting disease expanded info from KG for {disease_id}: {e}")
            import traceback
            traceback.print_exc()
        
        return result

    def get_diseases_from_hpo_id_from_kg(self, hpo_id: str) -> Dict:
        """
        Get all diseases associated with a phenotype HPO ID from knowledge graph.
        Only queries KG, does not use local data.
        
        Args:
            hpo_id: HPO ID (e.g., "HP:0000001")
            
        Returns:
            Dict mapping aggregated disease ID (D:X) to disease info.
            Each disease info follows the format of get_disease_exp_info_from_kg, but disease_id is a list:
            {
                'disease_id': ['OMIM:123456', 'ORPHA:789'],  # List of actual disease IDs
                'standard_name': '',
                'synonyms': [],
                'disease_type':'',
                'description': '',
                'is_rare': '',
                'link': [],
                'aggregated_disease_id': 'D:1021',
                'relationship_type': '',
            }
            Example:
            {
                'D:1021': {
                    'disease_id': ['OMIM:123456', 'ORPHA:789'],
                    'standard_name': '...',
                    'synonyms': [...],
                    'disease_type':'...',
                    'description': '...',
                    'is_rare': 'yes',
                    'link': [...],
                    'aggregated_disease_id': 'D:1021',
                    'relationship_type': 'disease_exact'
                },
                ...
            }
        """
        result = {}  # aggregated_disease_id -> disease info dict with disease_id as list
        
        if not hpo_id:
            return result
        
        try:
            # Query knowledge graph for phenotype to disease relationships
            # Using relsearch to find relationships from phenotype to disease
            kg_result = self.kg_query.search(
                search_type="relsearch",
                ent1="phenotype",
                prop1="hpo_id",
                label1=hpo_id,
                ent2="disease",
                prop2="",
                label2="",
                use_cache=True
            )
            
            if kg_result is None:
                return result
            
            # Parse the result according to the API response format
            # Format: {'data': {'results': '[{"n": {...}, "r": {...}, "m": {...}}]'}, ...}
            relationships = []
            
            if isinstance(kg_result, dict) and 'data' in kg_result:
                data = kg_result['data']
                if isinstance(data, dict) and 'results' in data:
                    results_str = data['results']
                    if isinstance(results_str, str):
                        try:
                            relationships = json.loads(results_str)
                        except json.JSONDecodeError as e:
                            print(f"Error parsing JSON string from results for {hpo_id}: {e}")
                            return result
                    elif isinstance(results_str, list):
                        relationships = results_str
            
            # Extract aggregated disease IDs and their information from m nodes
            # m node contains disease information (standard_name, synonyms, description, is_rare)
            aggregated_disease_map = {}  # aggregated_disease_id -> disease_info dict
            
            for item in relationships:
                if not isinstance(item, dict):
                    continue
                
                # Get relationship information
                relationship = item.get('r', {})
                if not isinstance(relationship, dict):
                    continue
                
                # Extract aggregated disease ID from the relationship
                # Priority: r.src > m.vid > r.properties.sourceID
                aggregated_disease_id = None
                
                # First, check relationship src (for "has" relationship, src is disease, dst is phenotype)
                if 'src' in relationship:
                    src_id = str(relationship['src']).strip()
                    if src_id.startswith('D:'):
                        aggregated_disease_id = src_id
                
                # Also check the target node (m) for the disease ID (disease node)
                if not aggregated_disease_id:
                    target_node = item.get('m', {})
                    if isinstance(target_node, dict):
                        # Check vid first
                        if 'vid' in target_node:
                            vid = str(target_node['vid']).strip()
                            if vid.startswith('D:'):
                                aggregated_disease_id = vid
                
                # Also check relationship properties sourceID
                if not aggregated_disease_id and 'properties' in relationship:
                    rel_props = relationship['properties']
                    if isinstance(rel_props, dict):
                        if 'sourceID' in rel_props:
                            source_id = str(rel_props['sourceID']).strip()
                            if source_id.startswith('D:'):
                                aggregated_disease_id = source_id
                
                if aggregated_disease_id:
                    # Extract disease information from m node (disease node)
                    # This contains standard_name, synonyms, description, is_rare
                    if aggregated_disease_id not in aggregated_disease_map:
                        disease_info = {
                            'disease_id': [],  # Will be filled from second query
                            'standard_name': '',
                            'synonyms': [],
                            'disease_type':'',
                            'description': '',
                            'is_rare': '',
                            'link': [],  # Will be filled from second query
                            'aggregated_disease_id': aggregated_disease_id,
                            'relationship_type': [],  # Will be filled from second query if needed
                        }
                        
                        # Extract from m node (disease node)
                        disease_node = item.get('m', {})
                        if isinstance(disease_node, dict) and 'properties' in disease_node:
                            props = disease_node['properties']
                            if isinstance(props, dict):
                                # Extract standard_name
                                if 'standard_name' in props:
                                    disease_info['standard_name'] = str(props['standard_name']).strip()
                                
                                # Extract synonyms (semicolon-separated string)
                                if 'synonyms' in props:
                                    synonyms = props['synonyms']
                                    if isinstance(synonyms, str):
                                        if synonyms.strip():
                                            disease_info['synonyms'] = [s.strip() for s in synonyms.split(';') if s.strip()]
                                    elif isinstance(synonyms, list):
                                        disease_info['synonyms'] = [str(s).strip() for s in synonyms if s]
                                
                                if 'disease_type' in props:
                                    disease_info['disease_type'] = str(props['disease_type']).strip()
                                
                                # Extract description
                                if 'description' in props:
                                    disease_info['description'] = str(props['description']).strip()
                                
                                # Extract is_rare
                                if 'is_rare' in props:
                                    disease_info['is_rare'] = str(props['is_rare']).strip()
                        
                        aggregated_disease_map[aggregated_disease_id] = disease_info
            
            # For each aggregated disease ID, get the public disease IDs (publicDisease) and link
            # Second query: from aggregated disease to publicDisease
            for aggregated_id, disease_info in aggregated_disease_map.items():
                try:
                    # Query relationship from disease (aggregated) to publicDisease
                    # This query only returns public disease IDs and link information
                    disease_kg_result = self.kg_query.search(
                        search_type="relsearch",
                        ent1="disease",
                        prop1="disease_id",
                        label1=aggregated_id,
                        ent2="publicDisease",
                        prop2="",
                        label2="",
                        use_cache=True
                    )
                    
                    if disease_kg_result is not None:
                        disease_relationships = []
                        
                        if isinstance(disease_kg_result, dict) and 'data' in disease_kg_result:
                            data = disease_kg_result['data']
                            if isinstance(data, dict) and 'results' in data:
                                results_str = data['results']
                                if isinstance(results_str, str):
                                    try:
                                        disease_relationships = json.loads(results_str)
                                    except json.JSONDecodeError:
                                        pass
                                elif isinstance(results_str, list):
                                    disease_relationships = results_str
                        
                        # Extract public disease IDs and link from publicDisease nodes
                        # Keep them as lists to maintain correspondence with relationship_type
                        public_disease_ids = []
                        public_links = []
                        relationship_types = []
                        
                        for rel_item in disease_relationships:
                            if not isinstance(rel_item, dict):
                                continue
                            
                            # Extract relationship information (r) - get relationship_type from relationship
                            relationship_type = ''
                            if isinstance(rel_item, dict) and 'r' in rel_item:
                                rel = rel_item['r']
                                if isinstance(rel, dict):
                                    # Extract relationship type from relationship
                                    if 'type' in rel:
                                        relationship_type = str(rel['type']).strip()
                            
                            # Check source node (n) - could be disease or publicDisease
                            source_node = rel_item.get('n', {})
                            target_node = rel_item.get('m', {})
                            
                            # Determine which node is publicDisease
                            public_disease_node = None
                            
                            if isinstance(source_node, dict):
                                tags = source_node.get('tags', [])
                                if isinstance(tags, list) and 'publicDisease' in tags:
                                    public_disease_node = source_node
                            
                            if public_disease_node is None and isinstance(target_node, dict):
                                tags = target_node.get('tags', [])
                                if isinstance(tags, list) and 'publicDisease' in tags:
                                    public_disease_node = target_node
                            
                            # Extract information from publicDisease node
                            if isinstance(public_disease_node, dict):
                                props = public_disease_node.get('properties', {})
                                if isinstance(props, dict):
                                    # Extract disease_id (public disease ID like OMIM:, ORPHA:)
                                    public_disease_id = None
                                    if 'disease_id' in props:
                                        public_disease_id = str(props['disease_id']).strip()
                                        # publicDisease IDs are typically OMIM:, ORPHA:, etc., not D:
                                        if public_disease_id and not public_disease_id.startswith('D:'):
                                            pass  # Will add to list below
                                        else:
                                            public_disease_id = None
                                    
                                    # Extract link (collect all non-empty links)
                                    link_value = None
                                    if 'link' in props:
                                        link_value = str(props['link']).strip()
                                        if not link_value:
                                            link_value = None
                                    
                                    # Only add if we have at least a disease_id
                                    if public_disease_id:
                                        public_disease_ids.append(public_disease_id)
                                        public_links.append(link_value if link_value else '')
                                        relationship_types.append(relationship_type if relationship_type else '')
                        
                        # Store lists maintaining correspondence (no sorting)
                        disease_info['disease_id'] = public_disease_ids
                        disease_info['link'] = public_links
                        disease_info['relationship_type'] = relationship_types
                
                except Exception as e:
                    print(f"Error getting public disease IDs for aggregated disease {aggregated_id}: {e}")
                    import traceback
                    traceback.print_exc()
                
                # Store the mapping (only if we have at least one disease_id)
                if disease_info['disease_id']:
                    result[aggregated_id] = disease_info
        
        except Exception as e:
            print(f"Error getting diseases from HPO ID {hpo_id}: {e}")
            import traceback
            traceback.print_exc()
        
        return result
    
    def _prepare_external_sample_cache(self):
        """Precompute normalized embedding matrices for external samples"""
        self._cached_sample_matrices = []
        self._cached_sample_embeddings = []

        if not self.phe2embedding or not self.external_samples:
            return

        cached_samples = []
        cached_embeddings = []

        for sample in self.external_samples:
            sample_vectors = []
            weighted_vectors = []
            ic_values = []

            for phe in sample.get('Phenotype', []) or []:
                try:
                    hp_code = self._extract_hp_code(phe)
                except Exception:
                    continue

                embedding = self.phe2embedding.get(hp_code)
                if embedding is None:
                    continue

                vector = np.asarray(embedding, dtype=np.float32)
                norm = np.linalg.norm(vector)
                normalized_vector = vector / norm if norm > 0 else vector
                sample_vectors.append(normalized_vector)

                if self.ic_dict:
                    ic_value = self.ic_dict.get(hp_code)
                    if ic_value is not None:
                        weighted_vectors.append(vector)
                        ic_values.append(ic_value)
                else:
                    weighted_vectors.append(vector)

            if not sample_vectors:
                continue

            sample_matrix = np.vstack(sample_vectors)
            cached_samples.append((sample, sample_matrix))

            if weighted_vectors:
                weighted_vectors_array = np.asarray(weighted_vectors, dtype=np.float32)

                if self.ic_dict and ic_values:
                    try:
                        ic_array = np.asarray(ic_values, dtype=np.float32)
                        sample_embedding = np.sum(weighted_vectors_array * ic_array.reshape(-1, 1), axis=0) / np.sum(ic_array)
                        # sample_embedding = np.mean(weighted_vectors_array, axis=0)
                    except Exception:
                        sample_embedding = np.mean(weighted_vectors_array, axis=0)
                else:
                    sample_embedding = np.mean(weighted_vectors_array, axis=0)

                embedding_norm = np.linalg.norm(sample_embedding)
                sample_embedding_normalized = sample_embedding / embedding_norm if embedding_norm > 0 else sample_embedding

                cached_embeddings.append((sample, sample_embedding_normalized))

        self._cached_sample_matrices = cached_samples
        self._cached_sample_embeddings = cached_embeddings

    def _compute_weighted_embedding(self, phenotype_ids: Iterable[str]) -> Optional[np.ndarray]:
        """
        Build a normalized embedding vector for a collection of phenotypes using IC-weighted averaging.

        Args:
            phenotype_ids: Iterable of phenotype identifiers (HPO terms or strings containing HPO IDs).

        Returns:
            A normalized embedding vector or None if no valid embeddings are available.
        """
        if not phenotype_ids:
            return None

        try:
            phenotype_list = list(dict.fromkeys([phe for phe in phenotype_ids if phe]))
        except TypeError:
            phenotype_list = [phe for phe in phenotype_ids if phe]

        if not phenotype_list:
            return None

        vectors: List[np.ndarray] = []
        weights: List[float] = []

        for phenotype in phenotype_list:
            try:
                hp_code = self._extract_hp_code(phenotype)
            except Exception:
                continue

            embedding = self.phe2embedding.get(hp_code)
            ic_value = self.ic_dict.get(hp_code)

            if embedding is None or ic_value is None:
                continue

            try:
                vector = np.asarray(embedding, dtype=np.float32).reshape(-1)
            except Exception:
                continue

            if vector.size == 0 or not np.isfinite(vector).all():
                continue

            vectors.append(vector)
            weights.append(float(ic_value))

        if not vectors:
            return None

        vectors_array = np.vstack(vectors)
        weights_array = np.asarray(weights, dtype=np.float32)

        total_weight = float(np.sum(weights_array))

        if not np.isfinite(total_weight) or total_weight <= 0:
            embedding = np.mean(vectors_array, axis=0)
        else:
            try:
                embedding = np.sum(vectors_array * weights_array.reshape(-1, 1), axis=0) / total_weight
            except Exception:
                embedding = np.mean(vectors_array, axis=0)

        norm = np.linalg.norm(embedding)
        if norm > 0:
            embedding = embedding / norm

        return embedding

    def calculate_case_similarity(self, patient_phenotypes: List[str], disease_id: str) -> float:
        """
        Calculate cosine similarity between patient phenotypes and a specific disease using phenotype embeddings.

        Args:
            patient_phenotypes: Phenotype list describing the current patient.
            disease_id: Target disease identifier.
            use_samples: Whether to use phenotypes derived from true samples for the disease.

        Returns:
            Cosine similarity score between patient and disease phenotype embeddings (0.0 if unavailable).
        """
        if not self.phe2embedding or not self.ic_dict:
            return 0.0

        patient_embedding = self._compute_weighted_embedding(patient_phenotypes)
        if patient_embedding is None:
            return 0.0

        disease_embedding: Optional[np.ndarray] = None
        if self._cached_sample_embeddings:
            for sample, sample_embedding in self._cached_sample_embeddings:
                try:
                    rare_disease = sample.get('RareDisease')
                except Exception:
                    rare_disease = None

                if isinstance(rare_disease, (list, tuple, set)):
                    is_match = disease_id in rare_disease
                else:
                    is_match = rare_disease == disease_id

                if is_match and sample_embedding is not None:
                    disease_embedding = sample_embedding
                    break

        if disease_embedding is not None:
            norm = np.linalg.norm(disease_embedding)
            if norm > 0:
                disease_embedding = disease_embedding / norm
        else:
            disease_phenotypes = self.disease_to_phenotypes.get(disease_id, set())
            disease_embedding = self._compute_weighted_embedding(disease_phenotypes)
            if disease_embedding is None:
                return 0.0

        try:
            similarity = float(np.dot(patient_embedding, disease_embedding))
        except Exception:
            return 0.0

        if not np.isfinite(similarity):
            return 0.0

        return similarity

    def calculate_case_similarity_v1(self, patient_phenotypes: List[str], disease_id: str) -> float:
        """
        Calculate similarity between patient phenotypes and a specific disease using phenotype embeddings.
        This v1 version uses the same similarity calculation method as find_similar_samples_with_embeddings_v1:
        - Builds matrices for patient and disease phenotypes (each phenotype is a normalized vector)
        - Computes similarity matrix using dot product
        - Finds best match for each patient phenotype
        - Uses IC-weighted average of best matches

        Args:
            patient_phenotypes: Phenotype list describing the current patient.
            disease_id: Target disease identifier.

        Returns:
            Similarity score between patient and disease phenotype embeddings (0.0 if unavailable).
        """
        if not self.phe2embedding or not self.ic_dict:
            return 0.0

        # Build normalized embedding vectors for the current patient phenotypes
        patient_vectors = []
        patient_weights = []  # IC of patient phenotypes as weight

        for phe in patient_phenotypes:
            try:
                hp_code = self._extract_hp_code(phe)
            except Exception:
                continue

            embedding = self.phe2embedding.get(hp_code)
            if embedding is None:
                continue

            ic_value = self.ic_dict.get(hp_code)
            if ic_value is None:
                ic_value = 1

            vector = np.asarray(embedding, dtype=np.float32)
            norm = np.linalg.norm(vector)
            if norm > 0:
                vector = vector / norm
            patient_vectors.append(vector)
            patient_weights.append(ic_value)

        if not patient_vectors:
            return 0.0

        patient_matrix = np.vstack(patient_vectors)
        patient_weights_array = np.array(patient_weights)  # to numpy for compute

        # Get disease phenotypes
        disease_phenotypes = self.disease_to_phenotypes.get(disease_id, set())
        if not disease_phenotypes:
            return 0.0

        # Build normalized embedding vectors for disease phenotypes
        disease_vectors = []
        for phe in disease_phenotypes:
            try:
                hp_code = self._extract_hp_code(phe)
            except Exception:
                continue

            embedding = self.phe2embedding.get(hp_code)
            if embedding is None:
                continue

            vector = np.asarray(embedding, dtype=np.float32)
            norm = np.linalg.norm(vector)
            if norm > 0:
                vector = vector / norm
            disease_vectors.append(vector)

        if not disease_vectors:
            return 0.0

        disease_matrix = np.vstack(disease_vectors)

        try:
            # Compute similarity matrix: patient_matrix (n_patient x dim) @ disease_matrix.T (dim x n_disease)
            similarities = np.dot(patient_matrix, disease_matrix.T)

            # For each patient phenotype, find the best match with disease phenotypes
            best_matches_for_patient = np.max(similarities, axis=1)

            # Use IC-weighted average of best matches
            semantic_score = float(np.average(best_matches_for_patient, weights=patient_weights_array))
        except Exception as exc:
            print(f"Error computing similarity for disease {disease_id}: {exc}")
            return 0.0

        if not np.isfinite(semantic_score):
            return 0.0

        return semantic_score
    
    def _compute_weighted_embedding_from_kg(self, phenotype_ids: Iterable[str], phenotype_info_cache: Dict[str, Dict] = None) -> Optional[np.ndarray]:
        """
        Build a normalized embedding vector for a collection of phenotypes using IC-weighted averaging.
        All data (embeddings and IC values) are fetched from KG, not from local files.

        Args:
            phenotype_ids: Iterable of phenotype identifiers (HPO terms or strings containing HPO IDs).
            phenotype_info_cache: Optional cache dictionary mapping hpo_id to phenotype info from KG.

        Returns:
            A normalized embedding vector or None if no valid embeddings are available.
        """
        if not phenotype_ids:
            return None

        try:
            phenotype_list = list(dict.fromkeys([phe for phe in phenotype_ids if phe]))
        except TypeError:
            phenotype_list = [phe for phe in phenotype_ids if phe]

        if not phenotype_list:
            return None

        if phenotype_info_cache is None:
            phenotype_info_cache = {}

        vectors: List[np.ndarray] = []
        weights: List[float] = []

        for phenotype in phenotype_list:
            try:
                hp_code = self._extract_hp_code(phenotype)
            except Exception:
                continue

            # Get phenotype info from KG (use cache if available)
            if hp_code not in phenotype_info_cache:
                phenotype_info = self.get_phenotype_info_from_kg(hp_code)
                phenotype_info_cache[hp_code] = phenotype_info
            else:
                phenotype_info = phenotype_info_cache[hp_code]

            embedding_list = phenotype_info.get('embedding', [])
            ic_value = phenotype_info.get('IC', 0.0)

            if not embedding_list or ic_value is None or ic_value <= 0:
                continue

            try:
                vector = np.asarray(embedding_list, dtype=np.float32).reshape(-1)
            except Exception:
                continue

            if vector.size == 0 or not np.isfinite(vector).all():
                continue

            vectors.append(vector)
            weights.append(float(ic_value))

        if not vectors:
            return None

        vectors_array = np.vstack(vectors)
        weights_array = np.asarray(weights, dtype=np.float32)

        total_weight = float(np.sum(weights_array))

        if not np.isfinite(total_weight) or total_weight <= 0:
            embedding = np.mean(vectors_array, axis=0)
        else:
            try:
                embedding = np.sum(vectors_array * weights_array.reshape(-1, 1), axis=0) / total_weight
            except Exception:
                embedding = np.mean(vectors_array, axis=0)

        norm = np.linalg.norm(embedding)
        if norm > 0:
            embedding = embedding / norm

        return embedding

    def calculate_case_similarity_from_kg(self, patient_phenotypes: List[str], disease_phenotypes: List[str], phenotype_info_cache: Dict[str, Dict] = None) -> float:
        """
        Calculate cosine similarity between patient phenotypes and disease phenotypes from KG using phenotype embeddings.
        All data (embeddings and IC values) are fetched from KG, not from local files.

        Args:
            patient_phenotypes: Phenotype list describing the current patient.
            disease_phenotypes: Phenotype list for the disease from KG.
            phenotype_info_cache: Optional cache dictionary mapping hpo_id to phenotype info from KG.

        Returns:
            Cosine similarity score between patient and disease phenotype embeddings (0.0 if unavailable).
        """
        if phenotype_info_cache is None:
            phenotype_info_cache = {}

        # Compute embeddings from KG for both patient and disease phenotypes
        patient_embedding = self._compute_weighted_embedding_from_kg(patient_phenotypes, phenotype_info_cache)
        if patient_embedding is None:
            return 0.0

        disease_embedding = self._compute_weighted_embedding_from_kg(disease_phenotypes, phenotype_info_cache)
        if disease_embedding is None:
            return 0.0

        try:
            similarity = float(np.dot(patient_embedding, disease_embedding))
        except Exception:
            return 0.0

        if not np.isfinite(similarity):
            return 0.0

        return similarity
     
    def calculate_case_similarity_from_kg_v1(self, patient_phenotypes: List[str], disease_phenotypes: List[str], phenotype_info_cache: Dict[str, Dict] = None) -> float:
        """
        Calculate similarity between patient phenotypes and disease phenotypes from KG using phenotype embeddings.
        This v1 version uses the same similarity calculation method as find_similar_samples_with_embeddings_v1:
        - Builds matrices for patient and disease phenotypes (each phenotype is a normalized vector)
        - Computes similarity matrix using dot product
        - Finds best match for each patient phenotype
        - Uses IC-weighted average of best matches
        All data (embeddings and IC values) are fetched from KG, not from local files.

        Args:
            patient_phenotypes: Phenotype list describing the current patient.
            disease_phenotypes: Phenotype list for the disease from KG.
            phenotype_info_cache: Optional cache dictionary mapping hpo_id to phenotype info from KG.

        Returns:
            Similarity score between patient and disease phenotype embeddings (0.0 if unavailable).
        """
        if phenotype_info_cache is None:
            phenotype_info_cache = {}

        # Build normalized embedding vectors for the current patient phenotypes
        patient_vectors = []
        patient_weights = []  # IC of patient phenotypes as weight

        for phe in patient_phenotypes:
            try:
                hp_code = self._extract_hp_code(phe)
            except Exception:
                continue

            # Get phenotype info from KG (use cache if available)
            if hp_code not in phenotype_info_cache:
                phenotype_info = self.get_phenotype_info_from_kg(hp_code)
                phenotype_info_cache[hp_code] = phenotype_info
            else:
                phenotype_info = phenotype_info_cache[hp_code]

            embedding_list = phenotype_info.get('embedding', [])
            ic_value = phenotype_info.get('IC', 0.0)

            if not embedding_list or ic_value is None or ic_value <= 0:
                continue

            try:
                vector = np.asarray(embedding_list, dtype=np.float32).reshape(-1)
            except Exception:
                continue

            if vector.size == 0 or not np.isfinite(vector).all():
                continue

            norm = np.linalg.norm(vector)
            if norm > 0:
                vector = vector / norm
            patient_vectors.append(vector)
            patient_weights.append(float(ic_value))

        if not patient_vectors:
            return 0.0

        patient_matrix = np.vstack(patient_vectors)
        patient_weights_array = np.array(patient_weights)  # to numpy for compute

        # Build normalized embedding vectors for disease phenotypes
        disease_vectors = []
        for phe in disease_phenotypes:
            try:
                hp_code = self._extract_hp_code(phe)
            except Exception:
                continue

            # Get phenotype info from KG (use cache if available)
            if hp_code not in phenotype_info_cache:
                phenotype_info = self.get_phenotype_info_from_kg(hp_code)
                phenotype_info_cache[hp_code] = phenotype_info
            else:
                phenotype_info = phenotype_info_cache[hp_code]

            embedding_list = phenotype_info.get('embedding', [])
            if not embedding_list:
                continue

            try:
                vector = np.asarray(embedding_list, dtype=np.float32).reshape(-1)
            except Exception:
                continue

            if vector.size == 0 or not np.isfinite(vector).all():
                continue

            norm = np.linalg.norm(vector)
            if norm > 0:
                vector = vector / norm
            disease_vectors.append(vector)

        if not disease_vectors:
            return 0.0

        disease_matrix = np.vstack(disease_vectors)

        try:
            # Compute similarity matrix: patient_matrix (n_patient x dim) @ disease_matrix.T (dim x n_disease)
            similarities = np.dot(patient_matrix, disease_matrix.T)

            # For each patient phenotype, find the best match with disease phenotypes
            best_matches_for_patient = np.max(similarities, axis=1)

            # Use IC-weighted average of best matches
            semantic_score = float(np.average(best_matches_for_patient, weights=patient_weights_array))
        except Exception as exc:
            print(f"Error computing similarity from KG: {exc}")
            return 0.0

        if not np.isfinite(semantic_score):
            return 0.0

        return semantic_score
    
    
    def _extract_hp_code(self, phenotype: str) -> str:
        """Extract HP code from phenotype string"""
        if '(' in phenotype and ')' in phenotype:
            return phenotype.split('(')[-1].split(')')[0]
        elif phenotype.startswith('HP:'):
            return phenotype
        else:
            return phenotype
    
    def _is_hp_with_only_child(self, parent_phenotype: str) -> str:
        """Check if a parent phenotype has only one child, return the child if so"""
        # use prebuilt parent-child map for fast lookup
        if hasattr(self, 'parent_to_children'):
            children = self.parent_to_children.get(parent_phenotype, [])
            return children[0] if len(children) == 1 else None
        else:
            return None
        # # if no prebuilt map, use legacy (backward compat)
        # children = []
        # for phenotype, parent_nodes in self.hpo_is_a.items():
        #     if parent_phenotype in parent_nodes:
        #         children.append(phenotype)
        
        # return children[0] if len(children) == 1 else None

    def find_similar_samples_with_embeddings(self, current_phenotypes: List[str], k_shot: int = 50, exclude_sample_id: int = None) -> Tuple[List[Dict], List[float]]:
        """Find similar samples using embedding method"""
        # TODO: patient-case match does not consider parent/child phenotypes; should in theory
        # print(f"DEBUG: Embedding method - phe2embedding: {len(self.phe2embedding)}, ic_dict: {len(self.ic_dict)}, external_samples: {len(self.external_samples)}")
        if not self.phe2embedding or not self.ic_dict or not self.external_samples:
            print("Warning: Embedding data or external samples not available")
            return [], []
        
        # Calculate current sample embedding
        current_embeddings = []
        current_ic_values = []
        
        for phe in current_phenotypes:
            try:
                hp_code = self._extract_hp_code(phe)
                if hp_code in self.phe2embedding and hp_code in self.ic_dict:
                    current_embeddings.append(np.array(self.phe2embedding[hp_code]))
                    current_ic_values.append(self.ic_dict[hp_code])
            except Exception as e:
                continue
        
        if not current_embeddings:
            print("Warning: No valid phenotypes found in embedding dictionary")
            return [], []
        
        # Calculate weighted average embedding
        try:
            current_embeddings = np.array(current_embeddings)
            current_ic_values = np.array(current_ic_values)
            current_embedding = np.sum(current_embeddings * current_ic_values.reshape(-1, 1), axis=0) / np.sum(current_ic_values)
            # current_embedding = np.mean(current_embeddings, axis=0)
        except Exception as e:
            current_embedding = np.mean(current_embeddings, axis=0)
        
        if not self._cached_sample_embeddings and self.external_samples:
            self._prepare_external_sample_cache()

        if not self._cached_sample_embeddings:
            print("Warning: No valid samples found with embeddings")
            return [], []

        # Normalize current embedding
        current_embedding_norm = np.linalg.norm(current_embedding)
        if current_embedding_norm > 0:
            current_embedding_normalized = current_embedding / current_embedding_norm
        else:
            current_embedding_normalized = current_embedding

        similarity_scores = []

        for sample, sample_embedding_normalized in self._cached_sample_embeddings:
            try:
                department = sample.get('Department', '')
            except Exception:
                department = ''

            if exclude_sample_id and department and department.split('_')[-1] == str(exclude_sample_id):
                continue

            try:
                similarity = float(np.dot(sample_embedding_normalized, current_embedding_normalized))
            except Exception:
                continue

            similarity_scores.append((sample, similarity))

        if not similarity_scores:
            print("Warning: No valid samples found with embeddings")
            return [], []

        similarity_scores.sort(key=lambda item: item[1], reverse=True)

        top_samples = [item[0] for item in similarity_scores[:k_shot]]
        top_scores = [item[1] for item in similarity_scores[:k_shot]]

        return top_samples, top_scores

    def find_similar_samples_with_embeddings_v1(self, current_phenotypes: List[str], k_shot: int = 50, exclude_sample_id: int = None) -> Tuple[List[Dict], List[float]]:
        """Find similar samples using semantic matching of phenotype embeddings."""
        if not self.phe2embedding or not self.ic_dict or not self.external_samples:
            print("Warning: Embedding data, IC dict or external samples not available")
            return [], []

        # Build normalized embedding vectors for the current patient phenotypes (Q)
        patient_vectors = []
        patient_weights = []  # IC of patient phenotypes as weight

        for phe in current_phenotypes:
            try:
                hp_code = self._extract_hp_code(phe)
            except Exception:
                continue

            embedding = self.phe2embedding.get(hp_code)
            if embedding is None:
                continue

            ic_value = self.ic_dict.get(hp_code)
            if ic_value is None:
                ic_value = 1

            vector = np.asarray(embedding, dtype=np.float32)
            norm = np.linalg.norm(vector)
            if norm > 0:
                vector = vector / norm
            patient_vectors.append(vector)
            patient_weights.append(ic_value)

        if not patient_vectors:
            print("Warning: No valid phenotypes found in embedding dictionary")
            return [], []

        patient_matrix = np.vstack(patient_vectors)
        patient_weights_array = np.array(patient_weights)  # to numpy for compute

        if not self._cached_sample_matrices and self.external_samples:
            self._prepare_external_sample_cache()

        if not self._cached_sample_matrices:
            print("Warning: No valid samples found with embeddings")
            return [], []

        similarity_scores: List[Tuple[Dict, float]] = []

        for sample, sample_matrix in self._cached_sample_matrices:
            try:
                department = sample.get('Department', '')
            except Exception:
                department = ''

            if exclude_sample_id and department and department.split('_')[-1] == str(exclude_sample_id):
                continue

            try:
                similarities = np.dot(patient_matrix, sample_matrix.T)

                # best_matches_for_sample = np.max(similarities, axis=0) # for many sample phenotypes, this alone is weak; real patient may have few
                # semantic_score_for_sample = float(np.mean(best_matches_for_sample))  # TODO: consider IC/freq weight
                
                best_matches_for_patient = np.max(similarities, axis=1)  # if patient has noisy phenotypes but 2 core match, unidirectional mean is pulled down.
                # semantic_score_for_patient = float(np.mean(best_matches_for_patient))  # TODO: consider IC/freq weight
                semantic_score_for_patient = float(np.average(best_matches_for_patient, weights=patient_weights_array))
                
                # semantic_score = (semantic_score_for_sample + semantic_score_for_patient) / 2
                semantic_score = semantic_score_for_patient
            except Exception as exc:
                print(f"Error computing similarity for sample {sample.get('Department')}: {exc}")
                continue

            similarity_scores.append((sample, semantic_score))

        if not similarity_scores:
            print("Warning: No valid samples found with embeddings")
            return [], []

        similarity_scores.sort(key=lambda item: item[1], reverse=True)

        top_samples = [item[0] for item in similarity_scores[:k_shot]]
        top_scores = [item[1] for item in similarity_scores[:k_shot]]

        return top_samples, top_scores
   
    # deprecated
    def find_similar_samples_with_phenotype_overlap(self, current_phenotypes: List[str], k_shot: int = 3) -> Tuple[List[Dict], List[float]]:
        """Find similar samples using phenotype overlap method (original method)"""
        print(f"DEBUG: Overlap method - external_samples: {len(self.external_samples)}")
        if not self.external_samples:
            print("Warning: External samples not available")
            return [], []
        
        # Convert current phenotypes to set for faster lookup
        current_phenotype_set = set(current_phenotypes)
        
        # Calculate overlap scores for all external samples
        sample_scores = []
        
        for sample in self.external_samples:
            sample_phenotypes = sample.get('Phenotype', [])
            sample_phenotype_set = set(sample_phenotypes)
            
            # Calculate Jaccard similarity
            intersection = len(current_phenotype_set.intersection(sample_phenotype_set))
            union = len(current_phenotype_set.union(sample_phenotype_set))
            
            if union > 0:
                jaccard_similarity = intersection / union
            else:
                jaccard_similarity = 0.0
            
            sample_scores.append((sample, jaccard_similarity))
        
        # Sort by similarity (descending)
        sample_scores.sort(key=lambda x: x[1], reverse=True)
        
        # Select top k_shot samples
        similar_samples = []
        similar_values = []
        for sample, score in sample_scores[:k_shot]:
            similar_samples.append(sample)
            similar_values.append(score)
        
        return similar_samples, similar_values

    def load_disease_descriptions(self):
        """Load disease descriptions from disease_descriptions_batch.json file"""
        if not self.disease_descriptions_file:
            print("No disease descriptions file specified, skipping loading...")
            return
            
        if not os.path.exists(self.disease_descriptions_file):
            print(f"Warning: Disease descriptions file not found at {self.disease_descriptions_file}")
            return
            
        print(f"Loading disease descriptions from {self.disease_descriptions_file}...")
        try:
            with open(self.disease_descriptions_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                
            # Extract disease descriptions from the JSON structure
            if 'results' in data:
                for disease_id, disease_info in data['results'].items():
                    if isinstance(disease_info, dict) and 'description' in disease_info:
                        description = disease_info.get('description', '').strip()
                        self.disease_descriptions[disease_id] = description
                
                for _, disease_ids in self.disease_name_to_ids.items():
                    longest_description = ""
                    for disease_id in disease_ids:
                        description = self.disease_descriptions.get(disease_id, "").strip()
                        if len(description) > len(longest_description):
                            longest_description = description

                    for disease_id in disease_ids:
                        self.disease_descriptions[disease_id] = longest_description
                            
            print(f"Loaded {len(self.disease_descriptions)} disease descriptions from file")
            
        except Exception as e:
            print(f"Error loading disease descriptions from {self.disease_descriptions_file}: {e}")
            self.disease_descriptions = {}

    def load_disease_names(self):
        """Load disease names from phenotype.hpoa file first, then from disease_mapping.json file"""
        print("Loading disease names from phenotype.hpoa file first...")
        
        # Initialize disease names dictionary
        self.disease_names = {}
        
        # Load disease names from phenotype.hpoa file first
        if self.phenotype_hpoa_file:
            print(f"Loading disease names from {self.phenotype_hpoa_file}...")
            try:
                hpoa_df = pd.read_csv(self.phenotype_hpoa_file, sep='\t', dtype=str, comment='#')
                
                # Count disease names added from phenotype.hpoa
                hpoa_disease_count = 0
                
                for _, row in hpoa_df.iterrows():
                    disease_id = str(row['database_id']) if pd.notna(row['database_id']) else ''
                    disease_name = str(row['disease_name']) if pd.notna(row['disease_name']) else ''
                    
                    # Skip if disease_id or disease_name is empty
                    if not disease_id or not disease_name:
                        continue
                    
                    # Add disease name from phenotype.hpoa
                    if not self._should_exclude_alias(disease_name, disease_id):
                        self.disease_names[disease_id] = disease_name
                        self.disease_mapping_with_synonyms[disease_id] = [disease_name]
                    hpoa_disease_count += 1
                
                print(f"Loaded {hpoa_disease_count} disease names from phenotype.hpoa")
                
            except Exception as e:
                print(f"Error loading disease names from phenotype.hpoa: {e}")
        
        # # Load additional disease names from disease_mapping.json file
        # print("Loading additional disease names from disease_mapping.json...")
        # try:
        #     with open(self.disease_mapping_file, 'r', encoding='utf-8') as f:
        #         disease_mapping = json.load(f)
            
        #     # Count new disease names added from disease_mapping.json
        #     mapping_disease_count = 0
            
        #     for disease_id, disease_name in disease_mapping.items():
        #         # Split disease name if it contains multiple variants separated by ';' or '/'
        #         split_disease_names = self._split_disease_names(disease_name)
                
        #         # Add disease name if not already present or if current name is None/empty
        #         if (disease_id not in self.disease_names or 
        #             not self.disease_names[disease_id] or 
        #             self.disease_names[disease_id] is None):
        #             # Use the first split name as the primary disease name
        #             self.disease_names[disease_id] = split_disease_names[0] if split_disease_names else disease_name
        #             # Add all split names as synonyms (filtered)
        #             filtered_split_names = [name for name in split_disease_names if not self._should_exclude_alias(name, disease_id)]
        #             self.disease_mapping_with_synonyms[disease_id] = filtered_split_names.copy()
        #             mapping_disease_count += 1
        #         else:
        #             # Add all split names as synonyms if they don't already exist
        #             for split_name in split_disease_names:
        #                 if not self._should_exclude_alias(split_name, disease_id) and split_name not in self.disease_mapping_with_synonyms[disease_id]:
        #                     self.disease_mapping_with_synonyms[disease_id].append(split_name)
            
        #     print(f"Added {mapping_disease_count} new disease names from disease_mapping.json")
            
        # except Exception as e:
        #     print(f"Error loading disease_mapping.json: {e}")
        #     print("Will continue with disease names from phenotype.hpoa only")
        
        # Load disease synonyms from OMIM file if config is available
        # if self.config and 'omim_file' in self.config:
        #     self._load_disease_synonyms()
        
        # supplement from Phenotypes_Associated_with_Rare_Disorders.json (phenotype-disease mapping)
        # see load_phenotype_disease_mappings; load from orphanet Phenotypes_Associated_with_Rare_Disorders.json
        self.load_phenotype_disease_mappings_from_orphanet()
        
        # Load Orphanet disease mappings if config is available
        if self.config and 'orphanet_files' in self.config:
            self._load_orphanet_alignment_mapping()
            self._load_orphanet_disease_mapping()
        
        # Load MONDO disease names if config is available
        if self.config and 'mondo_file' in self.config:
            self._load_disease_names_from_mondo()
        
        # Build disease name to disease IDs mapping (case-insensitive)
        # Include all disease names from both self.disease_names and disease synonyms
        self._build_disease_name_to_ids_mapping()
        
        # extend disease type to same-name diseases (after disease_name_to_ids)
        if self.disease_types:
            self._expand_disease_types_to_synonyms()
        
        # extend rare-disease type to same-name diseases
        if hasattr(self, 'rare_disease_types') and self.rare_disease_types:
            self._expand_rare_disease_types_to_synonyms()
    
    def _should_exclude_alias(self, alias: str, disease_id: str = None) -> bool:
        """Exclude alias: all-caps and len<6, unless first occurrence of disease ID."""
        if not alias or not alias.strip():
            return True
        alias = alias.strip()
        
        # if first occurrence of disease ID, keep even if abbrev
        if disease_id and disease_id not in self.disease_names:
            return False
            
        return alias.isupper() and len(alias) < 6

    def _is_placeholder_disease_name(self, name: str) -> bool:
        """Identify placeholder disease names that should be skipped."""
        if not name:
            return True

        normalized = name.strip().lower()
        if not normalized:
            return True

        if normalized == "removed from database":
            return True

        # Strings like "MOVED TO 123456" are handled separately but include here for safety
        if normalized.startswith("moved to"):
            return True

        return False

    def _load_disease_synonyms(self):
        """Load disease aliases from OMIM."""
        if not self.config or 'omim_file' not in self.config:
            print("Warning: No OMIM file path configured")
            return
            
        omim_file = self.config['omim_file']
        
        if not os.path.exists(omim_file):
            print(f"Warning: OMIM file not found at {omim_file}")
            return
        
        try:
            print(f"Loading disease synonyms from OMIM file: {omim_file}")
            
            with open(omim_file, 'r', encoding='utf-8') as f:
                omim_data = json.load(f)
            
            synonyms_added = 0
            
            for disease_id, disease_info in omim_data.items():
                # get English name
                eng_name = disease_info.get('ENG_NAME', '')
                if (
                    eng_name
                    and not self._is_placeholder_disease_name(eng_name)
                    and not eng_name.startswith('MOVED TO')
                    and not self._should_exclude_alias(eng_name, disease_id)
                ):
                    # if disease ID not in dict, create new list
                    if disease_id not in self.disease_mapping_with_synonyms:
                        self.disease_mapping_with_synonyms[disease_id] = [eng_name]
                        if disease_id not in self.disease_names:
                            self.disease_names[disease_id] = eng_name
                        synonyms_added += 1
                    elif eng_name not in self.disease_mapping_with_synonyms[disease_id]:
                        self.disease_mapping_with_synonyms[disease_id].append(eng_name)
                        synonyms_added += 1
                
                # get ALT_NAME and split by ;;
                alt_name = disease_info.get('ALT_NAME', '')
                if alt_name and disease_id in self.disease_mapping_with_synonyms:
                    # split ALT_NAME by ;;
                    alt_names = [name.strip() for name in alt_name.split(';;') if name.strip()]
                    for alt in alt_names:
                        if (
                            not self._is_placeholder_disease_name(alt)
                            and not self._should_exclude_alias(alt, disease_id)
                            and alt not in self.disease_mapping_with_synonyms[disease_id]
                        ):
                            self.disease_mapping_with_synonyms[disease_id].append(alt)
                            synonyms_added += 1
            
            print(f"Added {synonyms_added} disease synonyms from OMIM file")
            
        except Exception as e:
            print(f"Warning: Failed to load disease synonyms from OMIM file: {e}")
    
    def _load_orphanet_alignment_mapping(self):
        """Load orpha_code -> name/aliases from Orphanet JSON."""
        orphanet_alignment_file = self.config['orphanet_files']['alignment_json']
        
        if not os.path.exists(orphanet_alignment_file):
            print(f"Warning: Orphanet alignment JSON file not found at {orphanet_alignment_file}")
            return
        
        try:
            print(f"Loading disease mapping from Orphanet alignment JSON file: {orphanet_alignment_file}")
            added_count = 0
            synonyms_added = 0
            
            # read JSON
            with open(orphanet_alignment_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # check data structure
            if 'disorder_list' not in data or 'disorders' not in data['disorder_list']:
                print("Warning: Invalid JSON structure - missing disorder_list or disorders")
                return
            
            disorders = data['disorder_list']['disorders']
            print(f"Found {len(disorders)} disorders in Orphanet alignment file")
            
            # iterate all diseases
            for disorder in disorders:
                orpha_code = disorder.get('orpha_code', '').strip()
                disease_name = disorder.get('name', '').strip()
                synonyms = disorder.get('synonyms', [])
                
                # only valid orpha_code and disease name
                # create ORPHA:xxxxxx ID
                orpha_id = f"ORPHA:{orpha_code}"
                if orpha_code and disease_name and orpha_code != 'nan' and disease_name != 'nan' and not self._should_exclude_alias(disease_name, orpha_id):
                    
                    # add main disease name to disease_names
                    if orpha_id not in self.disease_names:
                        self.disease_names[orpha_id] = disease_name
                        self.disease_mapping_with_synonyms[orpha_id] = [disease_name]
                        added_count += 1
                    elif (orpha_id in self.disease_names and 
                          disease_name not in self.disease_mapping_with_synonyms[orpha_id]):
                        self.disease_mapping_with_synonyms[orpha_id].append(disease_name)
                        synonyms_added += 1
                    
                    # handle synonyms
                    if synonyms and isinstance(synonyms, list):
                        for synonym in synonyms:
                            if isinstance(synonym, dict) and 'text' in synonym:
                                synonym_text = synonym['text'].strip()
                                if (synonym_text and not self._should_exclude_alias(synonym_text, orpha_id) and
                                    synonym_text not in self.disease_mapping_with_synonyms[orpha_id]):
                                    self.disease_mapping_with_synonyms[orpha_id].append(synonym_text)
                                    synonyms_added += 1
            
            print(f"Added {added_count} disease mappings from Orphanet alignment JSON file")
            print(f"Added {synonyms_added} synonyms from Orphanet alignment JSON file")
            
        except Exception as e:
            print(f"Warning: Failed to load Orphanet alignment mapping: {e}")
            import traceback
            traceback.print_exc()
    
    def _load_orphanet_disease_mapping(self):
        """Load OrphaNumber->name from Orphanet CSV, supplement main mapping."""
        orphanet_file = self.config['orphanet_files']['categorization_csv']
        
        if not os.path.exists(orphanet_file):
            print(f"Warning: Orphanet CSV file not found at {orphanet_file}")
            return
        
        try:
            print(f"Loading disease mapping from Orphanet CSV file: {orphanet_file}")
            added_count = 0
            
            # read CSV with pandas
            df = pd.read_csv(orphanet_file)
            
            # check required columns
            required_columns = ['OrphaNumber', 'Disorder_Name']
            if not all(col in df.columns for col in required_columns):
                print(f"Warning: Required columns {required_columns} not found in Orphanet CSV file")
                return
            
            # check Category for disease type mapping
            has_category = 'Category' in df.columns
            
            # iterate df, add mapping
            for index, row in df.iterrows():
                orpha_number = str(row['OrphaNumber']).strip()
                disorder_name = str(row['Disorder_Name']).strip()
                
                # get disease type if exists
                disease_category = None
                if has_category:
                    disease_category = str(row['Category']).strip()
                    if disease_category == 'nan' or not disease_category:
                        disease_category = None
                
                # only valid OrphaNumber and disease name
                orpha_id = f"ORPHA:{orpha_number}"
                if orpha_number and disorder_name and orpha_number != 'nan' and disorder_name != 'nan' and not self._should_exclude_alias(disorder_name, orpha_id):
                    
                    # add Orphanet mapping, ORPHA: as key
                    if orpha_id not in self.disease_names:
                        self.disease_names[orpha_id] = disorder_name
                        self.disease_mapping_with_synonyms[orpha_id] = [disorder_name]
                        added_count += 1
                    elif (orpha_id in self.disease_names and disorder_name not in self.disease_mapping_with_synonyms[orpha_id]):
                        self.disease_mapping_with_synonyms[orpha_id].append(disorder_name)
                    
                    # add disease type mapping
                    if disease_category and orpha_id not in self.disease_types:
                        self.disease_types[orpha_id] = disease_category
            
            print(f"Added {added_count} disease mappings from Orphanet CSV file to main mapping")
            
            
        except Exception as e:
            print(f"Warning: Failed to load Orphanet disease mapping: {e}")
            import traceback
            traceback.print_exc()
    
    def _expand_disease_types_to_synonyms(self):
        """Extend disease type to same-name diseases per disease_name_to_ids."""
        if not self.disease_types or not self.disease_name_to_ids:
            return
        
        # for each disease ID without type, find same-name with type
        for disease_id, disease_name in self.disease_names.items():
            # if this disease ID has no type yet
            if disease_id not in self.disease_types:
                disease_name_lower = disease_name.lower()
                
                # find all disease IDs with same name
                if disease_name_lower in self.disease_name_to_ids:
                    synonym_ids = self.disease_name_to_ids[disease_name_lower]
                    
                    # find same-name with type
                    for synonym_id in synonym_ids:
                        if synonym_id != disease_id and synonym_id in self.disease_types:
                            # copy type from same-name
                            self.disease_types[disease_id] = self.disease_types[synonym_id]
                            break
    
    def _load_disease_names_from_mondo(self):
        """Supplement names/aliases from mondo_parsed_full.json, keyed by Orphanet/OMIM in equivalent_ids."""
        mondo_file = self.config['mondo_file']
        
        if not os.path.exists(mondo_file):
            print(f"Warning: MONDO parsed full file not found at {mondo_file}")
            return
        
        try:
            print(f"Loading disease names from MONDO parsed full file: {mondo_file}")
            orphanet_count = 0
            omim_count = 0
            
            # create separate rare-disease type dict
            if not hasattr(self, 'rare_disease_types'):
                self.rare_disease_types = {}
            
            with open(mondo_file, 'r', encoding='utf-8') as f:
                mondo_data = json.load(f)
            
            for mondo_id, disease_info in mondo_data.items():
                disease_name = disease_info.get('name', '')
                synonyms = disease_info.get('synonyms', [])
                equivalent_ids = disease_info.get('equivalent_ids', [])
                is_rare = disease_info.get('is_rare', False)
                
                # only when disease name present
                if not disease_name:
                    continue
                
                # collect synonym names (dedup)
                all_synonyms = []
                seen_synonyms = set()
                for synonym in synonyms:
                    synonym_type = synonym.get('type', '')
                    if not synonym_type or synonym_type.upper() != 'EXACT':
                        continue
                    synonym_name = synonym.get('name', '')
                    if synonym_name and not self._should_exclude_alias(synonym_name) and synonym_name not in seen_synonyms:
                        all_synonyms.append(synonym_name)
                        seen_synonyms.add(synonym_name)

                # # record MONDO disease name and synonyms
                # if mondo_id not in self.disease_names or not self.disease_names[mondo_id]:
                #     self.disease_names[mondo_id] = disease_name
                #     synonyms_without_main = [s for s in all_synonyms if s != disease_name]
                #     self.disease_mapping_with_synonyms[mondo_id] = [disease_name] + synonyms_without_main
                # else:
                #     existing_synonyms = self.disease_mapping_with_synonyms.get(mondo_id, [])
                #     existing_set = set(existing_synonyms)
                #     for synonym in all_synonyms:
                #         if synonym not in existing_set:
                #             existing_synonyms.append(synonym)
                #             existing_set.add(synonym)
                #     self.disease_mapping_with_synonyms[mondo_id] = existing_synonyms

                # # record MONDO rare-disease flag
                # if is_rare:
                #     self.rare_disease_types[mondo_id] = 'rare_disease'
                
                # handle Orphanet and OMIM in equivalent_ids
                for equiv_id in equivalent_ids:
                    if equiv_id.startswith('Orphanet:'):
                        orphanet_id = equiv_id.replace('Orphanet:', 'ORPHA:')
                        # key by Orphanet ID, record name and synonyms
                        if orphanet_id not in self.disease_names or not self.disease_names[orphanet_id]:
                            self.disease_names[orphanet_id] = disease_name
                            # ensure name not duplicated in synonyms
                            synonyms_without_main = [s for s in all_synonyms if s != disease_name]
                            self.disease_mapping_with_synonyms[orphanet_id] = [disease_name] + synonyms_without_main
                            orphanet_count += 1
                        else:
                            # if exists, add new synonym (dedup)
                            existing_synonyms = self.disease_mapping_with_synonyms.get(orphanet_id, [])
                            existing_set = set(existing_synonyms)
                            for synonym in all_synonyms:
                                if synonym not in existing_set:
                                    existing_synonyms.append(synonym)
                                    existing_set.add(synonym)
                            self.disease_mapping_with_synonyms[orphanet_id] = existing_synonyms
                        
                        # add rare-disease flag to separate dict
                        if is_rare:
                            self.rare_disease_types[orphanet_id] = 'rare_disease'
                    
                    elif equiv_id.startswith('OMIM:'):
                        omim_id = equiv_id
                        # key by OMIM ID, record name and synonyms
                        if omim_id not in self.disease_names or not self.disease_names[omim_id]:
                            self.disease_names[omim_id] = disease_name
                            # ensure name not duplicated in synonyms
                            synonyms_without_main = [s for s in all_synonyms if s != disease_name]
                            self.disease_mapping_with_synonyms[omim_id] = [disease_name] + synonyms_without_main
                            omim_count += 1
                        else:
                            # if exists, add new synonym (dedup)
                            existing_synonyms = self.disease_mapping_with_synonyms.get(omim_id, [])
                            existing_set = set(existing_synonyms)
                            for synonym in all_synonyms:
                                if synonym not in existing_set:
                                    existing_synonyms.append(synonym)
                                    existing_set.add(synonym)
                            self.disease_mapping_with_synonyms[omim_id] = existing_synonyms
                        
                        # add rare-disease flag to separate dict
                        if is_rare:
                            self.rare_disease_types[omim_id] = 'rare_disease'
            
            print(f"Added {orphanet_count} Orphanet disease mappings from mondo_parsed_full.json")
            print(f"Added {omim_count} OMIM disease mappings from mondo_parsed_full.json")
            
        except Exception as e:
            print(f"Warning: Failed to load mondo_parsed_full.json: {e}")
    
    def _expand_rare_disease_types_to_synonyms(self):
        """Extend rare-disease type to same-name diseases per disease_name_to_ids."""
        if not hasattr(self, 'rare_disease_types') or not self.rare_disease_types or not self.disease_name_to_ids:
            return
        
        # for each name, extend existing rare-disease type to all same-name IDs
        for disease_name_lower, synonym_ids in self.disease_name_to_ids.items():
            if not synonym_ids:
                continue

            # find IDs under this name with rare-disease type
            rare_ids = [synonym_id for synonym_id in synonym_ids if synonym_id in self.rare_disease_types]
            if not rare_ids:
                continue

            # merge and dedup rare-disease types
            aggregated_values = []
            for synonym_id in rare_ids:
                value = self.rare_disease_types[synonym_id]
                if isinstance(value, list):
                    aggregated_values.extend(value)
                else:
                    aggregated_values.append(value)

            # dict.fromkeys for ordered dedup
            rare_values = list(dict.fromkeys(aggregated_values))
            rare_value_str = ';'.join(rare_values)

            # set same rare-disease type (as list) for all IDs under name
            for synonym_id in synonym_ids:
                self.rare_disease_types[synonym_id] = rare_value_str
     
    def _build_disease_name_to_ids_mapping(self):
        """Build case-insensitive name->disease_id mapping; key=standard name, include IDs with that name or alias."""
        print("Building disease name to IDs mapping...")
        
        # clear existing mapping
        self.disease_name_to_ids.clear()
        
        # step 1: key by standard name, add IDs for standard name
        for disease_id, disease_name in self.disease_names.items():
            if disease_name:  # Only add if disease name is not None or empty
                disease_name_lower = disease_name.lower().strip()
                self.disease_name_to_ids[disease_name_lower].add(disease_id)
        
        # step 2: connect IDs via synonym intersection
        # first build synonym -> disease_id map
        synonym_to_disease_ids = defaultdict(set)
        for disease_id, synonyms in self.disease_mapping_with_synonyms.items():
            for synonym in synonyms:
                synonym_lower = synonym.lower().strip()
                if not synonym_lower:
                    continue
                synonym_to_disease_ids[synonym_lower].add(disease_id)

        # for synonym with multiple IDs, add those IDs to each other's standard-name set
        for related_ids in synonym_to_disease_ids.values():
            if len(related_ids) <= 1:
                continue

            cluster_ids = set(related_ids)
            for disease_id in cluster_ids:
                disease_name = self.disease_names.get(disease_id)
                if not disease_name:
                    continue

                disease_name_lower = disease_name.lower().strip()
                if not disease_name_lower:
                    continue

                self.disease_name_to_ids[disease_name_lower].update(cluster_ids)
        
        # step 3: remove duplicate mappings, keep one representative name
        signature_to_name = {}
        normalized_items: List[Tuple[str, Set[str]]] = []
        duplicate_name_count = 0

        for name, ids in self.disease_name_to_ids.items():
            if not ids:
                continue

            signature = frozenset(ids)
            if signature in signature_to_name:
                duplicate_name_count += 1
                continue

            signature_to_name[signature] = name
            normalized_items.append((name, set(ids)))

        if duplicate_name_count:
            print(f"Removed {duplicate_name_count} duplicate disease-name mappings with identical ID sets")

        # step 4: merge names whose ID sets intersect, in order
        normalized_pairs: List[Tuple[str, Set[str]]] = [
            (name, set(ids)) for name, ids in normalized_items
        ]

        merged_results: List[Tuple[str, Set[str]]] = []

        for i in range(len(normalized_pairs)):
            current_name, current_ids = normalized_pairs[i]
            merged_into = False

            for j in range(i + 1, len(normalized_pairs)):
                target_name, target_ids = normalized_pairs[j]
                if current_ids.intersection(target_ids):
                    target_ids.update(current_ids)
                    # print(
                    #     "Merged disease-name overlap: "
                    #     f"'{current_name}' -> '{target_name}'"
                    # )
                    merged_into = True
                    break

            if not merged_into:
                merged_results.append((current_name, current_ids))

        # keep mapping with deduped name set
        self.disease_name_to_ids = {
            name: ids for name, ids in merged_results
        }
        
        # check if name->id mappings have intersection
        for name, ids in self.disease_name_to_ids.items():
            for other_name, other_ids in self.disease_name_to_ids.items():
                if name != other_name and ids.intersection(other_ids):
                    print(f"Warning: Disease name {name} and {other_name} have overlapping disease IDs")
        
        # count name->id mappings
        # if self.disease_name_to_ids:
        #     longest_name, longest_ids = max(
        #         self.disease_name_to_ids.items(),
        #         key=lambda item: len(item[1])
        #     )
        #     print(
        #         "Max disease IDs sharing a name: "
        #         f"{len(longest_ids)} for '{longest_name}'"
        #     )
        # else:
        #     print("No disease name to IDs mappings built.")
        
        print(f"Built disease name to IDs mapping for {len(self.disease_name_to_ids)} unique disease names")
        
        # for name, ids in self.disease_name_to_ids.items():
        #     if name == 'mccune-albright syndrome':
        #         print(f"  {name}: {ids}")
        
        ## DEBUG
        # # Print detailed statistics about disease name duplicates
        # duplicate_count = sum(1 for ids in self.disease_name_to_ids.values() if len(ids) > 1)
        # total_duplicate_relationships = sum(len(ids) - 1 for ids in self.disease_name_to_ids.values() if len(ids) > 1)
        
        # if duplicate_count > 0:
        #     print(f"Found {duplicate_count} disease names with multiple disease IDs")
        #     print(f"Total duplicate relationships: {total_duplicate_relationships}")
            
        #     # Show detailed examples
        #     examples = [(name, list(ids)) for name, ids in self.disease_name_to_ids.items() if len(ids) > 1][:5]
        #     for i, (name, ids) in enumerate(examples, 1):
        #         print(f"  Example {i}: '{name}' -> {ids}")
        #         # Show the actual disease names for verification
        #         actual_names = [self.disease_names.get(did, did) for did in ids]
        #         print(f"    Actual names: {actual_names}")
        
        # # Verify that all relationships are captured
        # total_relationships = sum(len(ids) for ids in self.disease_name_to_ids.values())
        # print(f"Total disease name to ID relationships: {total_relationships}")
        # print(f"Total disease names loaded: {len(self.disease_names)}")
        
        # # Additional verification: check if any disease names are missing
        # missing_names = 0
        # for disease_id, disease_name in self.disease_names.items():
        #     if disease_name and disease_name.strip():
        #         disease_name_lower = disease_name.lower().strip()
        #         if disease_id not in self.disease_name_to_ids.get(disease_name_lower, set()):
        #             missing_names += 1
        #             if missing_names <= 3:  # Show first few examples
        #                 print(f"  WARNING: Missing relationship for {disease_id} -> '{disease_name}'")
        
        # if missing_names > 0:
        #     print(f"  Total missing relationships: {missing_names}")
        # else:
        #     print("  All disease name relationships correctly captured!")
    
    def load_phenotype_names_from_obo(self, obo_file_path: str):
        """Load phenotype names from HPO OBO file"""
        print("Loading phenotype names from HPO OBO file...")
        
        # First pass: collect all non-obsolete terms and their names
        all_terms = []
        current_term = {}
        
        try:
            with open(obo_file_path, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    
                    if line == '[Term]':
                        # Save previous term if we have one
                        if current_term:
                            all_terms.append(current_term)
                        
                        # Start new term
                        current_term = {}
                        
                    elif line.startswith('id: HP:'):
                        current_term['id'] = line.split(': ')[1]
                        
                    elif line.startswith('alt_id: HP:'):
                        if 'alt_ids' not in current_term:
                            current_term['alt_ids'] = []
                        current_term['alt_ids'].append(line.split(': ')[1])
                        
                    elif line.startswith('name: '):
                        current_term['name'] = line.split(': ', 1)[1]
                        
                    elif line == 'is_obsolete: true':
                        current_term['is_obsolete'] = True
                        
                    elif line.startswith('replaced_by: '):
                        current_term['replaced_by'] = line.split(': ', 1)[1]
                        
                    elif line.startswith('synonym: '):
                        if 'synonyms' not in current_term:
                            current_term['synonyms'] = []
                        # extract content inside double quotes
                        synonym_content = line.split(': ', 1)[1]
                        if '"' in synonym_content:
                            # find first and last double-quote positions
                            start_quote = synonym_content.find('"')
                            end_quote = synonym_content.rfind('"')
                            if start_quote != -1 and end_quote != -1 and start_quote < end_quote:
                                current_term['synonyms'].append(synonym_content[start_quote + 1:end_quote])
                            else:
                                current_term['synonyms'].append(synonym_content)
                        else:
                            current_term['synonyms'].append(synonym_content)
                        
                    elif line.startswith('def: '):
                        # extract content inside double quotes
                        def_content = line.split(': ', 1)[1]
                        if '"' in def_content:
                            # find first and last double-quote positions
                            start_quote = def_content.find('"')
                            end_quote = def_content.rfind('"')
                            if start_quote != -1 and end_quote != -1 and start_quote < end_quote:
                                current_term['def'] = def_content[start_quote + 1:end_quote]
                            else:
                                current_term['def'] = def_content
                        else:
                            current_term['def'] = def_content
                        
                    elif line.startswith('comment: '):
                        current_term['comment'] = line.split(': ', 1)[1]
                        
                    elif line.startswith('is_a: '):
                        if 'is_a' not in current_term:
                            current_term['is_a'] = []
                        # Parse is_a line: "is_a: HP:0001252 ! Hypotonia"
                        is_a_content = line.split(': ', 1)[1]
                        if ' ! ' in is_a_content:
                            # Extract only the HPO ID before "!"
                            parent_id = is_a_content.split(' ! ', 1)[0]
                            current_term['is_a'].append(parent_id)
                        else:
                            # If no name provided, use the HPO ID
                            current_term['is_a'].append(is_a_content)
            
            # Don't forget the last term
            if current_term:
                all_terms.append(current_term)
            
            # Build complete ID to name mapping for non-obsolete terms
            id_to_name = {}
            for term in all_terms:
                if not term.get('is_obsolete', False):
                    # Add primary ID
                    if 'id' in term and 'name' in term:
                        id_to_name[term['id']] = term['name']
                    
                    # Add alternative IDs
                    if 'alt_ids' in term and 'name' in term:
                        for alt_id in term['alt_ids']:
                            id_to_name[alt_id] = term['name']
            
            # Second pass: handle obsolete terms with complete replacement name mapping
            for term in all_terms:
                if term.get('is_obsolete', False) and 'name' in term:

                    if 'replaced_by' in term:
                        replacement_id = term['replaced_by']
                        replacement_name = id_to_name.get(replacement_id, replacement_id)
                        combined_name = f"{replacement_name};{term['name']}"
                    else:
                        combined_name = term['name']
                    
                    # Add obsolete term with combined name
                    if 'id' in term:
                        id_to_name[term['id']] = combined_name
                    
                    # Add alternative IDs for obsolete terms
                    if 'alt_ids' in term:
                        for alt_id in term['alt_ids']:
                            id_to_name[alt_id] = combined_name
            
            # Update phenotype names and additional information
            for hpo_id, phenotype_name in id_to_name.items():
                self.phenotype_names[hpo_id] = phenotype_name
                
                # Find the original term for this ID to get additional information
                original_term = None
                for term in all_terms:
                    if term.get('id') == hpo_id or (term.get('alt_ids') and hpo_id in term['alt_ids']):
                        original_term = term
                        break
                
                if original_term:
                    # Store synonyms
                    if 'synonyms' in original_term:
                        self.hpo_synonyms[hpo_id] = original_term['synonyms']
                    
                    # Store definition
                    if 'def' in original_term:
                        self.hpo_definitions[hpo_id] = original_term['def']
                    
                    # Store comment
                    if 'comment' in original_term:
                        self.hpo_comments[hpo_id] = original_term['comment']
                    
                    # Store is_a relationships
                    if 'is_a' in original_term:
                        if hpo_id not in self.hpo_is_a:
                            self.hpo_is_a[hpo_id] = []
                        for is_a_id in original_term['is_a']:
                            if is_a_id not in self.hpo_is_a[hpo_id]:
                                self.hpo_is_a[hpo_id].append(is_a_id)
                        # self.hpo_is_a[hpo_id] = original_term['is_a']                        

                    # Store alt_ids mapping (only for primary IDs, not alt_ids themselves)
                    if 'alt_ids' in original_term and original_term.get('id') == hpo_id:
                        self.hpo_alt_ids[hpo_id] = original_term['alt_ids']
            
            # if phenotype has no is_a, try replaced_by and use replaced phenotype's is_a
            for hpo_id, phenotype_name in id_to_name.items():
                if hpo_id not in self.hpo_is_a.keys():
                    # find original term in all_terms
                    original_term = None
                    for term in all_terms:
                        if term.get('id') == hpo_id or (term.get('alt_ids') and hpo_id in term['alt_ids']):
                            original_term = term
                            break
                    
                    # if found and has replaced_by
                    if original_term and 'replaced_by' in original_term:
                        replacement_id = original_term['replaced_by']
                        # get replaced phenotype's is_a
                        if replacement_id in self.hpo_is_a:
                            # copy replaced is_a to current
                            self.hpo_is_a[hpo_id] = self.hpo_is_a[replacement_id].copy()

            print(f"Loaded {len(self.phenotype_names)} phenotype names from OBO file")
            print(f"Loaded {len(self.hpo_synonyms)} HPO terms with synonyms")
            print(f"Loaded {len(self.hpo_definitions)} HPO terms with definitions")
            print(f"Loaded {len(self.hpo_comments)} HPO terms with comments")
            print(f"Loaded {len(self.hpo_is_a)} HPO terms with is_a relationships")
            
            # Build parent-to-children mapping for fast lookup
            print("Building parent-to-children mapping...")
            self._build_parent_to_children_mapping()
            
        except Exception as e:
            print(f"Error loading OBO file: {e}")
            print("Will use HPO IDs as phenotype names")
    
    def _build_parent_to_children_mapping(self):
        """Build a mapping from parent phenotypes to their children for fast lookup"""
        self.parent_to_children = {}
        
        # Build a mapping from parent to all children
        for child_phenotype, parent_nodes in self.hpo_is_a.items():
            for parent in parent_nodes:
                if parent not in self.parent_to_children:
                    self.parent_to_children[parent] = []
                self.parent_to_children[parent].append(child_phenotype)
        
        # Count statistics
        single_child_count = sum(1 for children in self.parent_to_children.values() if len(children) == 1)
        print(f"Built parent-to-children mapping: {len(self.parent_to_children)} parents, {single_child_count} with single child")
    
    def _get_all_descendants(self, phenotype_id: str, visited: Optional[Set[str]] = None) -> Set[str]:
        """
        Recursively get all descendant phenotypes (children, grandchildren, etc.) of a given phenotype.
        
        Args:
            phenotype_id: The phenotype ID to get descendants for
            visited: Set to track visited nodes to avoid cycles (optional)
            
        Returns:
            Set of all descendant phenotype IDs
        """
        if visited is None:
            visited = set()
        
        if phenotype_id in visited:
            return set()
        
        visited.add(phenotype_id)
        descendants = set()
        
        # Get direct children
        children = self.parent_to_children.get(phenotype_id, [])
        for child in children:
            if child not in visited:
                descendants.add(child)
                # Recursively get descendants of children
                descendants.update(self._get_all_descendants(child, visited))
        
        return descendants
    
    def _get_all_ancestors(self, phenotype_id: str, visited: Optional[Set[str]] = None) -> Set[str]:
        """
        Recursively get all ancestor phenotypes (parents, grandparents, etc.) of a given phenotype.
        
        Args:
            phenotype_id: The phenotype ID to get ancestors for
            visited: Set to track visited nodes to avoid cycles (optional)
            
        Returns:
            Set of all ancestor phenotype IDs
        """
        if visited is None:
            visited = set()
        
        if phenotype_id in visited:
            return set()
        
        visited.add(phenotype_id)
        ancestors = set()
        
        # Get direct parents
        parents = self.hpo_is_a.get(phenotype_id, [])
        for parent in parents:
            if parent not in visited:
                ancestors.add(parent)
                # Recursively get ancestors of parents
                ancestors.update(self._get_all_ancestors(parent, visited))
        
        return ancestors
    
    def get_disease_name(self, disease_id: str) -> str:
        """Get disease name from disease ID"""
        return self.disease_names.get(disease_id, disease_id)
    
    def get_disease_ids_by_name(self, disease_name: str) -> List[str]:
        """Get all disease IDs that have the given disease name (case-insensitive)"""
        disease_name_lower = disease_name.lower().strip()
        return list(self.disease_name_to_ids.get(disease_name_lower, set()))
    
    def get_disease_synonyms(self, disease_id: str) -> List[str]:
        """Get synonyms for a disease from disease ID"""
        synonyms = self.disease_mapping_with_synonyms.get(disease_id, [])
        if not synonyms and disease_id in self.disease_names:
            return ""
        # dedup, keep order (case-insensitive)
        seen_lower = {}
        result = []
        for name in synonyms:
            name_lower = name.lower()
            if name_lower not in seen_lower:
                seen_lower[name_lower] = name
                result.append(name)
        return result
    
    def get_disease_all_names(self, disease_id: str) -> List[str]:
        """Get all names (including synonyms) for a disease from disease ID"""
        synonyms = self.disease_mapping_with_synonyms.get(disease_id, [])
        if not synonyms and disease_id in self.disease_names:
            return [self.disease_names[disease_id]]
        # dedup, keep order (case-insensitive)
        seen_lower = {}
        result = []
        for name in synonyms:
            name_lower = name.lower()
            if name_lower not in seen_lower:
                seen_lower[name_lower] = name
                result.append(name)
        return result
        
    def validate_disease_names_loading(self):
        """Validate that disease names are properly loaded from both sources"""
        print("Validating disease names loading...")
        
        # Count diseases from different sources
        total_diseases = len(self.disease_names)
        diseases_with_names = sum(1 for name in self.disease_names.values() if name and name.strip())
        diseases_without_names = total_diseases - diseases_with_names
        
        print(f"Total disease IDs: {total_diseases}")
        print(f"Diseases with names: {diseases_with_names}")
        print(f"Diseases without names: {diseases_without_names}")
        
        # Check for diseases from different databases
        omim_count = sum(1 for did in self.disease_names.keys() if did.startswith('OMIM:'))
        orpha_count = sum(1 for did in self.disease_names.keys() if did.startswith('ORPHA:'))
        decipher_count = sum(1 for did in self.disease_names.keys() if did.startswith('DECIPHER:'))
        
        print(f"OMIM diseases: {omim_count}")
        print(f"ORPHA diseases: {orpha_count}")
        print(f"DECIPHER diseases: {decipher_count}")
        
        # Show some examples of diseases without names
        if diseases_without_names > 0:
            print(f"Examples of diseases without names:")
            examples = [did for did, name in self.disease_names.items() if not name or not name.strip()][:5]
            for did in examples:
                print(f"  {did}")
        
        return {
            'total_diseases': total_diseases,
            'diseases_with_names': diseases_with_names,
            'diseases_without_names': diseases_without_names,
            'omim_count': omim_count,
            'orpha_count': orpha_count,
            'decipher_count': decipher_count
        }
    
    def get_phenotype_name(self, hpo_id: str) -> str:
        """Get phenotype name from HPO ID"""
        return self.phenotype_names.get(hpo_id, hpo_id)
    
    def get_phenotype_synonyms(self, hpo_id: str) -> List[str]:
        """Get synonyms for a phenotype from HPO ID"""
        return self.hpo_synonyms.get(hpo_id, [])
    
    def get_phenotype_definition(self, hpo_id: str) -> str:
        """Get definition for a phenotype from HPO ID"""
        return self.hpo_definitions.get(hpo_id, "")
    
    def get_phenotype_comment(self, hpo_id: str) -> str:
        """Get comment for a phenotype from HPO ID"""
        return self.hpo_comments.get(hpo_id, "")
    
    def get_phenotype_is_a(self, hpo_id: str) -> List[str]:
        """Get is_a relationships for a phenotype from HPO ID"""
        return self.hpo_is_a.get(hpo_id, [])
    
    def get_phenotype_is_a_names(self, hpo_id: str) -> List[str]:
        """Get is_a relationship names for a phenotype from HPO ID"""
        is_a_ids = self.get_phenotype_is_a(hpo_id)
        is_a_names = []
        for parent_id in is_a_ids:
            parent_name = self.get_phenotype_name(parent_id)
            if parent_name and parent_name != parent_id:
                is_a_names.append(parent_name)
            else:
                is_a_names.append(parent_id)
        return is_a_names
    
    def get_phenotype_abnormal_category_from_kg(self, hpo_id: str) -> List[str]:
        """
        Get all phenotype abnormal categories for a given HPO ID from KG.
        Returns all direct children of HP:0000118 (Phenotypic abnormality) that the phenotype belongs to.
        All data is retrieved from knowledge graph (KG), not from local data.
        
        Args:
            hpo_id (str): The HPO ID to find the categories for
            
        Returns:
            List[str]: List of HPO IDs of the abnormal categories, or empty list if not found
        """
        # Direct children of HP:0000118 (Phenotypic abnormality)
        hp_0000118_subcategories = {
            'HP:0000119',  # Abnormality of the genitourinary system
            'HP:0000152',  # Abnormality of head or neck
            'HP:0000478',  # Abnormality of the eye
            'HP:0000598',  # Abnormality of the ear
            'HP:0000707',  # Abnormality of the nervous system
            'HP:0000769',  # Abnormality of the breast
            'HP:0000818',  # Abnormality of the endocrine system
            'HP:0001197',  # Abnormality of prenatal development or birth
            'HP:0001507',  # Growth abnormality
            'HP:0001574',  # Abnormality of the integument
            'HP:0001608',  # Abnormality of the voice
            'HP:0001626',  # Abnormality of the cardiovascular system
            'HP:0001871',  # Abnormality of blood and blood-forming tissues
            'HP:0001939',  # Abnormality of metabolism/homeostasis
            'HP:0002086',  # Abnormality of the respiratory system
            'HP:0002664',  # Neoplasm
            'HP:0002715',  # Abnormality of the immune system
            'HP:0025031',  # Abnormality of the digestive system
            'HP:0025142',  # Constitutional symptom
            'HP:0025354',  # Abnormal cellular phenotype
            'HP:0033127',  # Abnormality of the musculoskeletal system
            'HP:0040064',  # Abnormality of limbs
            'HP:0045027'   # Abnormality of the thoracic cavity
        }
        
        # Use BFS to traverse up the hierarchy and collect all categories from KG
        queue = [hpo_id]
        visited = set()
        found_categories = []
        
        while queue:
            current_id = queue.pop(0)
            if current_id in visited:
                continue
            visited.add(current_id)
            
            # Check if current phenotype is one of the direct subcategories
            if current_id in hp_0000118_subcategories:
                found_categories.append(current_id)
                # Continue searching for other categories instead of returning immediately
            
            # Get parent phenotypes from KG and add them to queue
            # Use get_phenotype_ancestors_from_kg with max_depth=1 to get direct parents only
            try:
                parents = self.get_phenotype_ancestors_from_kg(current_id, max_depth=1)
                for parent in parents:
                    if parent not in visited:
                        queue.append(parent)
            except Exception as e:
                print(f"Error getting parents from KG for {current_id}: {e}")
                continue
        
        # Return all found categories
        return found_categories
    
    def get_phenotype_abnormal_category(self, hpo_id: str) -> List[str]:
        """
        Get all phenotype abnormal categories for a given HPO ID.
        Returns all direct children of HP:0000118 (Phenotypic abnormality) that the phenotype belongs to.
        
        Args:
            hpo_id (str): The HPO ID to find the categories for
            
        Returns:
            List[str]: List of HPO IDs of the abnormal categories, or empty list if not found
        """
        # Direct children of HP:0000118 (Phenotypic abnormality)
        hp_0000118_subcategories = {
            'HP:0000119',  # Abnormality of the genitourinary system
            'HP:0000152',  # Abnormality of head or neck
            'HP:0000478',  # Abnormality of the eye
            'HP:0000598',  # Abnormality of the ear
            'HP:0000707',  # Abnormality of the nervous system
            'HP:0000769',  # Abnormality of the breast
            'HP:0000818',  # Abnormality of the endocrine system
            'HP:0001197',  # Abnormality of prenatal development or birth
            'HP:0001507',  # Growth abnormality
            'HP:0001574',  # Abnormality of the integument
            'HP:0001608',  # Abnormality of the voice
            'HP:0001626',  # Abnormality of the cardiovascular system
            'HP:0001871',  # Abnormality of blood and blood-forming tissues
            'HP:0001939',  # Abnormality of metabolism/homeostasis
            'HP:0002086',  # Abnormality of the respiratory system
            'HP:0002664',  # Neoplasm
            'HP:0002715',  # Abnormality of the immune system
            'HP:0025031',  # Abnormality of the digestive system
            'HP:0025142',  # Constitutional symptom
            'HP:0025354',  # Abnormal cellular phenotype
            'HP:0033127',  # Abnormality of the musculoskeletal system
            'HP:0040064',  # Abnormality of limbs
            'HP:0045027'   # Abnormality of the thoracic cavity
        }
        
        # Use BFS to traverse up the hierarchy and collect all categories
        queue = [hpo_id]
        visited = set()
        found_categories = []
        
        while queue:
            current_id = queue.pop(0)
            if current_id in visited:
                continue
            visited.add(current_id)
            
            # Check if current phenotype is one of the direct subcategories
            if current_id in hp_0000118_subcategories:
                found_categories.append(current_id)
                # Continue searching for other categories instead of returning immediately
            
            # Get parent phenotypes and add them to queue
            parents = self.get_phenotype_is_a(current_id)
            for parent in parents:
                if parent not in visited:
                    queue.append(parent)
        
        # Return all found categories
        return found_categories
    
    def get_phenotype_full_info(self, hpo_id: str) -> Dict:
        """Get complete information for a phenotype from HPO ID"""
        return {
            'id': hpo_id,
            'name': self.get_phenotype_name(hpo_id),
            'synonyms': self.get_phenotype_synonyms(hpo_id),
            'definition': self.get_phenotype_definition(hpo_id),
            'comment': self.get_phenotype_comment(hpo_id),
            'is_a': self.get_phenotype_is_a(hpo_id)
        }
    
    def _categorize_frequency(self, frequency: str) -> str:
        """
        Categorize frequency values into types.
        
        Args:
            frequency: Frequency value from HPOA file
            
        Returns:
            Category string: 'fraction', 'percentage', 'hpo_id', 'empty', or 'other'
        """
        import re
        
        frequency = frequency.strip()
        
        if not frequency:
            return 'empty'
        elif re.match(r'^\d+/\d+$', frequency):
            return 'fraction'
        elif re.match(r'^\d+(\.\d+)?%$', frequency):
            return 'percentage'
        elif re.match(r'^HP:\d+$', frequency):
            return 'hpo_id'
        else:
            return 'other'
    
    def _convert_hpo_frequency_to_description(self, hpo_id: str) -> str:
        """
        Convert HPO frequency ID to human-readable description.
        
        Args:
            hpo_id: HPO ID for frequency (e.g., HP:0040281) or multiple IDs separated by '; '
            
        Returns:
            Human-readable frequency description
        """
        # Handle multiple HPO IDs separated by '; '
        if '; ' in hpo_id:
            hpo_ids = hpo_id.split('; ')
            descriptions = []
            for single_hpo_id in hpo_ids:
                description = self.hpo_frequency_descriptions.get(single_hpo_id.strip(), single_hpo_id.strip())
                if description not in descriptions:  # avoid duplicate
                    descriptions.append(description)
            return '; '.join(descriptions)
        else:
            return self.hpo_frequency_descriptions.get(hpo_id, hpo_id)  # Return original ID if not found
    
    def _format_single_frequency_value(self, frequency_value: str, frequency_type: str) -> str:
        """Format a single frequency value into human-readable text."""
        if not frequency_value:
            return ""

        frequency_value = frequency_value.strip()
        if not frequency_value:
            return ""

        freq_type = frequency_type.strip() if frequency_type else ''
        if not freq_type:
            freq_type = self._categorize_frequency(frequency_value)

        if freq_type == 'hpo_id':
            return self._convert_hpo_frequency_to_description(frequency_value)

        return frequency_value

    def _format_frequency_display(self, frequency: str, frequency_type: str) -> str:
        """Format frequency information (value + type) into display text."""
        if not frequency:
            return ""

        values = [item.strip() for item in frequency.split(';') if item.strip()]
        types = [item.strip() for item in frequency_type.split(';')] if frequency_type else []

        displays = []
        for index, value in enumerate(values):
            freq_type = types[index] if index < len(types) else ''
            display_value = self._format_single_frequency_value(value, freq_type)
            if display_value:
                displays.append(display_value)

        # Remove duplicates while preserving order
        seen = set()
        unique_displays = []
        for item in displays:
            if item not in seen:
                unique_displays.append(item)
                seen.add(item)

        return '; '.join(unique_displays)

    def _build_case_database_entries(self) -> Dict[str, Dict]:
        """Build disease database keyed by disease_name_to_ids keys with strict consistency checks."""

        def _collect_aliases(disease_id: str, primary_name: str) -> Tuple[str, ...]:
            synonyms = self.disease_mapping_with_synonyms.get(disease_id, []) or []

            if not synonyms and primary_name:
                return (primary_name,)

            return tuple(synonyms)

        def _collect_phenotype_map(disease_id: str) -> Dict[str, Tuple[str, ...]]:
            phenotype_ids = self.disease_to_phenotypes.get(disease_id, set()) or set()
            freq_dict = self.disease_phenotype_frequency.get(disease_id, {})
            phenotype_map: Dict[str, Tuple[str, ...]] = {}

            for phenotype_id in phenotype_ids:
                freq_info = freq_dict.get(phenotype_id) or self.phenotype_disease_frequency.get((phenotype_id, disease_id))

                frequencies: List[str] = []
                if freq_info:
                    display_text = self._format_frequency_display(freq_info.get('frequency', ''), freq_info.get('frequency_type', ''))
                    if display_text:
                        frequencies = [fragment.strip() for fragment in display_text.split(';') if fragment.strip()]

                if not frequencies:
                    frequencies = ["Unknown"]

                phenotype_map[phenotype_id] = tuple(sorted(set(frequencies)))

            return phenotype_map

        # Ensure every disease ID is accounted for in the name-to-IDs mapping
        name_to_ids: Dict[str, Set[str]] = {name: set(ids) for name, ids in self.disease_name_to_ids.items()}
        disease_id_to_key: Dict[str, str] = {}

        for name_key, ids in name_to_ids.items():
            for disease_id in ids:
                disease_id_to_key[disease_id] = name_key

        for disease_id in self.disease_to_phenotypes.keys():
            if disease_id in disease_id_to_key:
                continue

            standard_name = (self.disease_names.get(disease_id, "") or disease_id).strip()
            fallback_key = standard_name.lower() if standard_name else disease_id.lower()
            name_to_ids.setdefault(fallback_key, set()).add(disease_id)
            disease_id_to_key[disease_id] = fallback_key

        case_database: Dict[str, Dict] = {}

        disease_phenotype_pairs = 0

        for name_key, disease_ids in name_to_ids.items():
            if not disease_ids:
                continue

            sorted_ids = sorted(disease_ids)

            # Collect standard names per disease ID
            standard_names = {disease_id: (self.disease_names.get(disease_id, "") or "").strip() for disease_id in sorted_ids}

            # Collect aliases and ensure consistency
            alias_sets = [
                _collect_aliases(disease_id, standard_names[disease_id])
                for disease_id in sorted_ids
            ]

            alias_reference = alias_sets[0] if alias_sets else tuple()
            alias_union = list(alias_reference)
            alias_errors = []

            for alias_set in alias_sets[1:]:
                for alias_value in alias_set:
                    if alias_value not in alias_union:
                        alias_union.append(alias_value)
                if alias_set != alias_reference:
                    alias_errors.append(
                        f"alias mismatch for diseases {sorted_ids}: reference={list(alias_reference)}, current={list(alias_set)}"
                    )

            aliases_sorted = alias_union

            # Collect phenotypes and ensure consistency
            phenotype_maps = [_collect_phenotype_map(disease_id) for disease_id in sorted_ids]
            phenotype_reference = phenotype_maps[0] if phenotype_maps else {}
            phenotype_union: Dict[str, Set[Tuple[str, ...]]] = {}
            phenotype_errors = []

            for phenotype_id, frequencies in phenotype_reference.items():
                phenotype_union[phenotype_id] = {frequencies}

            for phenotype_map in phenotype_maps[1:]:
                if phenotype_map != phenotype_reference:
                    phenotype_errors.append(
                        f"phenotype mismatch for diseases {sorted_ids}: reference={phenotype_reference}, current={phenotype_map}"
                    )

                for phenotype_id, frequencies in phenotype_map.items():
                    phenotype_union.setdefault(phenotype_id, set()).add(frequencies)

            phenotypes_output = {}
            for phenotype_id, freq_sets in phenotype_union.items():
                combined_freqs: Set[str] = set()
                for freq_tuple in freq_sets:
                    combined_freqs.update(freq_tuple)
                phenotypes_output[phenotype_id] = sorted(combined_freqs)

            unique_descriptions = []
            seen_descriptions = set()
            for disease_id in sorted_ids:
                description = self.disease_descriptions.get(disease_id, "").strip()
                if not description:
                    description = ""
                if description not in seen_descriptions:
                    seen_descriptions.add(description)
                    unique_descriptions.append(description)

            is_rare = False
            for disease_id in sorted_ids:
                if self.rare_disease_types.get(disease_id, False):
                    is_rare = True
                    break
            
            disease_types = ""
            for disease_id in sorted_ids:
                disease_type = self.disease_types.get(disease_id, False)
                if disease_type:
                    break
                else:
                    disease_type = ""

            entry = {
                "standard_names": standard_names,
                "aliases": aliases_sorted,
                "disease_type": disease_type,
                "phenotypes": dict(sorted(phenotypes_output.items())),
                "is_rare": is_rare,
                "description": unique_descriptions
            }

            disease_phenotype_pairs += len(phenotypes_output)

            if alias_errors:
                entry["alias_errors"] = alias_errors
            if phenotype_errors:
                entry["phenotype_errors"] = phenotype_errors

            case_database[name_key] = entry

        print(f"disease_phenotype_pairs: {disease_phenotype_pairs}")

        return case_database

    def _export_graph_data(self, case_database_entries: Dict[str, Dict]) -> Dict[str, str]:
        """
        Export graph CSV files for downstream knowledge graph usage.
        Returns a mapping of artifact names to their output paths.
        """
        graph_cfg = self.config.get("graph_output", {}) if self.config else {}
        output_dir = graph_cfg.get("output_dir")
        if not output_dir:
            base_path = (self.config or {}).get("base_path")
            if base_path:
                output_dir = os.path.join(base_path, "disease_phenotype_kg")
            else:
                output_dir = "disease_phenotype_kg"
        os.makedirs(output_dir, exist_ok=True)

        def _path(filename: str) -> str:
            return os.path.join(output_dir, filename) if output_dir else filename

        disease_nodes_path = _path(graph_cfg.get("disease_nodes_file", "disease_nodes.csv"))
        publicDisease_nodes_path = _path(graph_cfg.get("publicDisease_nodes_file", "publicDisease_nodes.csv"))
        phenotype_nodes_path = _path(graph_cfg.get("phenotype_nodes_file", "phenotype_nodes.csv"))
        disease_edge_path = _path(graph_cfg.get("disease_to_publicDisease_edges_file", "disease_to_publicDisease_edges.csv"))
        phenotype_edge_path = _path(graph_cfg.get("phenotype_to_phenotype_edges_file", "phenotype_to_phenotype_edges.csv"))
        disease_phenotype_edge_path = _path(graph_cfg.get("disease_phenotype_edges_file", "disease_to_phenotype_edges.csv"))

        # Build aggregated disease IDs (D:X)
        aggregated_map: Dict[str, List[str]] = {}
        disease_id_to_group: Dict[str, str] = {}
        disease_nodes_rows = []
        recomputed_ic: Dict[str, float] = {}

        for idx, (name_key, entry) in enumerate(sorted(case_database_entries.items(), key=lambda item: item[0])):
            disease_ids = sorted(entry.get("standard_names", {}).keys())
            if not disease_ids:
                continue

            aggregated_id = f"D:{idx + 1}"
            for disease_id in disease_ids:
                disease_id_to_group[disease_id] = aggregated_id
            aggregated_map[aggregated_id] = disease_ids

            standard_names = entry.get("standard_names", {})
            preferred_name = next((n for n in standard_names.values() if n), "")
            if not preferred_name:
                preferred_name = name_key if isinstance(name_key, str) else disease_ids[0]

            aliases = [a for a in entry.get("aliases", []) if a and a != preferred_name]
            descriptions = [desc for desc in entry.get("description", []) if desc]
            disease_type = entry.get("disease_type")

            disease_nodes_rows.append({
                "ID": aggregated_id,
                "standard_name": preferred_name,
                "synonyms": "; ".join(aliases),
                "disease_type": disease_type,
                "is_rare": "yes" if entry.get("is_rare") else "no",
                "description": " || ".join(descriptions)
            })

        # disease_nodes.csv
        with open(disease_nodes_path, "w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["ID", "standard_name", "synonyms", "disease_type", "is_rare", "description"])
            writer.writeheader()
            for row in disease_nodes_rows:
                writer.writerow(row)

        # publicDisease_nodes.csv (concrete disease IDs with links)
        def _build_disease_link(disease_id: str) -> str:
            """Build link for disease ID based on its type (OMIM or ORPHA)"""
            if disease_id.startswith("OMIM:"):
                # Extract numeric part from OMIM:608220 -> 608220
                numeric_part = disease_id.replace("OMIM:", "").strip()
                if numeric_part:
                    return f"https://www.omim.org/entry/{numeric_part}"
            elif disease_id.startswith("ORPHA:"):
                # Extract numeric part from ORPHA:101005 -> 101005
                numeric_part = disease_id.replace("ORPHA:", "").strip()
                if numeric_part:
                    return f"https://www.orpha.net/en/disease/detail/{numeric_part}"
            # For other disease ID types, return empty string
            return ""

        # Collect all unique concrete disease IDs with their standard names
        publicDisease_nodes_rows = []
        seen_disease_ids = set()
        
        for name_key, entry in sorted(case_database_entries.items(), key=lambda item: item[0]):
            standard_names = entry.get("standard_names", {})
            for disease_id, standard_name in standard_names.items():
                if disease_id not in seen_disease_ids:
                    seen_disease_ids.add(disease_id)
                    # Use standard_name from entry, or fallback to self.disease_names, or use disease_id
                    if not standard_name:
                        standard_name = self.disease_names.get(disease_id, disease_id)
                    link = _build_disease_link(disease_id)
                    publicDisease_nodes_rows.append({
                        "ID": disease_id,
                        "standard_name": standard_name,
                        "link": link
                    })
        
        # Also include disease IDs from aggregated_map that might not be in case_database_entries
        for aggregated_id, disease_ids in aggregated_map.items():
            for disease_id in disease_ids:
                if disease_id not in seen_disease_ids:
                    seen_disease_ids.add(disease_id)
                    standard_name = self.disease_names.get(disease_id, disease_id)
                    link = _build_disease_link(disease_id)
                    publicDisease_nodes_rows.append({
                        "ID": disease_id,
                        "standard_name": standard_name,
                        "link": link
                    })

        # Sort by ID for consistent output
        publicDisease_nodes_rows.sort(key=lambda x: x["ID"])

        with open(publicDisease_nodes_path, "w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["ID", "standard_name", "link"])
            writer.writeheader()
            for row in publicDisease_nodes_rows:
                writer.writerow(row)

        # disease_to_publicDisease_edges.csv (aggregated -> concrete disease IDs)
        with open(disease_edge_path, "w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["sourceID", "targetID", "source", "relationship"])
            writer.writeheader()
            for aggregated_id, disease_ids in aggregated_map.items():
                for disease_id in disease_ids:
                    writer.writerow({
                        "sourceID": aggregated_id,
                        "targetID": disease_id,
                        "source": "merged_by_name",
                        "relationship": "disease_exact"
                    })

        total_diseases = len(aggregated_map) if aggregated_map else 0

        # Collect all phenotypes that need IC calculation:
        # 1. Phenotypes directly associated with diseases
        # 2. All ancestors of these phenotypes (because ancestors should also count their descendants' associations)
        # 3. All descendants of these phenotypes (because descendants should also be included in the computation)
        phenotypes_directly_associated: Set[str] = set()
        for disease_ids in aggregated_map.values():
            for disease_id in disease_ids:
                phenotypes_directly_associated.update(self.disease_to_phenotypes.get(disease_id, set()))
        
        # Get all ancestors and descendants of directly associated phenotypes
        phenotypes_to_compute: Set[str] = set(phenotypes_directly_associated)
        for phenotype_id in phenotypes_directly_associated:
            # Get all ancestors (parents, grandparents, etc.)
            ancestors = self._get_all_ancestors(phenotype_id)
            phenotypes_to_compute.update(ancestors)
            # Get all descendants (children, grandchildren, etc.)
            descendants = self._get_all_descendants(phenotype_id)
            phenotypes_to_compute.update(descendants)

        # add phenotypes to compute IC: not above but under HP:0000118
        # HP:0000118 is Phenotypic abnormality root, all phenotype-abnormality descendants
        root_phenotype_id = "HP:0000118"
        if root_phenotype_id in self.phenotype_names or root_phenotype_id in self.parent_to_children:
            root_descendants = self._get_all_descendants(root_phenotype_id)
            # only HP:0000118 descendants not already in phenotypes_to_compute
            for descendant_id in root_descendants:
                if descendant_id not in phenotypes_to_compute:
                    phenotypes_to_compute.add(descendant_id)

        # If no direct associations, use all known phenotypes
        if not phenotypes_to_compute:
            phenotypes_to_compute = set(self.phenotype_names.keys())

        phenotype_rows = []
        for phenotype_id in sorted(phenotypes_to_compute):
            # Collect diseases associated with this phenotype and all its descendants
            all_associated_diseases = set()
            
            # Add diseases directly associated with this phenotype
            all_associated_diseases.update(self.phenotype_to_diseases.get(phenotype_id, set()))
            
            # Get all descendant phenotypes and add their associated diseases
            descendants = self._get_all_descendants(phenotype_id)
            for descendant_id in descendants:
                all_associated_diseases.update(self.phenotype_to_diseases.get(descendant_id, set()))
            
            # Map to aggregated disease groups
            associated_groups = {
                disease_id_to_group[d]
                for d in all_associated_diseases
                if d in disease_id_to_group
            }
            association_count = len(associated_groups)

            # if association_count still 0, treat as very specific, set to 1
            if association_count == 0:
                association_count = 1

            ic_value: Any = ""
            if total_diseases > 0 and association_count > 0:
                ic_value = -math.log(association_count / total_diseases)
                recomputed_ic[phenotype_id] = ic_value

            phenotype_rows.append({
                "ID": phenotype_id,
                "standard_name": self.phenotype_names.get(phenotype_id, phenotype_id),
                "synonyms": "; ".join(self.hpo_synonyms.get(phenotype_id, [])),
                "description": self.hpo_definitions.get(phenotype_id, ""),
                "comment": self.hpo_comments.get(phenotype_id, ""),
                # "associations": str(len(self.phenotype_to_diseases.get(phenotype_id, set()))),
                "associations": str(association_count),
                "IC": ic_value,
                "embedding": json.dumps(self.phe2embedding.get(phenotype_id, []), ensure_ascii=False)
            })

        ic_output_path = _path(graph_cfg.get("recomputed_ic_file", "ic_dict_recomputed.json"))
        if recomputed_ic:
            os.makedirs(os.path.dirname(os.path.abspath(ic_output_path)), exist_ok=True)
            with open(ic_output_path, "w", encoding="utf-8") as f:
                json.dump(recomputed_ic, f, ensure_ascii=False, indent=2)
            print(f"Recomputed IC dictionary saved to: {ic_output_path}")
        else:
            print("No IC values computed; IC dictionary not written.")

        # phenotype_nodes.csv
        with open(phenotype_nodes_path, "w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=["ID", "standard_name", "synonyms", "description", "comment", "associations", "IC", "embedding"]
            )
            writer.writeheader()
            for row in phenotype_rows:
                writer.writerow(row)

        # phenotype_to_phenotype_edges.csv (is_a relationships and alt_id exact relationships)
        with open(phenotype_edge_path, "w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["sourceID", "targetID", "source", "relationship"])
            writer.writeheader()
            # Write is_a relationships
            for child_id, parent_ids in self.hpo_is_a.items():
                for parent_id in parent_ids:
                    writer.writerow({
                        "sourceID": child_id,
                        "targetID": parent_id,
                        "source": "hpo",
                        "relationship": "is_a"
                    })
            # Write alt_id exact relationships
            # If primary_id is in phenotypes_to_compute, write all its alt_id relationships
            # even if alt_id itself is not in phenotypes_to_compute
            for primary_id, alt_ids in self.hpo_alt_ids.items():
                for alt_id in alt_ids:
                    # Write exact relationship if primary_id is in phenotypes_to_compute
                    # Don't skip if alt_id is not in phenotypes_to_compute, as they are equivalent
                    writer.writerow({
                        "sourceID": alt_id,
                        "targetID": primary_id,
                        "source": "hpo",
                        "relationship": "phenotype_exact"
                    })

        # disease_phenotype_edges.csv (aggregated disease -> phenotype)
        disease_pheno_edges: Dict[Tuple[str, str], Dict[str, Any]] = {}
        for aggregated_id, disease_ids in aggregated_map.items():
            for disease_id in disease_ids:
                for phenotype_id in self.disease_to_phenotypes.get(disease_id, set()):
                    key = (aggregated_id, phenotype_id)
                    freq_info = self.disease_phenotype_frequency.get(disease_id, {}).get(phenotype_id) \
                        or self.phenotype_disease_frequency.get((phenotype_id, disease_id))

                    frequency_display = ""
                    if freq_info:
                        frequency_display = self._format_frequency_display(
                            freq_info.get("frequency", ""),
                            freq_info.get("frequency_type", "")
                        )

                    frequency_numeric = -1
                    if frequency_display:
                        try:
                            frequency_numeric = self.get_max_frequency_from_frequency_string(frequency_display)
                        except Exception:
                            frequency_numeric = -1

                    edge_entry = disease_pheno_edges.setdefault(key, {"frequencies": set(), "max": -1, "sources": set()})
                    if frequency_display:
                        edge_entry["frequencies"].add(frequency_display)
                    edge_entry["max"] = max(edge_entry["max"], frequency_numeric)
                    freq_source = freq_info.get("source") if freq_info else None
                    if freq_source:
                        for src in [s.strip() for s in str(freq_source).split(';') if s.strip()]:
                            edge_entry["sources"].add(src)

        with open(disease_phenotype_edge_path, "w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=["sourceID", "targetID", "source", "frequency", "frequency_max", "relationship"]
            )
            writer.writeheader()
            for (source_id, target_id), info in disease_pheno_edges.items():
                freq_text = "; ".join(sorted(info["frequencies"])) if info["frequencies"] else ""
                # prevent fraction e.g. 1/2 being read as date by Excel
                # if freq has fraction, prefix single quote so Excel treats as text
                if freq_text and re.search(r'\b\d+/\d+\b', freq_text):
                    # for fraction in CSV, prefix single quote
                    # Excel treats single-quote-prefixed as text
                    freq_text = "'" + freq_text
                freq_max = "" if info["max"] < 0 else f"{info['max']:.4f}"
                source_text = "; ".join(sorted(info["sources"])) if info.get("sources") else "unknown"
                writer.writerow({
                    "sourceID": source_id,
                    "targetID": target_id,
                    "source": source_text,
                    "frequency": freq_text,
                    "frequency_max": freq_max,
                    "relationship": "has"
                })

        # Process gene-phenotype relationships using the same aggregated disease ID mapping
        gene_phenotype_output_path = self._process_gene_phenotype_relationships(disease_id_to_group, output_dir)
        
        # Process gene-disease relationships using the same aggregated disease ID mapping
        gene_disease_output_path = self._process_gene_disease_relationships(disease_id_to_group, output_dir)

        result = {
            "disease_nodes": disease_nodes_path,
            "publicDisease_nodes": publicDisease_nodes_path,
            "phenotype_nodes": phenotype_nodes_path,
            "disease_to_publicDisease_edges": disease_edge_path,
            "phenotype_to_phenotype_edges": phenotype_edge_path,
            "disease_phenotype_edges": disease_phenotype_edge_path,
            "recomputed_ic_dict": ic_output_path,
        }
        
        if gene_phenotype_output_path:
            result["gene_to_phenotype_edges"] = gene_phenotype_output_path
        
        if gene_disease_output_path:
            result["gene_to_disease_edges"] = gene_disease_output_path
        
        return result
    
    def _process_gene_phenotype_relationships(self, disease_id_to_group: Dict[str, str], output_dir: str) -> Optional[str]:
        """
        Process gene-phenotype relationships with expansion rules and output gene-phenotype relationship table.
        
        Expansion rules:
        1. Expand a specific "disease-phenotype" to n "disease-subphenotypes", then merge the related genes 
           of these n "disease-subphenotypes" as the associated genes of that specific "disease-phenotype".
        2. If a phenotype is the only child of its parent phenotype, then expand the associated genes 
           of "disease-parent phenotype" to this specific "disease-phenotype".
        
        Args:
            disease_id_to_group: Mapping from disease_id to aggregated disease ID (D:X)
            output_dir: Output directory for the gene-phenotype relationship file
            
        Returns:
            Path to the output gene-phenotype relationship CSV file, or None if skipped
        """
        if not self.genes_to_phenotype_file:
            print("No genes_to_phenotype_file specified, skipping gene-phenotype relationship processing")
            return None
        
        print("Processing gene-phenotype relationships with expansion rules...")
        
        # Debug: Check if we have any data loaded
        total_disease_phenotype_pairs = len(self.disease_phenotype_to_genes)
        print(f"  Total disease-phenotype-gene pairs loaded: {total_disease_phenotype_pairs}")
        
        # Step 1: Collect all disease-phenotype pairs from existing mappings
        # Only consider the most specific phenotypes (leaf phenotypes, not ancestors)
        disease_phenotype_genes = defaultdict(set)  # (disease_id, phenotype_id) -> set of (ncbi_gene_id, gene_symbol)
        
        # First, collect direct gene associations from genes_to_phenotype.txt
        matched_count = 0
        for (disease_id, phenotype_id), genes in self.disease_phenotype_to_genes.items():
            # Only include if this disease-phenotype pair exists in our disease_to_phenotypes mapping
            if disease_id in self.disease_to_phenotypes and phenotype_id in self.disease_to_phenotypes[disease_id]:
                disease_phenotype_genes[(disease_id, phenotype_id)].update(genes)
                matched_count += 1
        
        print(f"  Disease-phenotype pairs matched with existing mappings: {matched_count}")
        
        if not disease_phenotype_genes:
            print("  Warning: No disease-phenotype-gene pairs matched with existing disease-phenotype mappings")
            print("  This may be because:")
            print("    - The disease IDs in genes_to_phenotype.txt don't match those in disease_to_phenotypes")
            print("    - The phenotype IDs in genes_to_phenotype.txt don't match those in disease_to_phenotypes")
            return None
        
        # Step 2: Apply expansion rule 1 - expand to subphenotypes
        # For each disease-phenotype pair, find all its subphenotypes and merge their genes
        expanded_disease_phenotype_genes = defaultdict(set)
        
        for (disease_id, phenotype_id), genes in disease_phenotype_genes.items():
            # Get all descendant phenotypes (subphenotypes)
            descendants = self._get_all_descendants(phenotype_id)
            
            # Collect genes from all subphenotypes for this disease
            # Only consider subphenotypes that are also associated with this disease
            merged_genes = set(genes)  # Start with direct genes
            for descendant_id in descendants:
                # Check if this descendant has genes associated with this disease
                # AND if this descendant is also associated with this disease in disease_to_phenotypes
                if (disease_id in self.disease_to_phenotypes and 
                    descendant_id in self.disease_to_phenotypes[disease_id] and
                    (disease_id, descendant_id) in self.disease_phenotype_to_genes):
                    merged_genes.update(self.disease_phenotype_to_genes[(disease_id, descendant_id)])
            
            expanded_disease_phenotype_genes[(disease_id, phenotype_id)] = merged_genes
        
        # Step 3: Apply expansion rule 2 - expand from parent if only child
        # If a phenotype is the only child of its parent, inherit genes from parent
        final_disease_phenotype_genes = defaultdict(set)
        
        for (disease_id, phenotype_id), genes in expanded_disease_phenotype_genes.items():
            merged_genes = set(genes)
            
            # Get direct parents
            parents = self.hpo_is_a.get(phenotype_id, [])
            for parent_id in parents:
                # Check if this phenotype is the only child of its parent
                # AND if the parent is also associated with this disease
                children = self.parent_to_children.get(parent_id, [])
                if (len(children) == 1 and children[0] == phenotype_id and
                    disease_id in self.disease_to_phenotypes and 
                    parent_id in self.disease_to_phenotypes[disease_id]):
                    # This is the only child, inherit genes from parent
                    if (disease_id, parent_id) in expanded_disease_phenotype_genes:
                        merged_genes.update(expanded_disease_phenotype_genes[(disease_id, parent_id)])
            
            final_disease_phenotype_genes[(disease_id, phenotype_id)] = merged_genes
        
        # Step 4: Build gene-phenotype relationship table
        # Aggregate by aggregated disease ID (D:X)
        # First, create reverse mapping from aggregated_disease_id to list of actual disease_ids
        aggregated_to_actual_diseases = defaultdict(set)
        for actual_disease_id, aggregated_disease_id in disease_id_to_group.items():
            aggregated_to_actual_diseases[aggregated_disease_id].add(actual_disease_id)
        
        gene_phenotype_relationships = []  # List of (aggregated_disease_id, phenotype_id, ncbi_gene_id, gene_symbol, publicDisease_ids)
        
        for (disease_id, phenotype_id), genes in final_disease_phenotype_genes.items():
            if disease_id in disease_id_to_group:
                aggregated_disease_id = disease_id_to_group[disease_id]
                # Get all actual disease IDs for this aggregated disease ID
                publicDisease_ids = sorted(aggregated_to_actual_diseases[aggregated_disease_id])
                publicDisease_ids_str = ';'.join(publicDisease_ids)
                
                for (ncbi_gene_id, gene_symbol) in genes:
                    gene_phenotype_relationships.append({
                        'disease_id': aggregated_disease_id,
                        'phenotype_id': phenotype_id,
                        'ncbi_gene_id': ncbi_gene_id,
                        'gene_symbol': gene_symbol,
                        'publicDisease_ids': publicDisease_ids_str
                    })
        
        # Check if we have any relationships to output
        if not gene_phenotype_relationships:
            print("No gene-phenotype relationships found to output")
            return None
        
        # Step 5: Output gene-phenotype relationship table
        os.makedirs(output_dir, exist_ok=True)
        
        gene_phenotype_output_path = os.path.join(output_dir, "gene_to_phenotype_edges.csv")
        
        # Sort by disease_id, then phenotype_id, then gene_symbol for consistent output
        gene_phenotype_relationships.sort(key=lambda x: (x['disease_id'], x['phenotype_id'], x['gene_symbol']))
        
        with open(gene_phenotype_output_path, 'w', encoding='utf-8', newline='') as f:
            writer = csv.DictWriter(
                f,
                fieldnames=['ncbi_gene_id', 'gene_symbol', 'phenotype_id', 'disease_id', 'relationship', 'publicDisease_ids']
            )
            writer.writeheader()
            for rel in gene_phenotype_relationships:
                writer.writerow({
                    'ncbi_gene_id': rel['ncbi_gene_id'],
                    'gene_symbol': rel['gene_symbol'],
                    'phenotype_id': rel['phenotype_id'],
                    'disease_id': rel['disease_id'],
                    'relationship': 'associated_with',
                    'publicDisease_ids': rel['publicDisease_ids']
                })
        
        print(f"Saved {len(gene_phenotype_relationships)} gene-phenotype relationships to {gene_phenotype_output_path}")
        print(f"  Unique diseases: {len(set(rel['disease_id'] for rel in gene_phenotype_relationships))}")
        print(f"  Unique phenotypes: {len(set(rel['phenotype_id'] for rel in gene_phenotype_relationships))}")
        print(f"  Unique genes: {len(set((rel['ncbi_gene_id'], rel['gene_symbol']) for rel in gene_phenotype_relationships))}")
        
        return gene_phenotype_output_path
    
    def _process_gene_disease_relationships(self, disease_id_to_group: Dict[str, str], output_dir: str) -> Optional[str]:
        """
        Process gene-disease relationships and output gene-disease relationship table.
        Maps diseases to aggregated disease IDs (D:X).
        
        Args:
            disease_id_to_group: Mapping from disease_id to aggregated disease ID (D:X)
            output_dir: Output directory for the gene-disease relationship file
            
        Returns:
            Path to the output gene-disease relationship CSV file, or None if skipped
        """
        if not self.genes_to_disease_file:
            print("No genes_to_disease_file specified, skipping gene-disease relationship processing")
            return None
        
        print("Processing gene-disease relationships...")
        
        # Build gene-disease relationship list with aggregated disease IDs
        gene_disease_relationships = []
        
        for (ncbi_gene_id, gene_symbol), disease_ids in self.gene_to_diseases.items():
            for disease_id in disease_ids:
                # Map to aggregated disease ID if available
                if disease_id in disease_id_to_group:
                    aggregated_disease_id = disease_id_to_group[disease_id]
                    # Remove "NCBIGene" prefix from ncbi_gene_id
                    cleaned_ncbi_gene_id = ncbi_gene_id.replace('NCBIGene:', '').replace('NCBIGene', '')
                    gene_disease_relationships.append({
                        'disease_id': aggregated_disease_id,
                        'ncbi_gene_id': cleaned_ncbi_gene_id,
                        'gene_symbol': gene_symbol
                    })
        
        if not gene_disease_relationships:
            print("No gene-disease relationships found to output")
            return None
        
        # Output gene-disease relationship table
        os.makedirs(output_dir, exist_ok=True)
        
        gene_disease_output_path = os.path.join(output_dir, "gene_to_disease_edges.csv")
        
        # Sort by disease_id, then gene_symbol for consistent output
        gene_disease_relationships.sort(key=lambda x: (x['disease_id'], x['gene_symbol']))
        
        with open(gene_disease_output_path, 'w', encoding='utf-8', newline='') as f:
            writer = csv.DictWriter(
                f,
                fieldnames=['ncbi_gene_id', 'gene_symbol', 'disease_id', 'relationship']
            )
            writer.writeheader()
            for rel in gene_disease_relationships:
                writer.writerow({
                    'ncbi_gene_id': rel['ncbi_gene_id'],
                    'gene_symbol': rel['gene_symbol'],
                    'disease_id': rel['disease_id'],
                    'relationship': 'associated_with'
                })
        
        print(f"Saved {len(gene_disease_relationships)} gene-disease relationships to {gene_disease_output_path}")
        print(f"  Unique diseases: {len(set(rel['disease_id'] for rel in gene_disease_relationships))}")
        print(f"  Unique genes: {len(set((rel['ncbi_gene_id'], rel['gene_symbol']) for rel in gene_disease_relationships))}")
        
        return gene_disease_output_path

    def _get_frequency_weight(self, frequency: str, frequency_type: str) -> float:
        """
        Get weight for frequency annotation based on frequency value and type.
        
        Args:
            frequency: Frequency value (e.g., "80%", "HP:0040281", "4/5")
            frequency_type: Type of frequency ('fraction', 'percentage', 'hpo_id', 'empty', 'other')
            
        Returns:
            Weight value for scoring (higher weight = more important)
        """
        if not frequency or frequency.strip() == '':
            return 1.0  # Default weight for missing frequency
        
        frequency = frequency.strip()
        
        # Handle HPO ID frequencies
        if frequency_type == 'hpo_id':
            freq_description = self._convert_hpo_frequency_to_description(frequency).lower()
            
            # Map frequency descriptions to weights
            if 'very frequent' in freq_description or 'obligate' in freq_description:
                return 1.5
            elif 'frequent' in freq_description:
                return 1.5
            elif 'occasional' in freq_description:
                return 0.5
            elif 'very rare' in freq_description:
                return 0.1
            elif 'excluded' in freq_description:
                return 0.0  # Excluded phenotypes get no weight
            else:
                return 1.0  # Default for unknown HPO frequency terms
        
        # Handle percentage frequencies
        elif frequency_type == 'percentage':
            try:
                percentage = float(frequency.replace('%', ''))
                if percentage >= 90:
                    return 1.5  # Very frequent
                elif percentage >= 60:
                    return 1.5  # Frequent
                elif percentage >= 30:
                    return 0.5  # Occasional
                elif percentage >= 10:
                    return 0.1  # Very rare
                else:
                    return 0.0  # Extremely rare
            except ValueError:
                return 1.0  # Default for invalid percentage
        
        # Handle fraction frequencies
        elif frequency_type == 'fraction':
            try:
                parts = frequency.split('/')
                if len(parts) == 2:
                    numerator = int(parts[0])
                    denominator = int(parts[1])
                    if denominator > 0:
                        # If denominator is less than 5, set weight to 0.5
                        if denominator < 5:
                            return 0.3
                        
                        percentage = (numerator / denominator) * 100
                        if percentage >= 90:
                            return 1.5  # Very frequent
                        elif percentage >= 60:
                            return 1.5  # Frequent
                        elif percentage >= 30:
                            return 0.5  # Occasional
                        elif percentage >= 10:
                            return 0.1  # Very rare
                        else:
                            return 0.0  # Extremely rare
            except (ValueError, ZeroDivisionError):
                return 1.0  # Default for invalid fraction
        
        # Handle other frequency types
        else:
            return 1.0  # Default weight for other frequency types
    
    def get_frequency_info(self, phenotype_id: str, disease_id: str) -> dict:
        """
        Get frequency information for a specific phenotype-disease pair.
        
        Args:
            phenotype_id: HPO ID of the phenotype
            disease_id: Disease ID
            
        Returns:
            Dictionary containing frequency information or None if not found
        """
        key = (phenotype_id, disease_id)
        return self.phenotype_disease_frequency.get(key)

    # TODO: add phenotype frequency mapping
    def hpo2freq(self, hpo_code):
        """Convert HPO code to frequency weight"""

        if self.hpo2freq_dict is None:
            self.hpo2freq_dict = {
                'HP:0040285': [0.0, 0.0, 0.0],         # Excluded; 0%
                'HP:0040284': [0.01, 0.025, 0.04],     # Very rare; 1%-4%
                'HP:0040283': [0.05, 0.17, 0.29],      # Occasional; 5%-29%
                'HP:0040282': [0.3, 0.545, 0.79],      # Frequent; 30%-79%
                'HP:0040281': [0.8, 0.895, 0.99],      # Very frequent; 80%-99%
                'HP:0040280': [1.0, 1.0, 1.0]          # Obligate; 100%
		}
        return self.hpo2freq_dict[hpo_code]

    def word2freq(self, word):
        """Convert word to frequency weight"""
        
        if word.startswith('hp:') or word.startswith('HP:'):
            word = self._convert_hpo_frequency_to_description(word.upper())

        if self.word2freq_dict is None:
            self.word2freq_dict = {
            'very rare': 0.025, 'rare': 0.05, 'occasional': 0.17,
            'frequent': 0.545, 'very frequent': 0.895, 'obligate': 1.0  #
        }

        word = word.lower()
        if word in self.word2freq_dict:
            return self.word2freq_dict[word]

        match_obj = re.match(r'^(\d+)/(\d+)$', word)
        if match_obj:
            # if int(match_obj.group(2)) < 3:
            #     return 0.0
            return int(match_obj.group(1)) / int(match_obj.group(2))

        match_obj = re.match(r'^([\d\.]+?)\s*%$', word)
        if match_obj:
            return float(match_obj.group(1)) / 100

        match_obj = re.match(r'^(\d+)\s*of\s*(\d+)$', word)
        if match_obj:
            # if int(match_obj.group(2)) < 3:
            #     return 0.0
            return int(match_obj.group(1)) / int(match_obj.group(2))

        match_obj = re.match(r'^(\d+)%?-(\d+)%$', word)
        if match_obj:
            return (int(match_obj.group(1)) + int(match_obj.group(2))) / 200

        print('Error:', word); assert False

    def get_max_frequency_from_frequency_string(self, frequency_string: str) -> float:
        """
        Get max frequency from semicolon-sep string, return as float
        
        Args:
            frequency_string: semicolon-sep e.g. 80%; 60%; 90%
            
        Returns:
            float: max as decimal (0-1), or -1 if unparseable
        """
        if not frequency_string or not frequency_string.strip():
            return -1
        
        # split by semicolon
        frequency_parts = [part.strip() for part in frequency_string.split(';') if part.strip()]
        
        if not frequency_parts:
            return -1
        
        max_frequency = -1
        has_valid_frequency = False
        
        for part in frequency_parts:
            try:
                # use word2freq to convert each
                freq_value = self.word2freq(part)
                max_frequency = max(max_frequency, freq_value) if max_frequency >= 0 else freq_value
                has_valid_frequency = True
            except:
                # on conversion failure, log and skip
                print(f'Error converting frequency: {part}')
                continue
        
        # if no valid freq parsed, return -1
        return max_frequency if has_valid_frequency else -1

    def calculate_semantic_similarity(self, text1: str, text2: str) -> float:
        """Calculate semantic similarity between two texts"""
        text1 = text1.lower()
        text2 = text2.lower()
        try:
            embeddings = self.sentence_model.encode([text1, text2])
            similarity = np.dot(embeddings[0], embeddings[1]) / (np.linalg.norm(embeddings[0]) * np.linalg.norm(embeddings[1]))
            return float(similarity)
        except Exception as e:
            print(f"Error calculating semantic similarity: {e}")
            return 0.0
    
    def _split_disease_names(self, disease_name: str) -> List[str]:
        """
        Split disease name by ";" into list
        Args:
            disease_name: may contain ";"
        Returns:
            disease_names: split list
        """
        if ';' not in disease_name:
            return [disease_name.strip()]
        
        # split by ";"
        semicolon_split = [d.strip() for d in disease_name.split(';') if d.strip()]
        return semicolon_split
  
    def find_best_match_rank_optimized(self, true_diseases: List[str], predicted_diseases: List[str]) -> Dict:
        """Optimized version of find_best_match_rank that processes all true diseases at once with batch encoding"""
        if not predicted_diseases or not true_diseases:
            return {
                "best_match": None,
                "best_rank": -1,
                "best_similarity": 0.0,
                "all_similarities": []
            }
        
        # Process all true diseases and split them
        all_true_disease_variants = []
        true_disease_mapping = []  # Map back to original true disease index
        
        for i, true_disease in enumerate(true_diseases):
            if isinstance(true_disease, str):
                split_diseases = self._split_disease_names(true_disease)
                all_true_disease_variants.extend(split_diseases)
                true_disease_mapping.extend([i] * len(split_diseases))
            else:
                all_true_disease_variants.append(true_disease)
                true_disease_mapping.append(i)
        
        # Process all predicted diseases and split them
        all_predicted_disease_variants = []
        pred_disease_mapping = []  # Map back to original predicted disease index
        
        for i, pred_disease in enumerate(predicted_diseases):
            if isinstance(pred_disease, str):
                split_diseases = self._split_disease_names(pred_disease)
                all_predicted_disease_variants.extend(split_diseases)
                pred_disease_mapping.extend([i] * len(split_diseases))
            else:
                all_predicted_disease_variants.append(pred_disease)
                pred_disease_mapping.append(i)
        
        # Batch encode all disease names at once
        all_disease_names = all_true_disease_variants + all_predicted_disease_variants
        all_embeddings = self.sentence_model.encode(all_disease_names)
        
        # Split embeddings
        true_embeddings = all_embeddings[:len(all_true_disease_variants)]
        pred_embeddings = all_embeddings[len(all_true_disease_variants):]
        
        # Calculate similarity matrix
        similarities = []
        for i, true_emb in enumerate(true_embeddings):
            for j, pred_emb in enumerate(pred_embeddings):
                similarity = np.dot(true_emb, pred_emb) / (np.linalg.norm(true_emb) * np.linalg.norm(pred_emb))
                similarities.append({
                    "true_disease": all_true_disease_variants[i],
                    "pred_disease": all_predicted_disease_variants[j],
                    "original_true_index": true_disease_mapping[i],
                    "original_pred_index": pred_disease_mapping[j],
                    "similarity": float(similarity)
                })
        
        # Find best match
        if similarities:
            best_similarity_entry = max(similarities, key=lambda x: x["similarity"])
            
            # Debug: print the problematic match
            # if best_similarity_entry["similarity"] > 0.9:
            #     print(f"WARNING: High similarity match detected!")
            #     print(f"  True disease variant: '{best_similarity_entry['true_disease']}'")
            #     print(f"  Predicted disease variant: '{best_similarity_entry['pred_disease']}'")
            #     print(f"  Similarity: {best_similarity_entry['similarity']:.3f}")
            #     print(f"  Original true disease: '{true_diseases[best_similarity_entry['original_true_index']]}'")
            #     print(f"  Original predicted disease: '{predicted_diseases[best_similarity_entry['original_pred_index']]}'")
                
            #     # Test the similarity calculation directly
            #     test_similarity = self.calculate_semantic_similarity(
            #         best_similarity_entry['true_disease'], 
            #         best_similarity_entry['pred_disease']
            #     )
            #     print(f"  Direct test similarity: {test_similarity:.3f}")
            
            # Calculate top-k hits based on similarity threshold
            similarity_threshold = self.config.get('evaluation_config', {}).get('similarity_threshold', 0.8) # Use the same threshold as in ensemble_disease_ranking.py
            
            # Check top1, top3, top5, top10, top20, top30 hits
            top1_correct = False
            top3_correct = False
            top5_correct = False
            top10_correct = False
            top20_correct = False
            top30_correct = False
            
            # Check if any true disease matches with top-k predicted diseases
            for similarity_entry in similarities:
                pred_rank = similarity_entry["original_pred_index"] + 1  # Convert to 1-based ranking
                similarity = similarity_entry["similarity"]
                
                if similarity >= similarity_threshold:
                    if pred_rank <= 1:
                        top1_correct = True
                    if pred_rank <= 3:
                        top3_correct = True
                    if pred_rank <= 5:
                        top5_correct = True
                    if pred_rank <= 10:
                        top10_correct = True
                    if pred_rank <= 20:
                        top20_correct = True
                    if pred_rank <= 30:
                        top30_correct = True
                if top1_correct and top3_correct and top5_correct and top10_correct and top20_correct and top30_correct:
                    break
            

            return {
                "best_match": predicted_diseases[best_similarity_entry["original_pred_index"]],
                "best_rank": best_similarity_entry["original_pred_index"] + 1,
                "best_similarity": best_similarity_entry["similarity"],
                "best_true_disease": true_diseases[best_similarity_entry["original_true_index"]],
                "top1_correct": top1_correct,
                "top3_correct": top3_correct,
                "top5_correct": top5_correct,
                "top10_correct": top10_correct,
                "top20_correct": top20_correct,
                "top30_correct": top30_correct,
                "all_similarities": similarities
            }
        else:
            return {
                "best_match": None,
                "best_rank": -1,
                "best_similarity": 0.0,
                "all_similarities": []
            }
    
    def load_phenotype_disease_mappings(self):
        """Load phenotype-disease mappings from phenotype.hpoa file"""
        if not self.phenotype_hpoa_file:
            return
            
        print("Loading phenotype-disease mappings from phenotype.hpoa...")
        
        # Set to store frequency-related HPO IDs found in the file
        frequency_hpo_ids = set()
        
        try:
            hpoa_df = pd.read_csv(self.phenotype_hpoa_file, sep='\t', dtype=str, comment='#')
            
            for _, row in hpoa_df.iterrows():
                disease_id = str(row['database_id']) if pd.notna(row['database_id']) else ''
                disease_name = str(row['disease_name']) if pd.notna(row['disease_name']) else ''
                hpo_id = str(row['hpo_id']) if pd.notna(row['hpo_id']) else ''
                qualifier = str(row.get('qualifier', '')) if pd.notna(row.get('qualifier', '')) else ''
                frequency = str(row.get('frequency', '')) if pd.notna(row.get('frequency', '')) else ''
                
                # Skip phenotypes with NOT qualifier (disease does not show this phenotype)
                if qualifier == 'NOT':
                    continue
                
                # Skip if disease_id or hpo_id is empty
                if not disease_id or not hpo_id:
                    continue
                
                # Record OMIM disease IDs from phenotype.hpoa
                if disease_id.startswith('OMIM:'):
                    self.rare_disease_in_hpoa.add(disease_id)
                
                # Build phenotype to disease mapping
                self.phenotype_to_diseases[hpo_id].add(disease_id)
                self.disease_to_phenotypes[disease_id].add(hpo_id)
                self.disease_phenotype_counts[disease_id] += 1
                
                # Store frequency annotation - ensure all phenotype-disease assoc recorded
                key = (hpo_id, disease_id)
                existing_info = self.phenotype_disease_frequency.get(key)

                if frequency and frequency.strip():
                    # when frequency info present
                    new_freq = frequency.strip()
                    new_type = self._categorize_frequency(new_freq)

                    # If frequency is an HPO ID, add it to frequency HPO IDs set
                    if new_type == 'hpo_id':
                        frequency_hpo_ids.add(new_freq)

                    if existing_info:
                        # minimal merge on existing: frequency / frequency_type / source
                        # merge frequency
                        freq_vals = [
                            s.strip() for s in (existing_info.get('frequency') or '').split(';') if s.strip()
                        ]
                        if new_freq and new_freq not in freq_vals:
                            freq_vals.append(new_freq)
                        existing_info['frequency'] = '; '.join(freq_vals) if freq_vals else new_freq

                        # merge frequency_type (union; hpo_id handled later at caller)
                        type_vals = [
                            s.strip() for s in (existing_info.get('frequency_type') or '').split(';') if s.strip()
                        ]
                        if new_type and new_type not in type_vals:
                            type_vals.append(new_type)
                        existing_info['frequency_type'] = '; '.join(type_vals) if type_vals else new_type

                        # update name (HPOA overrides empty)
                        if disease_name:
                            existing_info['disease_name'] = disease_name
                        pheno_name = self.get_phenotype_name(hpo_id) or hpo_id
                        if pheno_name:
                            existing_info['phenotype_name'] = pheno_name

                        # merge source: ensure hpo in
                        src_vals = [
                            s.strip() for s in (existing_info.get('source') or '').split(';') if s.strip()
                        ]
                        if 'hpo' not in src_vals:
                            src_vals.append('hpo')
                        existing_info['source'] = '; '.join(src_vals) if src_vals else 'hpo'

                        frequency_info = existing_info
                    else:
                        frequency_info = {
                            'frequency': new_freq,
                            'frequency_type': new_type,
                            'disease_name': disease_name,
                            'phenotype_name': self.get_phenotype_name(hpo_id) or hpo_id,
                            'source': 'hpo'
                        }
                else:
                    # no frequency: ensure hpo in source, do not overwrite
                    if existing_info:
                        if not existing_info.get('frequency_type'):
                            existing_info['frequency_type'] = 'empty'

                        # name can still be filled from HPOA
                        if disease_name:
                            existing_info['disease_name'] = disease_name
                        pheno_name = self.get_phenotype_name(hpo_id) or hpo_id
                        if pheno_name:
                            existing_info['phenotype_name'] = pheno_name

                        src_vals = [
                            s.strip() for s in (existing_info.get('source') or '').split(';') if s.strip()
                        ]
                        if 'hpo' not in src_vals:
                            src_vals.append('hpo')
                        existing_info['source'] = '; '.join(src_vals) if src_vals else 'hpo'

                        frequency_info = existing_info
                    else:
                        frequency_info = {
                            'frequency': '',
                            'frequency_type': 'empty',
                            'disease_name': disease_name,
                            'phenotype_name': self.get_phenotype_name(hpo_id) or hpo_id,
                            'source': 'hpo'
                        }

                # finally write back to both index structures
                self.phenotype_disease_frequency[key] = frequency_info
                self.disease_phenotype_frequency[disease_id][hpo_id] = frequency_info
            
            # Store frequency HPO IDs and their descriptions from phenotype names
            for hpo_id in frequency_hpo_ids:
                phenotype_name = self.get_phenotype_name(hpo_id)
                if phenotype_name:
                    self.hpo_frequency_descriptions[hpo_id] = phenotype_name
            
            print(f"Loaded {len(self.phenotype_to_diseases)} phenotype-to-disease mappings")
            print(f"Loaded {len(self.phenotype_disease_frequency)} phenotype-disease associations (including empty frequency)")
            print(f"Loaded {len(self.rare_disease_in_hpoa)} unique OMIM disease IDs from phenotype.hpoa")
            print(f"Automatically detected {len(self.hpo_frequency_descriptions)} frequency HPO IDs from phenotype.hpoa:")
            for hpo_id, description in self.hpo_frequency_descriptions.items():
                print(f"  {hpo_id}: {description}")
            
        except Exception as e:
            print(f"Error loading phenotype.hpoa: {e}")
            print("Will continue without phenotype-disease mappings")

    def load_phenotype_disease_mappings_from_orphanet(self, orphanet_file: Optional[str] = None):
        """Load phenotype-disease mappings from Orphanet phenotype association file"""

        if orphanet_file is None and self.config:
            orphanet_files = self.config.get('orphanet_files')
            if isinstance(orphanet_files, dict):
                orphanet_file = orphanet_files.get('phenotype_disease_json')

        if not orphanet_file or not os.path.exists(orphanet_file):
            print(f"Warning: Orphanet phenotype-disease mapping file not found at {orphanet_file or 'N/A'}")
            return

        print(f"Loading phenotype-disease mappings from Orphanet file: {orphanet_file}...")

        try:
            with open(orphanet_file, 'r', encoding='utf-8') as f:
                orphanet_data = json.load(f)
        except Exception as e:
            print(f"Error loading Orphanet phenotype associations: {e}")
            return

        disorder_entries = orphanet_data.get('hpo_disorder_set_status_list', {}).get('entries', [])
        if not disorder_entries:
            print("Warning: No Orphanet phenotype-disease entries found in the file")
            return

        total_associations = 0
        new_associations = 0
        frequencies_added = 0
        frequencies_updated = 0
        skipped_missing_hpo = 0
        skipped_excluded_frequency = 0

        def _normalize_frequency(freq_obj):
            """Normalize Orphanet frequency information into canonical form."""

            freq_display = ''

            if isinstance(freq_obj, dict):
                freq_display = (freq_obj.get('name') or '').strip()
            elif isinstance(freq_obj, str):
                freq_display = freq_obj.strip()
            elif freq_obj is not None:
                freq_display = str(freq_obj).strip()
            
            freq_display = freq_display.split('(')[0].strip()

            if not freq_display:
                return '', 'empty', ''

            mapped_hpo_id = None
            for label, hpo_id in self.hpo_frequency_description_to_id.items():
                if freq_display == label or freq_display.lower() == label.lower():
                    mapped_hpo_id = hpo_id
                    break

            if mapped_hpo_id:
                if mapped_hpo_id not in self.hpo_frequency_descriptions:
                    self.hpo_frequency_descriptions[mapped_hpo_id] = freq_display
                return mapped_hpo_id, 'hpo_id', freq_display

            freq_display_lower = freq_display.lower()
            freq_type = self._categorize_frequency(freq_display_lower) if hasattr(self, '_categorize_frequency') else 'other'
            if freq_type == 'empty':
                freq_type = 'other'

            return freq_display_lower, freq_type, freq_display

        for entry in disorder_entries:
            disorder = entry.get('disorder') or {}
            orpha_code = str(disorder.get('orpha_code', '')).strip()
            if not orpha_code or orpha_code.lower() == 'nan':
                continue

            disease_id = f"ORPHA:{orpha_code}"
            disease_name = (disorder.get('name') or '').strip()

            if disease_name and not self._should_exclude_alias(disease_name, disease_id):
                if disease_id not in self.disease_names or not self.disease_names[disease_id]:
                    self.disease_names[disease_id] = disease_name
                synonyms = self.disease_mapping_with_synonyms.setdefault(disease_id, [])
                if disease_name not in synonyms:
                    synonyms.append(disease_name)
                for name_variant in self.disease_mapping_with_synonyms[disease_id]:
                    if name_variant:
                        self.disease_name_to_ids[name_variant.lower()].add(disease_id)

            # disorder_type = disorder.get('disorder_type', {})
            # disorder_type_name = (disorder_type.get('name') or '').strip()
            # if disorder_type_name and disease_id not in self.disease_types:
            #     self.disease_types[disease_id] = disorder_type_name

            hpo_associations = disorder.get('hpo_associations') or []
            for association in hpo_associations:
                hpo_info = association.get('hpo') or {}
                phenotype_id = (hpo_info.get('id') or '').strip()
                if not phenotype_id:
                    skipped_missing_hpo += 1
                    continue

                freq_value, freq_type, freq_display = _normalize_frequency(association.get('frequency'))

                if freq_display == 'Excluded':
                    skipped_excluded_frequency += 1
                    continue

                total_associations += 1
                already_present = phenotype_id in self.disease_to_phenotypes[disease_id]

                self.phenotype_to_diseases[phenotype_id].add(disease_id)
                self.disease_to_phenotypes[disease_id].add(phenotype_id)

                if not already_present:
                    self.disease_phenotype_counts[disease_id] += 1
                    new_associations += 1

                phenotype_name = self.get_phenotype_name(phenotype_id)
                disease_name_resolved = disease_name or self.get_disease_name(disease_id)

                freq_info = {
                    'frequency': freq_value,
                    'frequency_type': freq_type,
                    'disease_name': disease_name_resolved,
                    'phenotype_name': phenotype_name,
                    'source': 'orphanet',
                }

                key = (phenotype_id, disease_id)
                existing_info = self.phenotype_disease_frequency.get(key)

                if existing_info:
                    updated = False

                    existing_freq_values = [item.strip() for item in existing_info.get('frequency', '').split(';') if item.strip()]
                    if freq_value and freq_value not in existing_freq_values:
                        if existing_freq_values:
                            existing_info['frequency'] = existing_info['frequency'] + '; ' + freq_value
                        else:
                            existing_info['frequency'] = freq_value
                        updated = True

                    existing_freq_types = [item.strip() for item in existing_info.get('frequency_type', '').split(';') if item.strip()]
                    if freq_type and freq_type not in existing_freq_types:
                        if existing_freq_types:
                            existing_info['frequency_type'] = existing_info['frequency_type'] + '; ' + freq_type
                        else:
                            existing_info['frequency_type'] = freq_type
                        updated = True

                    existing_sources = [item.strip() for item in existing_info.get('source', '').split(';') if item.strip()]
                    if freq_info.get('source') and freq_info.get('source') not in existing_sources:
                        if existing_sources:
                            existing_info['source'] = existing_info['source'] + '; ' + freq_info.get('source')
                        else:
                            existing_info['source'] = freq_info.get('source')
                        updated = True

                    if updated:
                        frequencies_updated += 1
                else:
                    if not freq_value:
                        freq_info['frequency_type'] = 'empty'

                    self.phenotype_disease_frequency[key] = freq_info
                    self.disease_phenotype_frequency[disease_id][phenotype_id] = freq_info

                    if freq_value:
                        frequencies_added += 1

        print(f"Loaded {total_associations} Orphanet phenotype-disease associations")
        print(f"  New associations added: {new_associations}")
        print(f"  Frequency annotations added: {frequencies_added}, updated: {frequencies_updated}")
        if skipped_missing_hpo:
            print(f"  Skipped {skipped_missing_hpo} associations without phenotype IDs")
        if skipped_excluded_frequency:
            print(f"  Skipped {skipped_excluded_frequency} associations with excluded frequency annotations")

    def load_phenotype_to_diseases_from_genes_file(self):
        """Load phenotype-to-diseases mappings from phenotype_to_genes.txt file"""
        if not self.phenotype_to_genes_file:
            return
            
        print(f"Loading phenotype-to-diseases mappings from {self.phenotype_to_genes_file}...")
        
        try:
            with open(self.phenotype_to_genes_file, 'r', encoding='utf-8') as f:
                for line_num, line in enumerate(f, 1):
                    line = line.strip()
                    if not line or line.startswith('#'):
                        continue
                        
                    parts = line.split('\t')
                    if len(parts) < 5:  # hpo_id, hpo_name, ncbi_gene_id, gene_symbol, disease_id
                        print(f"Skipping line {line_num} with invalid format: {line}")
                        continue
                        
                    hpo_id = parts[0]
                    disease_id = parts[4]  # disease_id is the 5th column
                    
                    # Skip if disease_id is empty or '-'
                    if not disease_id or disease_id == '-':
                        continue
                    
                    self.phenotype_to_diseases_from_genes[hpo_id].add(disease_id)
                    self.disease_to_phenotypes_from_genes[disease_id].add(hpo_id)
            
            print(f"Loaded {len(self.phenotype_to_diseases_from_genes)} phenotype-to-disease mappings from genes file")
            
        except Exception as e:
            print(f"Error loading phenotype_to_genes.txt: {e}")
            print("Will continue without phenotype-to-diseases from genes mappings")
    
    def load_genes_to_phenotype_diseases_file(self):
        """Load genes-to-phenotype-diseases mappings from genes_to_phenotype.txt file"""
        if not self.genes_to_phenotype_file:
            return
            
        print(f"Loading genes-to-phenotype-diseases mappings from {self.genes_to_phenotype_file}...")
        
        try:
            with open(self.genes_to_phenotype_file, 'r', encoding='utf-8') as f:
                for line_num, line in enumerate(f, 1):
                    line = line.strip()
                    if not line or line.startswith('#'):
                        continue
                        
                    parts = line.split('\t')
                    if len(parts) < 6:  # ncbi_gene_id, gene_symbol, hpo_id, hpo_name, frequency, disease_id
                        print(f"Skipping line {line_num} with invalid format: {line}")
                        continue
                        
                    ncbi_gene_id = parts[0].strip()  # ncbi_gene_id is the 1st column
                    gene_symbol = parts[1].strip()  # gene_symbol is the 2nd column
                    hpo_id = parts[2].strip()  # hpo_id is the 3rd column
                    disease_id = parts[5].strip()  # disease_id is the 6th column
                    
                    # Skip if disease_id is empty or '-'
                    if not disease_id or disease_id == '-':
                        continue
                    
                    # Skip if hpo_id is empty or doesn't start with 'HP:'
                    if not hpo_id or not hpo_id.startswith('HP:'):
                        continue
                    
                    # Skip if gene information is missing
                    if not ncbi_gene_id or not gene_symbol:
                        continue
                    
                    # Store phenotype-disease mappings
                    self.phenotype_to_diseases_from_genes[hpo_id].add(disease_id)
                    self.disease_to_phenotypes_from_genes[disease_id].add(hpo_id)
                    
                    # Store gene-phenotype-disease mappings
                    gene_key = (ncbi_gene_id, gene_symbol)
                    self.disease_phenotype_to_genes[(disease_id, hpo_id)].add(gene_key)
                    self.phenotype_to_genes[hpo_id].add(gene_key)
                    self.gene_to_phenotypes[gene_key].add(hpo_id)
            
            print(f"Loaded {len(self.disease_to_phenotypes_from_genes)} genes-to-phenotype-diseases mappings")
            print(f"Loaded {len(self.disease_phenotype_to_genes)} disease-phenotype-gene associations")
            
        except Exception as e:
            print(f"Error loading genes_to_phenotype.txt: {e}")
            print("Will continue without genes-to-phenotype-diseases mappings")
    
    def load_genes_to_disease_file(self):
        """Load genes-to-disease mappings from genes_to_disease.txt file"""
        if not self.genes_to_disease_file:
            return
            
        print(f"Loading genes-to-disease mappings from {self.genes_to_disease_file}...")
        
        try:
            with open(self.genes_to_disease_file, 'r', encoding='utf-8') as f:
                # Skip header line
                next(f, None)
                
                for line_num, line in enumerate(f, 2):  # Start from line 2 (after header)
                    line = line.strip()
                    if not line or line.startswith('#'):
                        continue
                        
                    parts = line.split('\t')
                    if len(parts) < 5:  # ncbi_gene_id, gene_symbol, association_type, disease_id, source
                        if line_num <= 5:  # Only print first few errors
                            print(f"Skipping line {line_num} with invalid format: {line}")
                        continue
                        
                    ncbi_gene_id = parts[0].strip()  # ncbi_gene_id is the 1st column
                    gene_symbol = parts[1].strip()  # gene_symbol is the 2nd column
                    disease_id = parts[3].strip()  # disease_id is the 4th column
                    
                    # Skip if disease_id is empty or '-'
                    if not disease_id or disease_id == '-':
                        continue
                    
                    # Skip if gene information is missing
                    if not ncbi_gene_id or not gene_symbol:
                        continue
                    
                    # Store gene-disease mappings
                    gene_key = (ncbi_gene_id, gene_symbol)
                    self.gene_to_diseases[gene_key].add(disease_id)
                    self.disease_to_genes[disease_id].add(gene_key)
            
            print(f"Loaded {len(self.gene_to_diseases)} gene-to-disease mappings")
            print(f"Loaded {len(self.disease_to_genes)} disease-to-gene mappings")
            
        except Exception as e:
            print(f"Error loading genes_to_disease.txt: {e}")
            print("Will continue without genes-to-disease mappings")
        
    def enhance_phenotype_disease_mappings_with_genes(self):
        """Enhance phenotype-disease mappings using gene information"""
        print("Enhancing phenotype-disease mappings with gene information...")
        
        # Merge phenotype-disease mappings from gene files with existing mappings
        for phenotype_id, diseases in self.phenotype_to_diseases_from_genes.items():
            for disease_id in diseases:
                # Add phenotype-disease association from gene files
                self.phenotype_to_diseases[phenotype_id].add(disease_id)
                self.disease_to_phenotypes[disease_id].add(phenotype_id)
                self.disease_phenotype_counts[disease_id] += 1
        
        print(f"Enhanced mappings: {len(self.phenotype_to_diseases)} phenotype-to-disease associations")
    
    def merge_disease_phenotypes_by_name(self):
        """Merge phenotypes for disease IDs with the same disease name"""
        print("Merging disease phenotypes by disease name...")
        
        # for each name, all its IDs
        for disease_name_lower, disease_ids in self.disease_name_to_ids.items():
            disease_ids_list = list(disease_ids)
            
            if len(disease_ids_list) > 1:
                # multiple IDs for same name, merge
                # merge phenotypes across IDs
                all_phenotypes = set()
                
                # collect all freq info
                all_frequency_info = {}  # phenotype_id -> frequency_info
                total_count = 0
                
                for disease_id in disease_ids_list:
                    if disease_id in self.disease_to_phenotypes:
                        all_phenotypes.update(self.disease_to_phenotypes[disease_id])
                
                        # use merged phenotype set size as total, avoid double count
                        total_count = len(all_phenotypes)
                        
                        # collect all freq for this disease ID
                        for phenotype_id in self.disease_to_phenotypes[disease_id]:
                            freq_key = (phenotype_id, disease_id)
                            if freq_key in self.phenotype_disease_frequency:
                                current_freq_info = self.phenotype_disease_frequency[freq_key]
                                if phenotype_id not in all_frequency_info:
                                    # first time for this phenotype ID, use current freq
                                    all_frequency_info[phenotype_id] = current_freq_info.copy()
                                else:
                                    # already exists, merge freq and freq_type
                                    # merge freq - only add when new non-empty
                                    if current_freq_info['frequency'] and current_freq_info['frequency'].strip():
                                        existing_freq = all_frequency_info[phenotype_id].get('frequency', '')
                                        if existing_freq and existing_freq.strip():
                                            all_frequency_info[phenotype_id]['frequency'] = existing_freq + '; ' + current_freq_info['frequency']
                                        else:
                                            all_frequency_info[phenotype_id]['frequency'] = current_freq_info['frequency']
                                    # merge freq_type: prefer hpo_id
                                    if 'hpo_id' in all_frequency_info[phenotype_id].get('frequency_type', '') or 'hpo_id' in current_freq_info.get('frequency_type', ''):
                                        all_frequency_info[phenotype_id]['frequency_type'] = 'hpo_id'
                                    else:
                                        all_frequency_info[phenotype_id]['frequency_type'] = (
                                            (all_frequency_info[phenotype_id].get('frequency_type', '') or '') + '; ' + current_freq_info.get('frequency_type', '')
                                        ).strip('; ').strip()
                                    # merge source
                                    existing_source = all_frequency_info[phenotype_id].get('source', '')
                                    new_source = current_freq_info.get('source', '')
                                    sources = [s.strip() for s in (existing_source + ';' + new_source).split(';') if s.strip()]
                                    dedup_sources = []
                                    for s in sources:
                                        if s not in dedup_sources:
                                            dedup_sources.append(s)
                                    all_frequency_info[phenotype_id]['source'] = '; '.join(dedup_sources)
                
                # point all IDs to merged phenotype set
                for disease_id in disease_ids_list:
                    self.disease_to_phenotypes[disease_id] = all_phenotypes.copy()
                    self.disease_phenotype_counts[disease_id] = total_count
                    
                    # update freq: all IDs share same
                    self.disease_phenotype_frequency[disease_id] = all_frequency_info.copy()
                    
                    # update phenotype_disease_frequency and phenotype_to_diseases
                    for phenotype_id in all_phenotypes:
                        freq_key = (phenotype_id, disease_id)
                        if phenotype_id in all_frequency_info:
                            self.phenotype_disease_frequency[freq_key] = all_frequency_info[phenotype_id]
                        # ensure all merged disease IDs in phenotype_to_diseases
                        self.phenotype_to_diseases[phenotype_id].add(disease_id)
                
                # print(f"Merged {len(disease_ids_list)} disease IDs for '{disease_name_lower}': {disease_ids_list}")
                # print(f"  Combined phenotypes: {len(all_phenotypes)}")
                # print(f"  Combined frequency annotations: {len(all_frequency_info)}")
            else:
                # BUGFIX: single disease ID must be in disease_to_phenotypes
                # if not, add empty set
                disease_id = disease_ids_list[0]
                if disease_id not in self.disease_to_phenotypes:
                    self.disease_to_phenotypes[disease_id] = set()
                    self.disease_phenotype_counts[disease_id] = 0
        
        print(f"Merging complete. Total diseases: {len(self.disease_to_phenotypes)}")
    
    def merge_disease_synonyms_by_name(self):
        """Ensure disease IDs sharing a name have identical synonym lists."""
        print("Merging disease synonyms by disease name...")

        for disease_name_lower, disease_ids in self.disease_name_to_ids.items():
            if len(disease_ids) <= 1:
                continue

            combined_synonyms: List[str] = []
            seen_synonyms_lower: Set[str] = set()

            for disease_id in disease_ids:
                # ensure each disease ID has list for aliases
                synonyms = self.disease_mapping_with_synonyms.setdefault(disease_id, [])

                # main name also as alias
                primary_name = self.disease_names.get(disease_id)
                if primary_name and primary_name not in synonyms and not self._should_exclude_alias(primary_name, disease_id):
                    synonyms.append(primary_name)

                for alias in synonyms:
                    if not alias:
                        continue
                    alias_stripped = alias.strip()
                    if not alias_stripped:
                        continue
                    alias_lower = alias_stripped.lower()
                    if alias_lower in seen_synonyms_lower:
                        continue
                    combined_synonyms.append(alias_stripped)
                    seen_synonyms_lower.add(alias_lower)

            # sync merged alias list to all same-name IDs
            for disease_id in disease_ids:
                self.disease_mapping_with_synonyms[disease_id] = combined_synonyms.copy()
            
        print("Disease synonym merging complete.")

    def integrate_frequency_numeric(self):
        """
        Unified numeric for disease-phenotype frequency
        Multiple freqs: convert to number and take max
        """
        print("Integrating frequency information into numeric values...")
        
        # count processed freq
        total_processed = 0
        total_with_frequency = 0
        
        # iterate all phenotype-disease freq
        for freq_key, freq_info in self.phenotype_disease_frequency.items():
            phenotype_id, disease_id = freq_key
            frequency_string = freq_info.get('frequency', '')
            
            # use get_max_frequency_from_frequency_string to number
            if freq_info['frequency_type'] == 'hpo_id':
                frequency_string = self._convert_hpo_frequency_to_description(frequency_string)
            else:
                frequency_string = frequency_string
            frequency_numeric = self.get_max_frequency_from_frequency_string(frequency_string)

            if len(frequency_string) > 100:
                print(f"Frequency string too long: {freq_info.get('frequency', '')}...{frequency_string[:100]}...")
                return
            
            # print(f"Frequency numeric: {frequency_numeric} for {frequency_string}")
            
            # add numeric freq to freq info
            freq_info['frequency_numeric'] = frequency_numeric
                        
            total_processed += 1
            if frequency_string and frequency_string.strip():
                total_with_frequency += 1
        
        # also update disease_phenotype_frequency
        for disease_id, phenotype_freq_dict in self.disease_phenotype_frequency.items():
            for phenotype_id, freq_info in phenotype_freq_dict.items():
                frequency_string = freq_info.get('frequency', '')
                if freq_info['frequency_type'] == 'hpo_id':
                    frequency_string = self._convert_hpo_frequency_to_description(frequency_string)
                else:
                    frequency_string = frequency_string
                frequency_numeric = self.get_max_frequency_from_frequency_string(frequency_string)
                freq_info['frequency_numeric'] = frequency_numeric
        
        print(f"Frequency integration complete.")
        print(f"  Total frequency entries processed: {total_processed}")
        print(f"  Entries with frequency information: {total_with_frequency}")
        print(f"  Entries without frequency: {total_processed - total_with_frequency}")
    
    def get_mapping_summary(self) -> Dict:
        """Get a summary of all mappings loaded"""
        # Count frequency annotations by type
        frequency_type_counts = {}
        for freq_info in self.phenotype_disease_frequency.values():
            freq_type = freq_info['frequency_type']
            frequency_type_counts[freq_type] = frequency_type_counts.get(freq_type, 0) + 1
        
        # Count disease synonyms statistics
        diseases_with_synonyms = 0
        total_synonyms = 0
        for disease_id, synonyms in self.disease_mapping_with_synonyms.items():
            if len(synonyms) > 1:  # More than just the primary name
                diseases_with_synonyms += 1
                total_synonyms += len(synonyms) - 1  # Exclude the primary name
        
        return {
            "phenotype_to_diseases": len(self.phenotype_to_diseases),
            "disease_to_phenotypes": len(self.disease_to_phenotypes),
            "phenotype_to_diseases_from_genes": len(self.phenotype_to_diseases_from_genes),
            "disease_to_phenotypes_from_genes": len(self.disease_to_phenotypes_from_genes),
            "disease_synonyms": {
                "total_diseases": len(self.disease_mapping_with_synonyms),
                "diseases_with_synonyms": diseases_with_synonyms,
                "total_synonyms": total_synonyms
            },
            "frequency_annotations": {
                "total": len(self.phenotype_disease_frequency),
                "by_type": frequency_type_counts
            }
        }
    
    def get_phenotype_diseases(self, phenotype_id: str) -> List[str]:
        """Get diseases associated with a phenotype from phenotype.hpoa"""
        disease_ids = self.phenotype_to_diseases.get(phenotype_id, set())
        return [self.get_disease_name(did) for did in disease_ids]
    
    def get_disease_description(self, disease_id: str) -> str:
        """
        Get disease description, first from loaded JSON file, then from HPO website using the scraper
        
        Args:
            disease_id (str): Disease ID (e.g., OMIM:176270, ORPHA:739, MONDO:0008300)
            
        Returns:
            str: Disease description from JSON file or HPO website, or empty string if not available
        """
        # print(f"Getting disease description for {disease_id}...")
        
        # First, check if we have the description in our loaded JSON file
        if disease_id in self.disease_descriptions:
            description = self.disease_descriptions[disease_id]
            # print(f"Found description for {disease_id} in loaded JSON file")
            return description
        
        # If not found in JSON file, try scraping from HPO website
        print(f"Description for {disease_id} not found in JSON file, attempting to scrape from HPO website...")
        try:
            # Use the HPO scraper to get disease description
            result = disease_description_scraper(disease_id)
            
            if result and result.get('status') == 'success':
                return result.get('description', '')
            elif result and result.get('status') == 'no_content':
                # Successfully accessed but no description available
                return ''
            else:
                # Failed to access or other error
                return ''
                
        except Exception as e:
            # If any error occurs, return empty string
            return ''

   

    def _init_lr_ranking_engine(self):
        """Initialize LR ranking engine (lazy loading)"""
        if self.lr_ranking_engine is None:
            try:
                print("Initializing LR ranking engine...")
                ontology = load_minimal_ontology_from_hp_json()
                diseases = load_disease_models_from_hpoa()
                self.lr_ranking_engine = PhenotypeRankingEngine(ontology, diseases)
                print("LR ranking engine initialized successfully")
            except Exception as e:
                print(f"Warning: Failed to initialize LR ranking engine: {e}")
                self.lr_ranking_engine = None
    
    def _get_lr_score(self, phenotypes: List[str], disease_id: str, excluded_phenotypes: List[str] = None) -> float:
        """
        Calculate posttest probability for a disease given phenotypes using LR ranking.
        
        Args:
            phenotypes: List of observed phenotype IDs
            disease_id: Disease ID to calculate score for
            excluded_phenotypes: Optional list of excluded phenotype IDs
            
        Returns:
            Posttest probability (0.0 to 1.0), or 0.0 if calculation fails
        """
        if self.lr_ranking_engine is None:
            self._init_lr_ranking_engine()
        
        if self.lr_ranking_engine is None:
            return 0.0
        
        try:
            excluded = excluded_phenotypes or []
            results = self.lr_ranking_engine.rank(phenotypes, excluded)
            
            # Find the result for the specific disease
            for result in results:
                if result.disease_id.upper() == disease_id.upper():
                    # Return posttest probability directly (already in 0-1 range)
                    return float(result.posttest_probability)
            
            # Disease not found in results, return 0.0
            return 0.0
        except Exception as e:
            print(f"Warning: Error calculating posttest probability for {disease_id}: {e}")
            return 0.0
    
    def get_all_parent_phenotypes(self, phenotype_id: str, visited: set = None, max_depth: int = 1) -> set:
        """
        Recursively get all parent phenotypes (including grandparents, great-grandparents, etc.)
        
        Args:
            phenotype_id: The HPO ID to get parents for
            visited: Set to track visited phenotypes to avoid infinite loops
            max_depth: Maximum depth to traverse (1=parent, 2=grandparent, 3=great-grandparent)
            
        Returns:
            Set of all parent phenotype IDs
        """
        if visited is None:
            visited = set()
        
        # Avoid infinite loops in case of circular references
        if phenotype_id in visited:
            return set()
        
        # Check depth limit
        if max_depth <= 0:
            return set()
        
        visited.add(phenotype_id)
        all_parents = set()
        
        # Get direct parents
        direct_parents = self.hpo_is_a.get(phenotype_id, [])
        all_parents.update(direct_parents)
        
        # Recursively get parents of parents (with depth control)
        for parent in direct_parents:
            all_parents.update(self.get_all_parent_phenotypes(parent, visited.copy(), max_depth - 1))
        
        return all_parents

    def rank_diseases_by_phenotype_associations(self, phenotypes: List[str], use_frequency_weights: bool = False, use_IC_weights: bool = False, use_score: bool = False, use_samples: bool = False, exclude_sample_id: int = None, disease_examples: List[str] = None) -> List[Dict]:
        """
        Rank diseases based on their association count with the given phenotypes.
        
        Args:
            phenotypes: List of phenotype IDs (HPO terms)
            use_frequency_weights: Whether to use frequency weights to rank diseases
            use_IC_weights: Whether to use IC weights to rank diseases
            use_samples: Whether to use case_library JSONL data instead of phenotype.hpoa data
            exclude_sample_id: Whether to exclude a specific sample ID
            disease_examples: List of disease IDs that are associated with the given phenotypes
            
        Returns:
            List of dictionaries containing disease info and ranking scores, sorted by score
        """
        
        disease_scores = defaultdict(int)
        disease_info = {}

        # for testing: exclude given sample; in production no exclude, no sample_id; _load_trueSamples called in generate_dataset
        # note: true cases here merged by name, not sample-based; sample-based in rank_samples_by_phenotype_associations
        if use_samples:
            self.phenotype_to_diseases_from_trueSamples = defaultdict(set)
            self.disease_to_phenotypes_from_trueSamples = defaultdict(set)
            self._load_trueSamples(exclude_sample_id)

        # get all parent phenotypes (including grandparents, great-grandparents, etc.) for each phenotype
        parents_phenotypes = {}
        for phenotype in phenotypes:
            all_parents = self.get_all_parent_phenotypes(phenotype)
            parents_phenotypes[phenotype] = list(all_parents)

        total_IC = sum(float(self.ic_dict.get(phenotype_id, 0)) for phenotype_id in phenotypes)

        # Pre-compute LR ranking results once for all diseases (to avoid repeated calculations)
        lr_results_cache = {}
        if use_frequency_weights or use_score:
            if self.lr_ranking_engine is None:
                self._init_lr_ranking_engine()
            if self.lr_ranking_engine is not None:
                try:
                    # Get all LR ranking results once
                    all_lr_results = self.lr_ranking_engine.rank(phenotypes, [])
                    # Cache results by disease_id (uppercase for case-insensitive lookup)
                    for result in all_lr_results:
                        lr_results_cache[result.disease_id.upper()] = result.posttest_probability
                except Exception as e:
                    print(f"Warning: Failed to compute LR rankings: {e}")
                    lr_results_cache = {}

        # add topN diseases by embedding similarity
        # similar_samples, similar_values = self.find_similar_samples_with_embeddings_v1(phenotypes, 200)
        # candi_diseases_by_embedding = set()
        # for sample, similarity in zip(similar_samples, similar_values):
        #     # Get diseases directly from the similar case
        #     sample_diseases = sample.get('RareDisease', [])
        #     candi_diseases_by_embedding.update(sample_diseases)

        # Calculate scores for each disease based on phenotype associations
        for phenotype_id in phenotypes:
            # Get diseases associated with this phenotype
            if disease_examples:
                associated_diseases = disease_examples
            elif use_samples:
                associated_diseases = self.phenotype_to_diseases_from_trueSamples.get(phenotype_id, set())
            else:
                associated_diseases = self.phenotype_to_diseases.get(phenotype_id, set())
            
            # BUGFIX: if phenotype has no diseases, get parents and their diseases
            if not associated_diseases and phenotype_id in parents_phenotypes:
                for p_id in parents_phenotypes[phenotype_id]:
                    associated_diseases.update(self.phenotype_to_diseases.get(p_id, set()))
            
            # associated_diseases.update(candi_diseases_by_embedding)
            # print(f"rare_disease_in_hpoa: {len(self.rare_disease_in_hpoa)}")

            for disease_id in associated_diseases:
                # Store disease info if not already stored

                if disease_id not in disease_info:
                    
                    # if disease_id not in self.rare_disease_types:
                    #     continue
                    # if disease_id not in self.rare_disease_in_hpoa:
                    #     continue

                    # find same-name id group for disease_id; if same-name in disease_info, add id to disease_info['disease_id']
                    have_added = False
                    for disease_name, disease_ids in self.disease_name_to_ids.items():
                        if disease_id in disease_ids:
                            for did in disease_ids:
                                if did in disease_info and disease_id not in disease_info[did]['disease_id']:
                                    disease_info[did]['disease_id'].append(disease_id)
                                    have_added = True
                                    break
                            break

                    if have_added:
                        continue

                    # Calculate matching phenotypes for this disease
                    if use_samples:
                        matching_phenotypes = list(self.disease_to_phenotypes_from_trueSamples[disease_id].intersection(set(phenotypes)))
                    else:
                        matching_phenotypes = list(self.disease_to_phenotypes[disease_id].intersection(set(phenotypes)))

                    # Check for phenotype matches through parent relationships
                    matching_phenotypes_by_parents = []
                    matching_phenotypes_by_parents_map = {}

                    parents_phenotypes_in_disease = set()  # Use set for efficient lookup
                    
                    # Collect all parent phenotypes for this disease
                    disease_phenotypes = self.disease_to_phenotypes_from_trueSamples.get(disease_id, []) if use_samples else self.disease_to_phenotypes.get(disease_id, [])
                    for phenotype in disease_phenotypes:
                        phenotype_parents = self.hpo_is_a.get(phenotype, [])
                        parents_phenotypes_in_disease.update(phenotype_parents)
                    
                    # Check if any patient phenotype's parents match disease phenotype
                    for phenotype, parent_list in parents_phenotypes.items():
                        if parent_list:
                            for parent in parent_list:
                                if parent in disease_phenotypes:
                                    matching_phenotypes_by_parents.append(phenotype + " (by parent)")
                                    matching_phenotypes_by_parents_map[phenotype] = parent
                                    break

                        # Check if any parent phenotype is in the disease's parent phenotypes
                        # if any(parent in parents_phenotypes_in_disease for parent in parent_list):
                        #     if phenotype not in matching_phenotypes:
                        #         matching_phenotypes_by_parents.append(phenotype + " (by parent)")

                    # dedup
                    # dedup (by parent) in matching_phenotypes_by_parents
                    for phenotype in matching_phenotypes_by_parents:
                        if '(' in phenotype:
                            phenotype_id = phenotype.split('(')[0].strip()
                        else:
                            phenotype_id = phenotype.strip()
                            
                        if phenotype_id not in matching_phenotypes:
                            matching_phenotypes.append(phenotype)
                    
                    # ---------------------------------------------
                    # check if any phenotype in given phenotypes is a parent phenotype with only one child
                    matching_phenotypes_by_child_map = {}
                    for phenotype in phenotypes:
                        if (phenotype in matching_phenotypes) or (phenotype + " (by parent)" in matching_phenotypes):
                            continue
                        child_phenotype = self._is_hp_with_only_child(phenotype)
                        if child_phenotype and child_phenotype in disease_phenotypes:
                            matching_phenotypes.append(phenotype + " (by child)")
                            matching_phenotypes_by_child_map[phenotype] = child_phenotype
                    # ---------------------------------------------
                    
                    # Create matching phenotypes with frequency annotations
                    matching_phenotypes_with_freq = []
                    matching_phenotypes_freq_info = []
                    # listcomp to avoid modify-while-iterate
                    filtered_matching_phenotypes = []
                    
                    for matched_phenotype_id in matching_phenotypes:
                        phenotype_id_original = matched_phenotype_id
                        if 'by parent' in matched_phenotype_id:
                            matched_phenotype_id = matching_phenotypes_by_parents_map[matched_phenotype_id.split(' (by parent)')[0].strip()]
                        elif 'by child' in matched_phenotype_id:
                            matched_phenotype_id = matching_phenotypes_by_child_map[matched_phenotype_id.split(' (by child)')[0].strip()]

                        # if matched is parent, count its children
                        child_count = 1
                        if 'by parent' in phenotype_id_original:
                            parent_phenotype_id = phenotype_id_original.split(' (by parent)')[0].strip()
                            child_count = len(self.parent_to_children.get(parent_phenotype_id, []))
                            if child_count == 0:
                                child_count = 1

                        freq_info = self.get_frequency_info(matched_phenotype_id, disease_id)
                        if freq_info:
                            # Convert HPO ID frequencies to descriptions
                            if freq_info['frequency_type'] == 'hpo_id':
                                frequency_display = self._convert_hpo_frequency_to_description(freq_info['frequency'])
                            else:
                                frequency_display = freq_info['frequency']
                            
                            # Add frequency to phenotype ID
                            if frequency_display:
                                frequency_display = self.get_max_frequency_from_frequency_string(frequency_display)
                                
                                # if frequency_display < 0.17:
                                #     # skip low-freq phenotype
                                #     continue

                                phenotype_with_freq = f"{phenotype_id_original} ({frequency_display})"
                                matching_phenotypes_with_freq.append(phenotype_with_freq)
                                matching_phenotypes_freq_info.append(frequency_display)
                                filtered_matching_phenotypes.append(phenotype_id_original)
                            else:
                                # skip phenotype without freq
                                # continue
                            
                                phenotype_with_freq = phenotype_id_original
                                matching_phenotypes_with_freq.append(phenotype_with_freq)
                                matching_phenotypes_freq_info.append(0.5)
                                filtered_matching_phenotypes.append(phenotype_id_original)

                        else:
                            # print(f"Error: No frequency information for {phenotype_id_original}")
                            # skip no-freq phenotype
                            # continue
                        
                            phenotype_with_freq = phenotype_id_original
                            matching_phenotypes_with_freq.append(phenotype_with_freq)
                            matching_phenotypes_freq_info.append(0.5)
                            filtered_matching_phenotypes.append(phenotype_id_original)
                    
                    # update matching_phenotypes to filtered list
                    matching_phenotypes = filtered_matching_phenotypes
                    
                    matching_count = len(matching_phenotypes)
                    matching_count_weighted = sum(matching_phenotypes_freq_info)

                    # skip if disease name already exists
                    current_disease_name = self.get_disease_name(disease_id)
                    existing_disease_names = {info['disease_name'] for info in disease_info.values()}
                    if current_disease_name in existing_disease_names:
                        continue
                    
                    matching_HPO = []
                    for matched_phenotype_id in matching_phenotypes:
                        if '(' in matched_phenotype_id:
                            matching_HPO.append(matched_phenotype_id.split('(')[0].strip())
                        else:
                            matching_HPO.append(matched_phenotype_id.strip())
                    
                    # check if all matched_phenotype_id in matching_HPO are in ic_dict
                    # for matched_phenotype_id in matching_HPO:
                    #     if matched_phenotype_id not in self.ic_dict:
                    #         print(matched_phenotype_id, "not in ic_dict")
                    #     else:
                    #         print(matched_phenotype_id, self.ic_dict[matched_phenotype_id])

                    matching_IC = sum(float(self.ic_dict.get(matched_phenotype_id, 0)) for matched_phenotype_id in matching_HPO)

                    # Calculate total phenotype associations based on data source
                    if use_samples:
                        total_phenotype_associations = len(self.disease_to_phenotypes_from_trueSamples.get(disease_id, set()))
                    else:
                        total_phenotype_associations = self.disease_phenotype_counts[disease_id]

                    
                    disease_id_with_same_name = set()
                    for disease_name, disease_ids in self.disease_name_to_ids.items():
                        if disease_id in disease_ids:
                            disease_id_with_same_name.update(disease_ids)

                    # finish embedding similarity for disease vs patient phenotypes
                    case_similarity = self.calculate_case_similarity(phenotypes, disease_id)

                    if total_IC > 0:
                        matching_IC = matching_IC/total_IC
                    else:
                        matching_IC = 0.0
                    
                    # # total_frequency = matched sum + unmatched penalty
                    # # matched sum (computed)
                    # matched_frequency_sum = matching_count_weighted
                    # # unmatched: patient has, disease not (reuse matching_HPO)
                    # matched_phenotype_ids = set(matching_HPO)
                    # unmatched_phenotypes = [p for p in phenotypes if p not in matched_phenotype_ids]
                    # unmatched_penalty = len(unmatched_phenotypes) * 1.0 # default 0.5
                    # # matching_count_weighted: matched/(matched+penalty) in [0,1]
                    # matching_count_weighted = matched_frequency_sum / (matched_frequency_sum + unmatched_penalty + 1e-10)
                    
                    # matching_count_weighted = matching_count_weighted/(len(phenotypes)*1.0)
                    # matching_count_weighted = 0.0
                    
                    if use_frequency_weights or use_score:
                        # get LR posterior from cache (avoid recompute)
                        # disease_id may be absent in lr_results_cache; get via same-name id
                        disease_id_upper = disease_id.upper()
                        if disease_id_upper in lr_results_cache:
                            posttest_probability = lr_results_cache[disease_id_upper]
                        else:
                            # if disease_id not in cache, try same-name disease IDs
                            posttest_probability = 0.0
                            if disease_id_with_same_name:
                                for same_name_id in disease_id_with_same_name:
                                    same_name_id_upper = same_name_id.upper()
                                    if same_name_id_upper in lr_results_cache:
                                        posttest_probability = lr_results_cache[same_name_id_upper]
                                        break
                        matching_count_weighted = posttest_probability
                    else:
                        matching_count_weighted = 0.0
                    
                    score = matching_IC + case_similarity + matching_count_weighted

                    disease_name = "; ".join(self.get_disease_all_names(disease_id))

                    disease_info[disease_id] = {
                        'disease_id': [disease_id],
                        'disease_name': disease_name, # self.get_disease_name(disease_id),
                        'disease_id_with_same_name': list(disease_id_with_same_name),
                        'disease_synonyms': list(self.disease_mapping_with_synonyms.get(disease_id, [])),   
                        'total_phenotype_associations': total_phenotype_associations,
                        'matching_phenotypes': matching_phenotypes_with_freq,
                        'matching_phenotype_count': matching_count,
                        'matching_phenotype_count_weighted': matching_count_weighted,
                        'matching_phenotype_IC': matching_IC,
                        'case_similarity': case_similarity,
                        'score': score
                    }
                    
                    # Use matching count as score for ranking
                    disease_scores[disease_id] = matching_count
        

        # Convert to list and sort by matching count (descending)
        ranked_diseases = []
        for disease_id, matching_count in disease_scores.items():
            disease_data = disease_info[disease_id].copy()
            ranked_diseases.append(disease_data)
        
        # # dedup ranked_diseases by name after ; split
        # disease_name_to_id = {}  # name->id
        unique_diseases = []  # unique disease IDs
        
        # method 1: dedup by name (deprecated)
        # for disease_info in ranked_diseases:
        #     disease_id = disease_info['disease_id']
        #     disease_name = self.get_disease_name(disease_id)
        #     disease_name_list = self._split_disease_names(disease_name)
            
        #     # check duplicate names
        #     has_duplicate = False
        #     for disease_name_part in disease_name_list:
        #         if disease_name_part in disease_name_to_id:
        #             # if dup, keep first
        #             has_duplicate = True
        #             break
            
        #     if not has_duplicate:
        #         # if no dup, add
        #         unique_diseases.append(disease_info)
        #         # update mapping
        #         for disease_name_part in disease_name_list:
        #             disease_name_to_id[disease_name_part] = disease_id

        # method 2: dedup by disease_id_with_same_name
        processed_disease_ids = set()  # processed IDs
        
        for disease_info in ranked_diseases:
            disease_id_with_same_name = set(disease_info['disease_id_with_same_name'])
            
            # check duplicate (incl same-name)
            # use set intersection, efficient and clear
            has_duplicate = bool(disease_id_with_same_name & processed_disease_ids)
            
            if not has_duplicate:
                # if no dup, add
                unique_diseases.append(disease_info)
                # mark same-name IDs processed
                processed_disease_ids.update(disease_id_with_same_name)
        
        ranked_diseases = unique_diseases

        # print(f"ranked_diseases: {len(ranked_diseases)}")

        # Sort by matching phenotype count (descending), then by total phenotype associations (ascending - fewer is better), then by disease_id for stability
        # Note: total_phenotype_associations is sorted in ascending order because fewer associations means more specific disease
        # ranked_diseases.sort(key=lambda x: (x['matching_phenotype_count'], -x['total_phenotype_associations'], x['disease_id']), reverse=True)
        
        # TEST
        if use_IC_weights and use_frequency_weights:
            use_frequency_weights = False

        if use_IC_weights:
            ranked_diseases.sort(key=lambda x: (x['matching_phenotype_IC'], x['matching_phenotype_count'], -x['total_phenotype_associations']), reverse=True)
        elif use_frequency_weights:
            ranked_diseases.sort(key=lambda x: (x['matching_phenotype_count_weighted'], x['matching_phenotype_count'], -x['total_phenotype_associations']), reverse=True)
        elif use_score:
            ranked_diseases.sort(key=lambda x: (x['score'], x['matching_phenotype_count'], -x['total_phenotype_associations']), reverse=True)
        else:
            ranked_diseases.sort(key=lambda x: (x['matching_phenotype_count'], x['matching_phenotype_IC'], -x['total_phenotype_associations']), reverse=True)


        return ranked_diseases
    

    def rank_diseases_by_phenotype_associations_from_kg_local(self, phenotypes: List[str], use_frequency_weights: bool = False, use_IC_weights: bool = False, use_score: bool = False, use_samples: bool = False, exclude_sample_id: int = None, disease_examples: List[str] = None) -> List[Dict]:
        """
        Rank diseases based on their association count with the given phenotypes.
        All data is retrieved from local knowledge graph files in disease_phenotype_kg folder.
        
        Args:
            phenotypes: List of phenotype IDs (HPO terms)
            use_frequency_weights: Whether to use frequency weights to rank diseases
            use_IC_weights: Whether to use IC weights to rank diseases
            use_score: Whether to use combined score (matching_count_weighted + matching_IC + case_similarity) to rank diseases
            use_samples: Whether to use case_library JSONL data instead of local KG data
            exclude_sample_id: Whether to exclude a specific sample ID
            disease_examples: List of disease IDs that are associated with the given phenotypes
            
        Returns:
            List of dictionaries containing disease info and ranking scores, sorted by score
        """
        
        # load local KG (instance cache, load once)
        if not self._local_kg_data_loaded or self._local_kg_data_cache is None:
            self._local_kg_data_cache = LocalKnowledgeGraphQuery.load_kg_data(self.config)
            self._local_kg_data_loaded = True
        kg_data = self._local_kg_data_cache
        
        # extract to local vars
        phenotype_to_diseases = kg_data['phenotype_to_diseases']
        disease_to_phenotypes = kg_data['disease_to_phenotypes']
        disease_phenotype_counts = kg_data['disease_phenotype_counts']
        ic_dict = kg_data['ic_dict']
        
        # create local KG query instance
        local_kg_query = LocalKnowledgeGraphQuery(kg_data)
        
        # main logic (same as rank_diseases_by_phenotype_associations_from_kg)
        disease_scores = defaultdict(int)
        disease_info = {}
        
        # Build phenotype_to_diseases_from_kg and disease_to_phenotypes_from_kg from local KG
        phenotype_to_diseases_from_kg = defaultdict(set)
        disease_to_phenotypes_from_kg = defaultdict(set)
        disease_kg_info_cache = {}  # Cache disease info from local KG to avoid repeated queries
        phenotype_ic_cache = {}  # Cache phenotype IC values from local KG
        phenotype_info_cache = {}  # Cache complete phenotype info (including embedding and IC) from local KG
        
        # Get all parent phenotypes (including grandparents, great-grandparents, etc.) for each phenotype from local KG
        parents_phenotypes = {}
        for phenotype in phenotypes:
            all_parents = local_kg_query.get_all_parent_phenotypes(phenotype)
            parents_phenotypes[phenotype] = list(all_parents)
        
        # Get IC values and full phenotype info from local KG for all phenotypes
        for phenotype_id in phenotypes:
            if phenotype_id not in phenotype_ic_cache:
                # Directly use local_kg_query attributes
                phenotype_ic_cache[phenotype_id] = float(local_kg_query.ic_dict.get(phenotype_id, 0.0))
                # Also cache full phenotype info for embedding calculation
                phenotype_info_cache[phenotype_id] = {
                    'ID': phenotype_id,
                    'IC': str(local_kg_query.ic_dict.get(phenotype_id, 0.0)),
                    'embedding': local_kg_query.phe2embedding.get(phenotype_id, [])
                }
        
        # Calculate total IC from local KG
        total_IC = sum(phenotype_ic_cache.get(phenotype_id, 0.0) for phenotype_id in phenotypes)
        
        # Query local KG for diseases associated with each phenotype
        for phenotype_id in phenotypes:
            associated_diseases = phenotype_to_diseases.get(phenotype_id, set())
            
            # Extract disease IDs from local KG result
            for disease_id in associated_diseases:
                phenotype_to_diseases_from_kg[phenotype_id].add(disease_id)
                
                # Cache disease KG info if not already cached
                if disease_id not in disease_kg_info_cache:
                    disease_kg_info_cache[disease_id] = local_kg_query.get_disease_exp_info_from_local(disease_id)
                
                # Get phenotypes for this disease from local KG
                disease_kg_info = disease_kg_info_cache[disease_id]
                disease_phenotypes = disease_kg_info.get('phenotypes', [])
                disease_to_phenotypes_from_kg[disease_id].update(disease_phenotypes)
            
            # BUGFIX: if phenotype has no diseases, get parents and their diseases
            if not phenotype_to_diseases_from_kg[phenotype_id] and phenotype_id in parents_phenotypes:
                for p_id in parents_phenotypes[phenotype_id]:
                    parent_associated_diseases = phenotype_to_diseases.get(p_id, set())
                    for disease_id in parent_associated_diseases:
                        phenotype_to_diseases_from_kg[phenotype_id].add(disease_id)
                        
                        if disease_id not in disease_kg_info_cache:
                            disease_kg_info_cache[disease_id] = local_kg_query.get_disease_exp_info_from_local(disease_id)
                        
                        disease_kg_info = disease_kg_info_cache[disease_id]
                        disease_phenotypes = disease_kg_info.get('phenotypes', [])
                        disease_to_phenotypes_from_kg[disease_id].update(disease_phenotypes)
        
        # Calculate scores for each disease based on phenotype associations
        for phenotype_id in phenotypes:
            # Get diseases associated with this phenotype from local KG
            associated_diseases = phenotype_to_diseases_from_kg.get(phenotype_id, set())
            
            for disease_id in associated_diseases:
                # Store disease info if not already stored
                if disease_id not in disease_info:
                    
                    # find same-name id group for disease_id; if in disease_info, add to disease_info['disease_id']
                    # For local KG version, we check by disease name from local KG instead of disease_name_to_ids
                    have_added = False
                    disease_kg_info = disease_kg_info_cache.get(disease_id, {})
                    current_disease_name_from_kg = disease_kg_info.get('standard_name', '').lower().strip()
                    
                    if current_disease_name_from_kg:
                        for existing_disease_id, existing_info in disease_info.items():
                            existing_disease_kg_info = disease_kg_info_cache.get(existing_disease_id, {})
                            existing_disease_name_from_kg = existing_disease_kg_info.get('standard_name', '').lower().strip()
                            
                            if existing_disease_name_from_kg == current_disease_name_from_kg:
                                if disease_id not in existing_info['disease_id']:
                                    existing_info['disease_id'].append(disease_id)
                                have_added = True
                                break
                    
                    if have_added:
                        continue
                    
                    # Calculate matching phenotypes for this disease from local KG
                    matching_phenotypes = list(disease_to_phenotypes_from_kg[disease_id].intersection(set(phenotypes)))
                    
                    # Check for phenotype matches through parent relationships
                    matching_phenotypes_by_parents = []
                    matching_phenotypes_by_parents_map = {}
                    
                    # parents_phenotypes_in_disease = set()  # Use set for efficient lookup
                    
                    # Collect all parent phenotypes for this disease from local KG
                    disease_phenotypes = list(disease_to_phenotypes_from_kg[disease_id])
                    # for phenotype in disease_phenotypes:
                    #     # Get parent phenotypes from local KG
                    #     phenotype_parents = local_kg_query.get_all_parent_phenotypes(phenotype)
                    #     parents_phenotypes_in_disease.update(phenotype_parents)
                    
                    # Check if any patient phenotype's parents match disease phenotype parents
                    for phenotype, parent_list in parents_phenotypes.items():
                        if parent_list:
                            for parent in parent_list:
                                if parent in disease_phenotypes:
                                    matching_phenotypes_by_parents.append(phenotype + " (by parent)")
                                    matching_phenotypes_by_parents_map[phenotype] = parent
                                    break
                    
                    # dedup
                    # dedup (by parent) in matching_phenotypes_by_parents
                    for phenotype in matching_phenotypes_by_parents:
                        if '(' in phenotype:
                            phenotype_id_clean = phenotype.split('(')[0].strip()
                        else:
                            phenotype_id_clean = phenotype.strip()
                            
                        if phenotype_id_clean not in matching_phenotypes:
                            matching_phenotypes.append(phenotype)
                    
                    # ---------------------------------------------
                    # check if any phenotype in given phenotypes is a parent phenotype with only one child
                    # Use local KG to get children
                    matching_phenotypes_by_child_map = {}
                    for phenotype in phenotypes:
                        if (phenotype in matching_phenotypes) or (phenotype + " (by parent)" in matching_phenotypes):
                            continue
                        # Get children from local KG - directly use parent_to_children
                        children = local_kg_query.parent_to_children.get(phenotype, [])
                        if len(children) == 1:
                            child_phenotype = children[0]
                            if child_phenotype in disease_phenotypes:
                                matching_phenotypes.append(phenotype + " (by child)")
                                matching_phenotypes_by_child_map[phenotype] = child_phenotype
                    # ---------------------------------------------
                    
                    # Create matching phenotypes with frequency annotations
                    matching_phenotypes_with_freq = []
                    matching_phenotypes_freq_info = []
                    # listcomp to avoid modify-while-iterate
                    filtered_matching_phenotypes = []
                    
                    # Get disease KG info for frequency information
                    disease_kg_info = disease_kg_info_cache.get(disease_id, {})
                    phenotype_max_frequencies = disease_kg_info.get('phenotype_max_frequencies', [])
                    disease_kg_phenotypes = disease_kg_info.get('phenotypes', [])
                    # Create mapping from phenotype to frequency
                    phenotype_to_frequency = {}
                    for idx, ph in enumerate(disease_kg_phenotypes):
                        if idx < len(phenotype_max_frequencies):
                            phenotype_to_frequency[ph] = phenotype_max_frequencies[idx]
                    
                    for matched_phenotype_id in matching_phenotypes:
                        phenotype_id_original = matched_phenotype_id
                        matched_phenotype_id_for_freq = matched_phenotype_id
                        if 'by parent' in matched_phenotype_id:
                            matched_phenotype_id_for_freq = matching_phenotypes_by_parents_map[matched_phenotype_id.split(' (by parent)')[0].strip()]
                        elif 'by child' in matched_phenotype_id:
                            matched_phenotype_id_for_freq = matching_phenotypes_by_child_map[matched_phenotype_id.split(' (by child)')[0].strip()]
                        
                        # if matched is parent, count its children
                        # Get children from local KG
                        child_count = 1
                        if 'by parent' in phenotype_id_original:
                            parent_phenotype_id = phenotype_id_original.split(' (by parent)')[0].strip()
                            children = local_kg_query.parent_to_children.get(parent_phenotype_id, [])
                            child_count = len(children) if children else 1
                            if child_count == 0:
                                child_count = 1
                        
                        # Get frequency from local KG data
                        frequency_max = phenotype_to_frequency.get(matched_phenotype_id_for_freq, '')
                        
                        if frequency_max:
                            # Convert frequency string to float (frequency_max is already the max frequency)
                            try:
                                frequency_display = float(frequency_max)
                                
                                if frequency_display > 0:
                                    phenotype_with_freq = f"{phenotype_id_original} ({frequency_display})"
                                    matching_phenotypes_with_freq.append(phenotype_with_freq)
                                    matching_phenotypes_freq_info.append(frequency_display)
                                    filtered_matching_phenotypes.append(phenotype_id_original)
                                else:
                                    phenotype_with_freq = phenotype_id_original
                                    matching_phenotypes_with_freq.append(phenotype_with_freq)
                                    matching_phenotypes_freq_info.append(0.5)
                                    filtered_matching_phenotypes.append(phenotype_id_original)
                            except:
                                # If conversion fails, use default
                                phenotype_with_freq = phenotype_id_original
                                matching_phenotypes_with_freq.append(phenotype_with_freq)
                                matching_phenotypes_freq_info.append(0.5)
                                filtered_matching_phenotypes.append(phenotype_id_original)
                        else:
                            # No frequency information from local KG, use default
                            phenotype_with_freq = phenotype_id_original
                            matching_phenotypes_with_freq.append(phenotype_with_freq)
                            matching_phenotypes_freq_info.append(0.5)
                            filtered_matching_phenotypes.append(phenotype_id_original)
                    
                    # update matching_phenotypes to filtered list
                    matching_phenotypes = filtered_matching_phenotypes
                    
                    matching_count = len(matching_phenotypes)
                    matching_count_weighted = sum(matching_phenotypes_freq_info)
                    
                    # skip if disease name already exists (using local KG data)
                    disease_kg_info = disease_kg_info_cache.get(disease_id, {})
                    current_disease_name = disease_kg_info.get('standard_name', '')
                    if not current_disease_name:
                        current_disease_name = disease_id  # Fallback to disease_id if no name from local KG
                    
                    existing_disease_names = {info['disease_name'] for info in disease_info.values()}
                    if current_disease_name in existing_disease_names:
                        continue
                    
                    matching_HPO = []
                    for matched_phenotype_id in matching_phenotypes:
                        if '(' in matched_phenotype_id:
                            matching_HPO.append(matched_phenotype_id.split('(')[0].strip())
                        else:
                            matching_HPO.append(matched_phenotype_id.strip())
                    
                    # Calculate matching IC from local KG
                    # Get IC values from local KG cache, query local KG if not cached
                    matching_IC = 0.0
                    for matched_phenotype_id in matching_HPO:
                        if matched_phenotype_id not in phenotype_ic_cache:
                            # Directly use local_kg_query attributes
                            phenotype_ic_cache[matched_phenotype_id] = float(local_kg_query.ic_dict.get(matched_phenotype_id, 0.0))
                            # Also cache full phenotype info for embedding calculation
                            phenotype_info_cache[matched_phenotype_id] = {
                                'ID': matched_phenotype_id,
                                'IC': str(local_kg_query.ic_dict.get(matched_phenotype_id, 0.0)),
                                'embedding': local_kg_query.phe2embedding.get(matched_phenotype_id, [])
                            }
                        matching_IC += phenotype_ic_cache.get(matched_phenotype_id, 0.0)
                    
                    # Calculate total phenotype associations from local KG
                    total_phenotype_associations = len(disease_to_phenotypes_from_kg.get(disease_id, set()))
                    
                    # Build disease_id_with_same_name from local KG data instead of disease_name_to_ids
                    disease_id_with_same_name = set([disease_id])
                    disease_kg_info = disease_kg_info_cache.get(disease_id, {})
                    current_disease_name_from_kg = disease_kg_info.get('standard_name', '').lower().strip()
                    
                    if current_disease_name_from_kg:
                        # Find all diseases with the same name from local KG cache
                        for other_disease_id, other_kg_info in disease_kg_info_cache.items():
                            if other_disease_id != disease_id:
                                other_disease_name_from_kg = other_kg_info.get('standard_name', '').lower().strip()
                                if other_disease_name_from_kg == current_disease_name_from_kg:
                                    disease_id_with_same_name.add(other_disease_id)
                    
                    # finish embedding similarity for disease vs patient phenotypes
                    # For local KG version, we get disease phenotypes from local KG for similarity calculation
                    disease_phenotypes_for_similarity = list(disease_to_phenotypes_from_kg.get(disease_id, set()))
                    # Calculate similarity directly using local_kg_query attributes
                    case_similarity = 0.0
                    if local_kg_query.phe2embedding and local_kg_query.ic_dict:
                        # weighted embedding for patient phenotypes
                        patient_embeddings = []
                        patient_ic_values = []
                        for phe in phenotypes:
                            if phe in local_kg_query.phe2embedding and phe in local_kg_query.ic_dict:
                                patient_embeddings.append(local_kg_query.phe2embedding[phe])
                                patient_ic_values.append(local_kg_query.ic_dict[phe])
                        
                        if patient_embeddings:
                            patient_embeddings = np.array(patient_embeddings)
                            patient_ic_values = np.array(patient_ic_values)
                            
                            # normalize IC as weight
                            if patient_ic_values.sum() > 0:
                                weights = patient_ic_values / patient_ic_values.sum()
                            else:
                                weights = np.ones(len(patient_ic_values)) / len(patient_ic_values)
                            
                            patient_embedding = np.average(patient_embeddings, axis=0, weights=weights)
                            patient_embedding = patient_embedding / (np.linalg.norm(patient_embedding) + 1e-10)
                            
                            # weighted embedding for disease phenotypes
                            disease_embeddings = []
                            disease_ic_values = []
                            for phe in disease_phenotypes_for_similarity:
                                if phe in local_kg_query.phe2embedding and phe in local_kg_query.ic_dict:
                                    disease_embeddings.append(local_kg_query.phe2embedding[phe])
                                    disease_ic_values.append(local_kg_query.ic_dict[phe])
                            
                            if disease_embeddings:
                                disease_embeddings = np.array(disease_embeddings)
                                disease_ic_values = np.array(disease_ic_values)
                                
                                if disease_ic_values.sum() > 0:
                                    weights = disease_ic_values / disease_ic_values.sum()
                                else:
                                    weights = np.ones(len(disease_ic_values)) / len(disease_ic_values)
                                
                                disease_embedding = np.average(disease_embeddings, axis=0, weights=weights)
                                disease_embedding = disease_embedding / (np.linalg.norm(disease_embedding) + 1e-10)
                                
                                # compute cosine similarity
                                try:
                                    case_similarity = float(np.dot(patient_embedding, disease_embedding))
                                    if not np.isfinite(case_similarity):
                                        case_similarity = 0.0
                                except:
                                    case_similarity = 0.0
                    
                    matching_IC = matching_IC/total_IC if total_IC > 0 else 0
                    
                    matching_count_weighted = matching_count_weighted/(len(phenotypes)*1.0) if len(phenotypes) > 0 else 0
                    
                    score = matching_IC + case_similarity + matching_count_weighted
                    
                    # Get disease name from local KG info (all from local KG, no other data)
                    disease_kg_info = disease_kg_info_cache.get(disease_id, {})
                    disease_name = disease_kg_info.get('standard_name', '')
                    if not disease_name:
                        # If no standard_name, try to construct from synonyms
                        synonyms = disease_kg_info.get('synonyms', [])
                        if synonyms:
                            disease_name = "; ".join(synonyms)
                        else:
                            disease_name = disease_id  # Fallback to disease_id
                    
                    disease_info[disease_id] = {
                        'disease_id': [disease_id],
                        'disease_name': disease_name,
                        'disease_id_with_same_name': list(disease_id_with_same_name),
                        'disease_synonyms': disease_kg_info.get('synonyms', []),
                        'total_phenotype_associations': total_phenotype_associations,
                        'matching_phenotypes': matching_phenotypes_with_freq,
                        'matching_phenotype_count': matching_count,
                        'matching_phenotype_count_weighted': matching_count_weighted,
                        'matching_phenotype_IC': matching_IC,
                        'case_similarity': case_similarity,
                        'score': score
                    }
                    
                    # Use matching count as score for ranking
                    disease_scores[disease_id] = matching_count
        
        # Convert to list and sort by matching count (descending)
        ranked_diseases = []
        for disease_id, matching_count in disease_scores.items():
            disease_data = disease_info[disease_id].copy()
            ranked_diseases.append(disease_data)
        
        unique_diseases = []  # unique disease IDs
        
        # method 2: dedup by disease_id_with_same_name
        processed_disease_ids = set()  # processed IDs
        
        for disease_info_item in ranked_diseases:
            disease_id_with_same_name = set(disease_info_item['disease_id_with_same_name'])
            
            # check duplicate (incl same-name)
            # use set intersection, efficient and clear
            has_duplicate = bool(disease_id_with_same_name & processed_disease_ids)
            
            if not has_duplicate:
                # if no dup, add
                unique_diseases.append(disease_info_item)
                # mark same-name IDs processed
                processed_disease_ids.update(disease_id_with_same_name)
        
        ranked_diseases = unique_diseases
        
        # Sort by matching phenotype count (descending), then by total phenotype associations (ascending - fewer is better), then by disease_id for stability
        if use_IC_weights and use_frequency_weights:
            use_frequency_weights = False
        
        if use_IC_weights:
            ranked_diseases.sort(key=lambda x: (x['matching_phenotype_IC'], x['matching_phenotype_count'], -x['total_phenotype_associations']), reverse=True)
        elif use_frequency_weights:
            ranked_diseases.sort(key=lambda x: (x['matching_phenotype_count_weighted'], x['matching_phenotype_count'], -x['total_phenotype_associations']), reverse=True)
        elif use_score:
            ranked_diseases.sort(key=lambda x: (x['score'], x['matching_phenotype_count'], -x['total_phenotype_associations']), reverse=True)
        else:
            ranked_diseases.sort(key=lambda x: (x['matching_phenotype_count'], x['matching_phenotype_IC'], -x['total_phenotype_associations']), reverse=True)
        
        return ranked_diseases

    def rank_diseases_by_phenotype_associations_from_kg(self, phenotypes: List[str], use_frequency_weights: bool = False, use_IC_weights: bool = False, use_score: bool = False, use_samples: bool = False, exclude_sample_id: int = None, disease_examples: List[str] = None) -> List[Dict]:
        """
        Rank diseases based on their association count with the given phenotypes.
        All data is retrieved from knowledge graph (KG).
        
        Args:
            phenotypes: List of phenotype IDs (HPO terms)
            use_frequency_weights: Whether to use frequency weights to rank diseases
            use_IC_weights: Whether to use IC weights to rank diseases
            use_score: Whether to use combined score (matching_count_weighted + matching_IC + case_similarity) to rank diseases
            
        Returns:
            List of dictionaries containing disease info and ranking scores, sorted by score
        """
        
        disease_scores = defaultdict(int)
        disease_info = {}
        
        # Build phenotype_to_diseases_from_kg and disease_to_phenotypes_from_kg from KG
        phenotype_to_diseases_from_kg = defaultdict(set)
        disease_to_phenotypes_from_kg = defaultdict(set)
        disease_kg_info_cache = {}  # Cache disease info from KG to avoid repeated queries
        phenotype_ic_cache = {}  # Cache phenotype IC values from KG to avoid repeated queries
        phenotype_info_cache = {}  # Cache complete phenotype info (including embedding and IC) from KG
        
        # Get all parent phenotypes (including grandparents, great-grandparents, etc.) for each phenotype from KG
        parents_phenotypes = {}
        for phenotype in phenotypes:
            all_parents = self.get_phenotype_ancestors_from_kg(phenotype)
            parents_phenotypes[phenotype] = list(all_parents)
        
        # Get IC values and full phenotype info from KG for all phenotypes
        for phenotype_id in phenotypes:
            if phenotype_id not in phenotype_ic_cache:
                phenotype_info = self.get_phenotype_info_from_kg(phenotype_id)
                phenotype_ic_cache[phenotype_id] = float(phenotype_info.get('IC', 0.0))
                # Also cache full phenotype info for embedding calculation
                phenotype_info_cache[phenotype_id] = phenotype_info
        
        # Calculate total IC from KG
        total_IC = sum(phenotype_ic_cache.get(phenotype_id, 0.0) for phenotype_id in phenotypes)
        
        # Query KG for diseases associated with each phenotype
        for phenotype_id in phenotypes:
            diseases_from_kg = self.get_diseases_from_hpo_id_from_kg(phenotype_id)
            
            # Extract disease IDs from KG result
            # diseases_from_kg returns dict: {aggregated_disease_id: {disease_id: [list of public IDs], ...}}
            for aggregated_disease_id, disease_kg_data in diseases_from_kg.items():
                # Get all public disease IDs for this aggregated disease
                public_disease_ids = disease_kg_data.get('disease_id', [])
                if not public_disease_ids:
                    # If no public IDs, use aggregated ID
                    public_disease_ids = [aggregated_disease_id]
                
                for disease_id in public_disease_ids:
                    phenotype_to_diseases_from_kg[phenotype_id].add(disease_id)
                    
                    # Cache disease KG info if not already cached
                    if disease_id not in disease_kg_info_cache:
                        disease_kg_info_cache[disease_id] = self.get_disease_exp_info_from_kg(disease_id)
                    
                    # Get phenotypes for this disease from KG
                    disease_kg_info = disease_kg_info_cache[disease_id]
                    disease_phenotypes = disease_kg_info.get('phenotypes', [])
                    disease_to_phenotypes_from_kg[disease_id].update(disease_phenotypes)
            
            # BUGFIX: if phenotype has no diseases, get parents and their diseases
            if not phenotype_to_diseases_from_kg[phenotype_id] and phenotype_id in parents_phenotypes:
                for p_id in parents_phenotypes[phenotype_id]:
                    parent_diseases_from_kg = self.get_diseases_from_hpo_id_from_kg(p_id)
                    for aggregated_disease_id, disease_kg_data in parent_diseases_from_kg.items():
                        public_disease_ids = disease_kg_data.get('disease_id', [])
                        if not public_disease_ids:
                            public_disease_ids = [aggregated_disease_id]
                        
                        for disease_id in public_disease_ids:
                            phenotype_to_diseases_from_kg[phenotype_id].add(disease_id)
                            
                            if disease_id not in disease_kg_info_cache:
                                disease_kg_info_cache[disease_id] = self.get_disease_exp_info_from_kg(disease_id)
                            
                            disease_kg_info = disease_kg_info_cache[disease_id]
                            disease_phenotypes = disease_kg_info.get('phenotypes', [])
                            disease_to_phenotypes_from_kg[disease_id].update(disease_phenotypes)
        
        # Calculate scores for each disease based on phenotype associations
        for phenotype_id in phenotypes:
            # Get diseases associated with this phenotype from KG
            associated_diseases = phenotype_to_diseases_from_kg.get(phenotype_id, set())
            
            for disease_id in associated_diseases:
                # Store disease info if not already stored
                if disease_id not in disease_info:
                    
                    # find same-name id group for disease_id; if same-name in disease_info, add id to disease_info['disease_id']
                    # For KG version, we check by disease name from KG instead of local disease_name_to_ids
                    have_added = False
                    disease_kg_info = disease_kg_info_cache.get(disease_id, {})
                    current_disease_name_from_kg = disease_kg_info.get('standard_name', '').lower().strip()
                    
                    if current_disease_name_from_kg:
                        for existing_disease_id, existing_info in disease_info.items():
                            existing_disease_kg_info = disease_kg_info_cache.get(existing_disease_id, {})
                            existing_disease_name_from_kg = existing_disease_kg_info.get('standard_name', '').lower().strip()
                            
                            if existing_disease_name_from_kg == current_disease_name_from_kg:
                                if disease_id not in existing_info['disease_id']:
                                    existing_info['disease_id'].append(disease_id)
                                have_added = True
                                break
                    
                    if have_added:
                        continue
                    
                    # Calculate matching phenotypes for this disease from KG
                    matching_phenotypes = list(disease_to_phenotypes_from_kg[disease_id].intersection(set(phenotypes)))
                    
                    # Check for phenotype matches through parent relationships
                    matching_phenotypes_by_parents = []
                    matching_phenotypes_by_parents_map = {}
                    
                    # parents_phenotypes_in_disease = set()  # Use set for efficient lookup
                    
                    # Collect all parent phenotypes for this disease from KG
                    disease_phenotypes = list(disease_to_phenotypes_from_kg[disease_id])
                    # for phenotype in disease_phenotypes:
                    #     # Get parent phenotypes from KG
                    #     phenotype_parents = self.get_phenotype_ancestors_from_kg(phenotype)
                    #     parents_phenotypes_in_disease.update(phenotype_parents)
                    
                    # Check if any patient phenotype's parents match disease phenotype parents
                    for phenotype, parent_list in parents_phenotypes.items():
                        if parent_list:
                            for parent in parent_list:
                                if parent in disease_phenotypes:
                                    matching_phenotypes_by_parents.append(phenotype + " (by parent)")
                                    matching_phenotypes_by_parents_map[phenotype] = parent
                                    break
                    
                    # dedup
                    # dedup (by parent) in matching_phenotypes_by_parents
                    for phenotype in matching_phenotypes_by_parents:
                        if '(' in phenotype:
                            phenotype_id_clean = phenotype.split('(')[0].strip()
                        else:
                            phenotype_id_clean = phenotype.strip()
                            
                        if phenotype_id_clean not in matching_phenotypes:
                            matching_phenotypes.append(phenotype)
                    
                    # ---------------------------------------------
                    # check if any phenotype in given phenotypes is a parent phenotype with only one child
                    # Use KG to get children instead of local parent_to_children
                    matching_phenotypes_by_child_map = {}
                    for phenotype in phenotypes:
                        if (phenotype in matching_phenotypes) or (phenotype + " (by parent)" in matching_phenotypes):
                            continue
                        # Get children from KG
                        children = self.get_phenotype_descendants_from_kg(phenotype, max_depth=1)
                        if len(children) == 1:
                            child_phenotype = children[0]
                            if child_phenotype in disease_phenotypes:
                                matching_phenotypes.append(phenotype + " (by child)")
                                matching_phenotypes_by_child_map[phenotype] = child_phenotype
                    # ---------------------------------------------
                    
                    # Create matching phenotypes with frequency annotations
                    matching_phenotypes_with_freq = []
                    matching_phenotypes_freq_info = []
                    # listcomp to avoid modify-while-iterate
                    filtered_matching_phenotypes = []
                    
                    # Get disease KG info for frequency information
                    disease_kg_info = disease_kg_info_cache.get(disease_id, {})
                    phenotype_max_frequencies = disease_kg_info.get('phenotype_max_frequencies', [])
                    disease_kg_phenotypes = disease_kg_info.get('phenotypes', [])
                    # Create mapping from phenotype to frequency
                    phenotype_to_frequency = {}
                    for idx, ph in enumerate(disease_kg_phenotypes):
                        if idx < len(phenotype_max_frequencies):
                            phenotype_to_frequency[ph] = phenotype_max_frequencies[idx]
                    
                    for matched_phenotype_id in matching_phenotypes:
                        phenotype_id_original = matched_phenotype_id
                        matched_phenotype_id_for_freq = matched_phenotype_id
                        if 'by parent' in matched_phenotype_id:
                            matched_phenotype_id_for_freq = matching_phenotypes_by_parents_map[matched_phenotype_id.split(' (by parent)')[0].strip()]
                        elif 'by child' in matched_phenotype_id:
                            matched_phenotype_id_for_freq = matching_phenotypes_by_child_map[matched_phenotype_id.split(' (by child)')[0].strip()]
                        
                        # if matched is parent, count its children
                        # Get children from KG instead of local parent_to_children
                        child_count = 1
                        if 'by parent' in phenotype_id_original:
                            parent_phenotype_id = phenotype_id_original.split(' (by parent)')[0].strip()
                            children = self.get_phenotype_descendants_from_kg(parent_phenotype_id, max_depth=1)
                            child_count = len(children) if children else 1
                            if child_count == 0:
                                child_count = 1
                        
                        # Get frequency from KG data
                        frequency_max = phenotype_to_frequency.get(matched_phenotype_id_for_freq, '')
                        
                        if frequency_max:
                            # Convert frequency string to float (frequency_max is already the max frequency)
                            try:
                                frequency_display = float(frequency_max)
                                
                                if frequency_display > 0:
                                    phenotype_with_freq = f"{phenotype_id_original} ({frequency_display})"
                                    matching_phenotypes_with_freq.append(phenotype_with_freq)
                                    matching_phenotypes_freq_info.append(frequency_display)
                                    filtered_matching_phenotypes.append(phenotype_id_original)
                                else:
                                    phenotype_with_freq = phenotype_id_original
                                    matching_phenotypes_with_freq.append(phenotype_with_freq)
                                    matching_phenotypes_freq_info.append(0.5)
                                    filtered_matching_phenotypes.append(phenotype_id_original)
                            except:
                                # If conversion fails, use default
                                phenotype_with_freq = phenotype_id_original
                                matching_phenotypes_with_freq.append(phenotype_with_freq)
                                matching_phenotypes_freq_info.append(0.5)
                                filtered_matching_phenotypes.append(phenotype_id_original)
                        else:
                            # No frequency information from KG, use default
                            phenotype_with_freq = phenotype_id_original
                            matching_phenotypes_with_freq.append(phenotype_with_freq)
                            matching_phenotypes_freq_info.append(0.5)
                            filtered_matching_phenotypes.append(phenotype_id_original)
                    
                    # update matching_phenotypes to filtered list
                    matching_phenotypes = filtered_matching_phenotypes
                    
                    matching_count = len(matching_phenotypes)
                    matching_count_weighted = sum(matching_phenotypes_freq_info)
                    
                    # skip if disease name already exists (using KG data)
                    disease_kg_info = disease_kg_info_cache.get(disease_id, {})
                    current_disease_name = disease_kg_info.get('standard_name', '')
                    if not current_disease_name:
                        current_disease_name = disease_id  # Fallback to disease_id if no name from KG
                    
                    existing_disease_names = {info['disease_name'] for info in disease_info.values()}
                    if current_disease_name in existing_disease_names:
                        continue
                    
                    matching_HPO = []
                    for matched_phenotype_id in matching_phenotypes:
                        if '(' in matched_phenotype_id:
                            matching_HPO.append(matched_phenotype_id.split('(')[0].strip())
                        else:
                            matching_HPO.append(matched_phenotype_id.strip())
                    
                    # Calculate matching IC from KG
                    # Get IC values from KG cache, query KG if not cached
                    matching_IC = 0.0
                    for matched_phenotype_id in matching_HPO:
                        if matched_phenotype_id not in phenotype_ic_cache:
                            phenotype_info = self.get_phenotype_info_from_kg(matched_phenotype_id)
                            phenotype_ic_cache[matched_phenotype_id] = float(phenotype_info.get('IC', 0.0))
                            # Also cache full phenotype info for embedding calculation
                            phenotype_info_cache[matched_phenotype_id] = phenotype_info
                        matching_IC += phenotype_ic_cache.get(matched_phenotype_id, 0.0)
                    
                    # Calculate total phenotype associations from KG
                    total_phenotype_associations = len(disease_to_phenotypes_from_kg.get(disease_id, set()))
                    
                    # Build disease_id_with_same_name from KG data instead of local disease_name_to_ids
                    disease_id_with_same_name = set([disease_id])
                    disease_kg_info = disease_kg_info_cache.get(disease_id, {})
                    current_disease_name_from_kg = disease_kg_info.get('standard_name', '').lower().strip()
                    
                    if current_disease_name_from_kg:
                        # Find all diseases with the same name from KG cache
                        for other_disease_id, other_kg_info in disease_kg_info_cache.items():
                            if other_disease_id != disease_id:
                                other_disease_name_from_kg = other_kg_info.get('standard_name', '').lower().strip()
                                if other_disease_name_from_kg == current_disease_name_from_kg:
                                    disease_id_with_same_name.add(other_disease_id)
                    
                    # finish embedding similarity for disease vs patient phenotypes
                    # For KG version, we get disease phenotypes from KG for similarity calculation
                    disease_phenotypes_for_similarity = list(disease_to_phenotypes_from_kg.get(disease_id, set()))
                    case_similarity = self.calculate_case_similarity_from_kg(phenotypes, disease_phenotypes_for_similarity, phenotype_info_cache)
                    
                    matching_IC = matching_IC/total_IC if total_IC > 0 else 0
                    
                    matching_count_weighted = matching_count_weighted/(len(phenotypes)*1.0) if len(phenotypes) > 0 else 0
                    
                    score = matching_IC + case_similarity + matching_count_weighted
                    
                    # Get disease name from KG info (all from KG, no local data)
                    disease_kg_info = disease_kg_info_cache.get(disease_id, {})
                    disease_name = disease_kg_info.get('standard_name', '')
                    if not disease_name:
                        # If no standard_name, try to construct from synonyms
                        synonyms = disease_kg_info.get('synonyms', [])
                        if synonyms:
                            disease_name = "; ".join(synonyms)
                        else:
                            disease_name = disease_id  # Fallback to disease_id
                    
                    disease_info[disease_id] = {
                        'disease_id': [disease_id],
                        'disease_name': disease_name,
                        'disease_id_with_same_name': list(disease_id_with_same_name),
                        'disease_synonyms': disease_kg_info.get('synonyms', []),
                        'total_phenotype_associations': total_phenotype_associations,
                        'matching_phenotypes': matching_phenotypes_with_freq,
                        'matching_phenotype_count': matching_count,
                        'matching_phenotype_count_weighted': matching_count_weighted,
                        'matching_phenotype_IC': matching_IC,
                        'case_similarity': case_similarity,
                        'score': score
                    }
                    
                    # Use matching count as score for ranking
                    disease_scores[disease_id] = matching_count
        
        # Convert to list and sort by matching count (descending)
        ranked_diseases = []
        for disease_id, matching_count in disease_scores.items():
            disease_data = disease_info[disease_id].copy()
            ranked_diseases.append(disease_data)
        
        unique_diseases = []  # unique disease IDs
        
        # method 2: dedup by disease_id_with_same_name
        processed_disease_ids = set()  # processed IDs
        
        for disease_info_item in ranked_diseases:
            disease_id_with_same_name = set(disease_info_item['disease_id_with_same_name'])
            
            # check duplicate (incl same-name)
            # use set intersection, efficient and clear
            has_duplicate = bool(disease_id_with_same_name & processed_disease_ids)
            
            if not has_duplicate:
                # if no dup, add
                unique_diseases.append(disease_info_item)
                # mark same-name IDs processed
                processed_disease_ids.update(disease_id_with_same_name)
        
        ranked_diseases = unique_diseases
        
        # Sort by matching phenotype count (descending), then by total phenotype associations (ascending - fewer is better), then by disease_id for stability
        if use_IC_weights and use_frequency_weights:
            use_frequency_weights = False
        
        if use_IC_weights:
            ranked_diseases.sort(key=lambda x: (x['matching_phenotype_IC'], x['matching_phenotype_count'], -x['total_phenotype_associations']), reverse=True)
        elif use_frequency_weights:
            ranked_diseases.sort(key=lambda x: (x['matching_phenotype_count_weighted'], x['matching_phenotype_count'], -x['total_phenotype_associations']), reverse=True)
        elif use_score:
            ranked_diseases.sort(key=lambda x: (x['score'], x['matching_phenotype_count'], -x['total_phenotype_associations']), reverse=True)
        else:
            ranked_diseases.sort(key=lambda x: (x['matching_phenotype_count'], x['matching_phenotype_IC'], -x['total_phenotype_associations']), reverse=True)
        
        return ranked_diseases
    
    def rank_samples_by_phenotype_associations(self, phenotypes: List[str], use_frequency_weights: bool = False, use_IC_weights: bool = False, use_score: bool = False, exclude_sample_id: int = None) -> List[Dict]:
        """
        Rank samples based on their association count with the given phenotypes.
        
        Args:
            phenotypes: List of phenotype IDs (HPO terms)
            use_frequency_weights: Whether to use frequency weights to rank samples
            use_IC_weights: Whether to use IC weights to rank samples
            use_score: Whether to use combined score (matching_count_weighted + matching_IC + case_similarity) to rank samples
            exclude_sample_id: Sample ID to exclude from ranking (for testing purposes)
            
        Returns:
            List of dictionaries containing sample info and ranking scores, sorted by score
        """
        
        # Load external samples if not already loaded
        if not self.external_samples:
            self._load_external_samples()
        
        if not self.external_samples:
            print("Warning: No external samples available")
            return []
        
        # Get all parent phenotypes (including grandparents, great-grandparents, etc.) for each phenotype
        parents_phenotypes = {}
        for phenotype in phenotypes:
            all_parents = self.get_all_parent_phenotypes(phenotype)
            parents_phenotypes[phenotype] = list(all_parents)
        
        total_IC = sum(float(self.ic_dict.get(phenotype_id, 0)) for phenotype_id in phenotypes)
        
        # Calculate scores for each sample based on phenotype associations
        sample_scores = []
        
        for sample in self.external_samples:
            # print(f"Processing sample: {sample.get('RareDisease')}")
            # Skip excluded sample if specified
            if exclude_sample_id:
                sample_source = sample.get('Department', '')
                if sample_source.split('_')[-1] == str(exclude_sample_id):
                    print(f"{exclude_sample_id} is excluded")
                    continue
            
            sample_phenotypes = sample.get('Phenotype', [])
            sample_phenotype_set = set(sample_phenotypes)
            
            # Calculate direct phenotype matches
            direct_matches = set(phenotypes).intersection(sample_phenotype_set)
            matching_phenotypes = list(direct_matches)
            
            # Check for phenotype matches through parent relationships
            matching_phenotypes_by_parents = []
            matching_phenotypes_by_parents_map = {}
            
            parents_phenotypes_in_sample = set()  # Use set for efficient lookup
            
            # Collect all parent phenotypes for this sample
            for phenotype in sample_phenotypes:
                phenotype_parents = self.hpo_is_a.get(phenotype, [])
                parents_phenotypes_in_sample.update(phenotype_parents)
            
            # Check if any patient phenotype's parents match sample phenotype parents
            for phenotype, parent_list in parents_phenotypes.items():
                if parent_list:
                    for parent in parent_list:
                        if parent in sample_phenotypes:
                            matching_phenotypes_by_parents.append(phenotype + " (by parent)")
                            matching_phenotypes_by_parents_map[phenotype] = parent
                            break
            
            # dedup
            # dedup (by parent) in matching_phenotypes_by_parents
            for phenotype in matching_phenotypes_by_parents:
                if '(' in phenotype:
                    phenotype_id = phenotype.split('(')[0].strip()
                else:
                    phenotype_id = phenotype.strip()
                    
                if phenotype_id not in matching_phenotypes:
                    matching_phenotypes.append(phenotype)
            
            # ---------------------------------------------
            # check if any phenotype in given phenotypes is a parent phenotype with only one child
            matching_phenotypes_by_child_map = {}
            for phenotype in phenotypes:
                if (phenotype in matching_phenotypes) or (phenotype + " (by parent)" in matching_phenotypes):
                    continue
                child_phenotype = self._is_hp_with_only_child(phenotype)
                if child_phenotype and child_phenotype in sample_phenotypes:
                    matching_phenotypes.append(phenotype + " (by child)")
                    matching_phenotypes_by_child_map[phenotype] = child_phenotype
            # ---------------------------------------------
            
            # Create matching phenotypes with frequency annotations
            matching_phenotypes_with_freq = []
            matching_phenotypes_freq_info = []
            # listcomp to avoid modify-while-iterate
            filtered_matching_phenotypes = []
            
            for matched_phenotype_id in matching_phenotypes:
                phenotype_id_original = matched_phenotype_id
                if 'by parent' in matched_phenotype_id:
                    matched_phenotype_id = matching_phenotypes_by_parents_map[matched_phenotype_id.split(' (by parent)')[0].strip()]
                elif 'by child' in matched_phenotype_id:
                    matched_phenotype_id = matching_phenotypes_by_child_map[matched_phenotype_id.split(' (by child)')[0].strip()]

                # Check frequency info from all diseases in the sample
                frequency_displays = []
                frequency_values = []
                for disease_id in sample.get('RareDisease', []):
                    freq_info = self.get_frequency_info(matched_phenotype_id, disease_id)
                    if not freq_info:
                        continue

                    if freq_info['frequency_type'] == 'hpo_id':
                        frequency_display = self._convert_hpo_frequency_to_description(freq_info['frequency'])
                    else:
                        frequency_display = freq_info['frequency']

                    if not frequency_display:
                        continue

                    try:
                        frequency_value = self.get_max_frequency_from_frequency_string(frequency_display)
                    except Exception:
                        continue

                    # For samples, we can optionally filter low frequency phenotypes
                    # if frequency_value < 0.17:
                    #     continue

                    frequency_displays.append(frequency_display)
                    frequency_values.append(frequency_value)

                if not frequency_values:
                    # For samples, use default frequency if not available
                    frequency_value_max = 0.5
                    frequency_text = ''
                else:
                    frequency_value_max = max(frequency_values)
                    unique_frequencies = list(set(frequency_displays))
                    frequency_text = '; '.join(unique_frequencies)

                phenotype_with_freq = f"{phenotype_id_original} ({frequency_text})" if frequency_text else phenotype_id_original
                matching_phenotypes_with_freq.append(phenotype_with_freq)
                matching_phenotypes_freq_info.append(frequency_value_max)
                filtered_matching_phenotypes.append(phenotype_id_original)
            
            # update matching_phenotypes to filtered list
            matching_phenotypes = filtered_matching_phenotypes
            
            matching_count = len(matching_phenotypes)
            matching_count_weighted = sum(matching_phenotypes_freq_info)
            
            matching_HPO = []
            for matched_phenotype_id in matching_phenotypes:
                if '(' in matched_phenotype_id:
                    matching_HPO.append(matched_phenotype_id.split('(')[0].strip())
                else:
                    matching_HPO.append(matched_phenotype_id.strip())
            
            matching_IC = sum(float(self.ic_dict.get(matched_phenotype_id, 0)) for matched_phenotype_id in matching_HPO)
            
            # Calculate case similarity for each disease in the sample
            # Use the maximum case similarity across all diseases in the sample
            case_similarities = []
            for disease_id in sample.get('RareDisease', []):
                case_similarity = self.calculate_case_similarity(phenotypes, disease_id)
                case_similarities.append(case_similarity)
            
            case_similarity = max(case_similarities) if case_similarities else 0.0
            
            # Normalize scores similar to disease ranking
            matching_IC = matching_IC / total_IC if total_IC > 0 else 0.0
            matching_count_weighted = matching_count_weighted/(len(phenotypes)*1.0)
            score = matching_count_weighted + matching_IC + case_similarity

            # disease_name = []
            # for disease_id in sample.get('RareDisease', []):
            #     disease_name.append(self.get_disease_name(disease_id))
            # disease_name = '; '.join(disease_name)
            
            disease_ids = sample.get('RareDisease', [])
            disease_ids = [disease_id for disease_id in disease_ids if disease_id.startswith('OMIM:') or disease_id.startswith('ORPHA:')]
            disease_name_set = set()
            for disease_id in disease_ids:       
                disease_name_set.update(self.get_disease_all_names(disease_id))
            disease_name = "; ".join(disease_name_set)

            # Create sample info
            sample_info = {
                'disease_id': disease_ids,  # Use same key as disease_info
                'disease_name': disease_name,  # Use same key as disease_info
                'total_phenotype_associations': len(sample_phenotypes),
                'matching_phenotypes': matching_phenotypes_with_freq,  # Now with frequency info
                'matching_phenotype_count': matching_count,
                'matching_phenotype_count_weighted': matching_count_weighted,
                'matching_phenotype_IC': matching_IC,
                'case_similarity': case_similarity,
                'score': score,
                'case_id': sample.get('Department')
            }
            
            sample_scores.append((sample_info, matching_count))
        
        if use_IC_weights and use_frequency_weights:
            use_frequency_weights = False

        # Sort by matching phenotype count (descending), then by IC or frequency weights if requested
        if use_IC_weights:
            sample_scores.sort(key=lambda x: (x[0]['matching_phenotype_IC'], x[0]['matching_phenotype_count'], -x[0]['total_phenotype_associations']), reverse=True)
        elif use_frequency_weights:
            sample_scores.sort(key=lambda x: (x[0]['matching_phenotype_count_weighted'], x[0]['matching_phenotype_count'], -x[0]['total_phenotype_associations']), reverse=True)
        elif use_score:
            sample_scores.sort(key=lambda x: (x[0]['score'], x[0]['matching_phenotype_count'], -x[0]['total_phenotype_associations']), reverse=True)
        else:
            sample_scores.sort(key=lambda x: (x[0]['matching_phenotype_count'], x[0]['matching_phenotype_IC'], -x[0]['total_phenotype_associations']), reverse=True)
        
        # dedup sample_scores by disease_id
        # Extract ranked samples while removing duplicate disease_ids, keeping the highest-ranked entry
        ranked_samples = []
        seen_disease_ids = set()

        for sample_info, _ in sample_scores:
            disease_ids = sample_info.get('disease_id')

            if isinstance(disease_ids, list):
                disease_key = tuple(sorted(disease_ids))
            elif disease_ids is None:
                disease_key = None
            else:
                disease_key = disease_ids

            if disease_key in seen_disease_ids:
                continue

            seen_disease_ids.add(disease_key)
            ranked_samples.append(sample_info)
        
        return ranked_samples

    # TODO: weight tuning pending
    def rank_diseases_by_frequency_weighted_associations(self, phenotypes: List[str]) -> List[Dict]:
        """
        Rank diseases based on their frequency-weighted association scores with the given phenotypes.
        
        Args:
            phenotypes: List of phenotype IDs (HPO terms)
            
        Returns:
            List of dictionaries containing disease info and weighted ranking scores, sorted by score
        """
        disease_scores = defaultdict(float)  # Use float for weighted scores
        disease_info = {}
        
        # Calculate weighted scores for each disease based on phenotype associations
        for phenotype_id in phenotypes:
            # Get diseases associated with this phenotype
            associated_diseases = self.phenotype_to_diseases.get(phenotype_id, set())
            
            for disease_id in associated_diseases:
                # Store disease info if not already stored
                if disease_id not in disease_info:
                    # Calculate matching phenotypes for this disease
                    matching_phenotypes = list(self.disease_to_phenotypes[disease_id].intersection(set(phenotypes)))
                    matching_count = len(matching_phenotypes)
                    
                    # Create matching phenotypes with frequency annotations and weights
                    matching_phenotypes_with_freq = []
                    total_weighted_score = 0.0
                    
                    for phenotype_id in matching_phenotypes:
                        freq_info = self.get_frequency_info(phenotype_id, disease_id)
                        if freq_info:
                            # Convert HPO ID frequencies to descriptions
                            if freq_info['frequency_type'] == 'hpo_id':
                                frequency_display = self._convert_hpo_frequency_to_description(freq_info['frequency'])
                            else:
                                frequency_display = freq_info['frequency']
                            
                            # Calculate weight for this phenotype-disease association
                            weight = self._get_frequency_weight(freq_info['frequency'], freq_info['frequency_type'])
                            total_weighted_score += weight
                            
                            # Add frequency and weight to phenotype ID
                            phenotype_with_freq = f"{phenotype_id} ({frequency_display}, weight={weight:.2f})"
                        else:
                            # No frequency info, use default weight
                            weight = 0.5  # Default weight for missing frequency
                            total_weighted_score += weight
                            phenotype_with_freq = f"{phenotype_id} (no_freq, weight={weight:.2f})"
                        
                        matching_phenotypes_with_freq.append(phenotype_with_freq)
                    
                    disease_info[disease_id] = {
                        'disease_id': disease_id,
                        'disease_name': self.get_disease_name(disease_id),
                        'total_phenotype_associations': self.disease_phenotype_counts[disease_id],
                        'matching_phenotypes': matching_phenotypes_with_freq,
                        'matching_phenotype_count': matching_count,
                        'weighted_score': total_weighted_score,
                        'average_weight': total_weighted_score / matching_count if matching_count > 0 else 0.0
                    }
                    
                    # Use weighted score for ranking
                    disease_scores[disease_id] = total_weighted_score
        
        # Convert to list and sort by weighted score (descending)
        ranked_diseases = []
        for disease_id, weighted_score in disease_scores.items():
            disease_data = disease_info[disease_id].copy()
            ranked_diseases.append(disease_data)
        
        # Sort by weighted score (descending), then by matching count (descending), then by total phenotype associations (ascending - fewer is better), then by disease_id for stability
        # Note: total_phenotype_associations is sorted in ascending order because fewer associations means more specific disease
        ranked_diseases.sort(key=lambda x: (x['weighted_score'], x['matching_phenotype_count'], -x['total_phenotype_associations'], x['disease_id']), reverse=True)
        
        return ranked_diseases

    def _disease_key(self, did):
        """Return a hashable key for disease_id (str or list) for use in sets."""
        if isinstance(did, list):
            return tuple(sorted(did)) if did else ()
        return (did,) if did is not None and did != '' else ()

    def _get_ranked_diseases_for_phenotypes(self, phenotypes: List[str], use_frequency_weights: bool = False, use_IC_weights: bool = False, use_score: bool = False, use_samples: bool = False, exclude_sample_id: int = None) -> List[Dict]:
        """
        Helper method to get ranked diseases for given phenotypes.
        
        Args:
            phenotypes: List of phenotype IDs (HPO terms)
            use_frequency_weights: Whether to use frequency-weighted scoring
            use_IC_weights: Whether to use Information Content (IC) weights
            
        Returns:
            List of dictionaries with disease info and scores
        """

        if use_samples:
            return self.rank_samples_by_phenotype_associations(phenotypes, use_frequency_weights, use_IC_weights, use_score, exclude_sample_id)
        else:
            if self.config.get('use_kg', False):
                return self.rank_diseases_by_phenotype_associations_from_kg_local(phenotypes, use_frequency_weights, use_IC_weights, use_score, use_samples, exclude_sample_id)
            else:
                return self.rank_diseases_by_phenotype_associations(phenotypes, use_frequency_weights, use_IC_weights, use_score, use_samples, exclude_sample_id)
            
        # if use_frequency_weights:
        #     return self.rank_diseases_by_frequency_weighted_associations(phenotypes)  # TODO: weight tuning pending
        # else:
        #     return self.rank_diseases_by_phenotype_associations(phenotypes, use_IC_weights, use_samples, exclude_sample_id)

    def _get_ranked_diseases_for_phenotypes_embedding(self, phenotypes: List[str], top_k: int = 20, embedding_resort: bool = True, exclude_sample_id: int = None) -> List[Dict]:
        """
        Helper method to get ranked diseases for given phenotypes using embedding-based case extraction.
        
        Args:
            phenotypes: List of phenotype IDs (HPO terms)
            top_k: Number of top diseases to return
            embedding_resort: If True, sort by matching phenotype count first, then by case similarity.
                            If False, sort by case similarity first, then by matching phenotype count.
            
        Returns:
            List of dictionaries with disease info and scores from embedding method
        """
        if not self.external_samples:
            raise ValueError("External samples are required for embedding method. Please provide case_library.")
        
        try:
            top_k_add = top_k + 50
            similar_samples, similar_values = self.find_similar_samples_with_embeddings_v1(phenotypes, top_k_add, exclude_sample_id)
            
            if not similar_samples:
                raise ValueError("No similar samples found using embedding method. Please check your case library file.")
            
            # Collect all diseases from similar cases and rank them by similarity
            all_case_diseases = []
            total_IC = sum(float(self.ic_dict.get(phenotype_id, 0)) for phenotype_id in phenotypes)

            # Get all parent phenotypes (including grandparents, great-grandparents, etc.) for each phenotype
            parents_phenotypes = {}
            for phenotype in phenotypes:
                all_parents = self.get_all_parent_phenotypes(phenotype)
                parents_phenotypes[phenotype] = list(all_parents)
            
            for sample, similarity in zip(similar_samples, similar_values):
                # Get diseases directly from the similar case
                sample_diseases = sample.get('RareDisease', [])
                sample_phenotypes = sample.get('Phenotype', [])
                
                # Treat all diseases in this sample as a single unit
                if not sample_diseases:
                    continue
                
                # Get phenotypes that overlap between this sample and the target phenotypes
                sample_phenotype_set = set(sample_phenotypes)
                target_phenotype_set = set(phenotypes)
                matching_phenotypes = list(sample_phenotype_set.intersection(target_phenotype_set))

                # Check for phenotype matches through parent relationships
                matching_phenotypes_by_parents = []
                matching_phenotypes_by_parents_map = {}
                # Check if any patient phenotype's parents match sample phenotype parents
                for phenotype, parent_list in parents_phenotypes.items():
                    if phenotype in matching_phenotypes:
                        continue  # Already matched directly
                        
                    if parent_list:
                        for parent in parent_list:
                            if parent in sample_phenotypes:
                                matching_phenotypes_by_parents.append(phenotype + " (by parent)")
                                matching_phenotypes_by_parents_map[phenotype] = parent
                                break

                # Remove duplicates from parent matches
                for phenotype in matching_phenotypes_by_parents:
                    if '(' in phenotype:
                        phenotype_id = phenotype.split('(')[0].strip()
                    else:
                        phenotype_id = phenotype.strip()
                        
                    if phenotype_id not in matching_phenotypes:
                        matching_phenotypes.append(phenotype)

                # Check if any phenotype in given phenotypes is a parent phenotype with only one child
                matching_phenotypes_by_child_map = {}
                for phenotype in phenotypes:
                    if (phenotype in matching_phenotypes) or (phenotype + " (by parent)" in matching_phenotypes):
                        continue
                    child_phenotype = self._is_hp_with_only_child(phenotype)
                    if child_phenotype and child_phenotype in sample_phenotypes:
                        matching_phenotypes.append(phenotype + " (by child)")
                        matching_phenotypes_by_child_map[phenotype] = child_phenotype
                
                matching_count = len(matching_phenotypes)
                                
                # Create matching phenotypes with frequency annotations
                matching_phenotypes_with_freq = []
                matching_phenotypes_freq_info = []
                for phenotype_id in matching_phenotypes:
                    
                    phenotype_id_original = phenotype_id
                    if 'by parent' in phenotype_id:
                        phenotype_id_original = phenotype_id.split(' (by parent)')[0].strip()
                        phenotype_id = matching_phenotypes_by_parents_map[phenotype_id_original]
                    elif 'by child' in phenotype_id:
                        phenotype_id_original = phenotype_id.split(' (by child)')[0].strip()
                        phenotype_id = matching_phenotypes_by_child_map[phenotype_id_original]

                    # Check frequency info from all diseases in the sample
                    frequency_displays = []
                    frequency_values = []
                    for disease_id in sample_diseases:
                        freq_info = self.get_frequency_info(phenotype_id, disease_id)
                        if freq_info:
                            # Convert HPO ID frequencies to descriptions
                            if freq_info['frequency_type'] == 'hpo_id':
                                frequency_display = self._convert_hpo_frequency_to_description(freq_info['frequency'])
                            else:
                                frequency_display = freq_info['frequency']
                            frequency_displays.append(frequency_display)
                            
                            if frequency_display:
                                try:
                                    frequency_value = self.get_max_frequency_from_frequency_string(frequency_display)
                                except Exception:
                                    frequency_value = None
                            else:
                                frequency_value = None

                            if frequency_value is not None:
                                frequency_values.append(frequency_value)
                    
                    if frequency_displays:
                        # Remove duplicates and join with semicolon
                        unique_frequencies = list(set(frequency_displays))
                        frequency_text = ';'.join(unique_frequencies)
                        phenotype_with_freq = f"{phenotype_id_original} ({frequency_text})"
                    else:
                        phenotype_with_freq = phenotype_id_original
                    
                    matching_phenotypes_with_freq.append(phenotype_with_freq)
                    
                    if frequency_values:
                        frequency_value_max = max(frequency_values)
                    else:
                        frequency_value_max = 0.0
                    matching_phenotypes_freq_info.append(frequency_value_max)
                
                matching_HPO = []
                for phenotype_id in matching_phenotypes:
                    if '(' in phenotype_id:
                        matching_HPO.append(phenotype_id.split('(')[0].strip())
                    else:
                        matching_HPO.append(phenotype_id.strip())

                matching_IC = sum(float(self.ic_dict.get(phenotype_id, 0)) for phenotype_id in matching_HPO)
                matching_count_weighted = sum(matching_phenotypes_freq_info) if matching_phenotypes_freq_info else 0.0
                
                # Create disease names for all diseases in the sample
                # disease_names = []
                # for disease_id in sample_diseases:
                #     disease_names.append(self.get_disease_name(disease_id))
                # disease_name = '; '.join(disease_names)
                disease_name = "; ".join(self.get_disease_all_names(sample_diseases[0]))

                matching_IC = matching_IC/total_IC
                matching_count_weighted = matching_count_weighted/(len(phenotypes)*1.0)
                score = matching_count_weighted + matching_IC + similarity

                # Create sample info (treating all diseases as a single unit)
                sample_info = {
                    'disease_id': sample_diseases,  # All disease IDs as a list
                    'disease_name': disease_name,   # All disease names joined with semicolon
                    'total_phenotype_associations': len(sample_phenotypes),
                    'matching_phenotypes': matching_phenotypes_with_freq,
                    'matching_phenotype_count': matching_count,
                    'matching_phenotype_count_weighted': matching_count_weighted,
                    'matching_phenotype_IC': matching_IC,
                    'case_similarity': similarity,
                    'score': score,
                    'case_id': sample.get('Department'),
                    'method': 'embedding'
                }
                all_case_diseases.append(sample_info)
            
            # re-sort by matching phenotype count (descending) and then by case similarity (descending)
            if embedding_resort:
                all_case_diseases.sort(key=lambda x: (x['matching_phenotype_count'], x['case_similarity']), reverse=True)
            else:
                all_case_diseases.sort(key=lambda x: (x['case_similarity'], x['matching_phenotype_count']), reverse=True)

            # No deduplication for real samples - keep all samples as they are
            
            return all_case_diseases
                    
        except Exception as e:
            print(f"Error extracting similar cases: {e}")
            import traceback
            traceback.print_exc()
            raise ValueError(f"Failed to extract similar cases using embedding method: {e}")

    def generate_phenotype_to_disease_prompt(self, phenotypes: List[str], diseases: List[str], top_k: int = 20, 
                                           case_extraction_method: str = "overlap", embedding_resort: bool = True,
                                           use_frequency_weights: bool = False, use_IC_weights: bool = False, use_score: bool = False,
                                           use_samples: bool = False, prompt_steps: int = 3, exclude_sample_id: int = None) -> Optional[Dict]:
        """Generate phenotype to disease inference prompt from input phenotypes and diseases"""
        
        # Filter to only OMIM and ORPHA diseases (skip CCRD)
        valid_diseases = [d for d in diseases if d.startswith(('OMIM:', 'ORPHA:'))]
        if not valid_diseases:
            return None
        
        # Get all disease names including synonyms for each disease
        true_disease_names_with_synonyms = []
        for disease_id in valid_diseases:
            # Get all names (including synonyms) for this disease
            disease_names = self.get_disease_all_names(disease_id)
            true_disease_names_with_synonyms.extend(disease_names)
        
        # Remove duplicates while preserving order
        true_disease_names_with_synonyms = list(dict.fromkeys(true_disease_names_with_synonyms))
        
        # Use all valid diseases instead of just the first one
        all_true_diseases = valid_diseases
        all_true_disease_names = true_disease_names_with_synonyms
        
        # Initialize ranked_diseases_text
        ranked_diseases_text = ""
        
        # Generate ranked diseases based on case extraction method
        # Validate case extraction method
        valid_methods = ["overlap", "embedding", "both"]
        if case_extraction_method not in valid_methods:
            raise ValueError(f"Invalid case_extraction_method: '{case_extraction_method}'. Valid options are: {valid_methods}")
        
        if case_extraction_method == "both":
            # Both methods: combine results from overlap and embedding methods
            # print("Using both overlap and embedding methods for case extraction...")
            
            # Get top_k diseases from overlap method
            overlap_ranked_diseases = self._get_ranked_diseases_for_phenotypes(phenotypes, use_frequency_weights, use_IC_weights, use_score, use_samples, exclude_sample_id)
            overlap_top_k = overlap_ranked_diseases[:top_k]
            
            # Get top_k diseases from embedding method
            try:
                embedding_ranked_diseases = self._get_ranked_diseases_for_phenotypes_embedding(phenotypes, top_k, embedding_resort, exclude_sample_id)
                embedding_top_k = embedding_ranked_diseases[:top_k]
            except Exception as e:
                print(f"Warning: Error in embedding method: {e}")
                embedding_top_k = []
            
            # Add method identifier to overlap diseases
            for disease_info in overlap_top_k:
                disease_info['method'] = 'overlap'
            
            # Merge top_k from both methods and remove duplicates
            all_ranked_diseases = []
            seen_diseases = set()

            if not use_samples:
                # First add overlap top_k diseases (they have higher priority for ranking)
                for disease_info in overlap_top_k:
                    key = self._disease_key(disease_info['disease_id'])
                    if key not in seen_diseases:
                        seen_diseases.add(key)
                        all_ranked_diseases.append(disease_info)
                
                # Then add embedding top_k diseases that weren't already included
                for disease_info in embedding_top_k:
                    key = self._disease_key(disease_info['disease_id'])
                    if key not in seen_diseases:
                        seen_diseases.add(key)
                        all_ranked_diseases.append(disease_info)
            
            # dedup by case_id
            else:
                seen_case_ids = set()
                for disease_info in overlap_top_k:
                    all_ranked_diseases.append(disease_info)
                    seen_case_ids.add(disease_info['case_id'])
                for disease_info in embedding_top_k:
                    if disease_info['case_id'] not in seen_case_ids:
                        all_ranked_diseases.append(disease_info)
                        seen_case_ids.add(disease_info['case_id'])

            if use_IC_weights:
                all_ranked_diseases.sort(key=lambda x: (x['matching_phenotype_IC'], x['matching_phenotype_count']), reverse=True)
            else:
                all_ranked_diseases.sort(key=lambda x: (x['matching_phenotype_count'], x['matching_phenotype_IC']), reverse=True)
            
            complete_ranked_diseases = all_ranked_diseases
            # ranked_diseases = all_ranked_diseases  # Use all merged diseases as final list
        elif case_extraction_method == "embedding":
            # Embedding method: needs external samples for case extraction
            complete_ranked_diseases = self._get_ranked_diseases_for_phenotypes_embedding(phenotypes, top_k, embedding_resort, exclude_sample_id)
        else:  # case_extraction_method == "overlap"
            # Overlap method: direct phenotype-based ranking
            complete_ranked_diseases = self._get_ranked_diseases_for_phenotypes(phenotypes, use_frequency_weights, use_IC_weights, use_score, use_samples, exclude_sample_id)
            
            # Add method identifier to overlap diseases
            for disease_info in complete_ranked_diseases:
                disease_info['method'] = 'overlap'
            
            # ranked_diseases = complete_ranked_diseases[:top_k]
                
        # Find which input phenotypes are known to be associated with ANY of the true diseases
        all_true_disease_associated_phenotypes = []
        for hpo_id in phenotypes:
            # Get diseases associated with this phenotype
            phenotype_diseases = self.phenotype_to_diseases.get(hpo_id, set())
            # Check if any true disease is in the disease list for this phenotype
            for true_disease_id in all_true_diseases:
                if true_disease_id in phenotype_diseases:
                    all_true_disease_associated_phenotypes.append(hpo_id)
                    break  # Found one match, no need to check other true diseases for this phenotype
        
        # Optimized semantic similarity calculation
        similarity_threshold = self.config.get('evaluation_config', {}).get('similarity_threshold', 0.8)
                
        # Use optimized advanced disease matching method with batch processing
        # Find overall best match based on method using optimized matching logic
        target_disease_rank = None
        best_matched_disease = None
        best_similarity = 0
        
        if case_extraction_method == "both":
            # For 'both' method, search across all merged diseases
            ranked_diseases = complete_ranked_diseases
        else:
            # For other methods, search only in top-k diseases
            ranked_diseases = complete_ranked_diseases[:top_k]
        
        # Get ranked disease names for matching
        ranked_disease_names = [d['disease_name'] for d in ranked_diseases]
        # Choose matching method based on data size for performance optimization
        total_diseases = len(all_true_disease_names) + len(ranked_disease_names)
        
        # Use optimized matching method for larger datasets
        match_result = self.find_best_match_rank_optimized(all_true_disease_names, ranked_disease_names)
        
        if match_result["best_similarity"] > 0:
            best_similarity = match_result["best_similarity"]
            best_matched_disease = match_result["best_match"]
            target_disease_rank = match_result["best_rank"]
        
        # Only consider it a match if similarity >= threshold
        if best_similarity < similarity_threshold:
            target_disease_rank = None
            best_matched_disease = None
        
        # Get phenotype names and build phenotype-disease mappings
        phenotype_names = []
        phenotype_disease_mappings = {}
        any_true_disease_in_phenotype_lists = False
        
        for hpo_id in phenotypes:
            phenotype_name = self.get_phenotype_name(hpo_id)
            if phenotype_name:
                full_phenotype_name = f"{phenotype_name} ({hpo_id})"
                phenotype_names.append(full_phenotype_name)
                
                # Get diseases associated with this phenotype from phenotype.hpoa
                phenotype_diseases = self.get_phenotype_diseases(hpo_id)
                if phenotype_diseases:
                    phenotype_disease_mappings[full_phenotype_name] = phenotype_diseases
                    # Check if any true disease is in this phenotype's disease list
                    for true_disease_name in all_true_disease_names:
                        if true_disease_name in phenotype_diseases:
                            any_true_disease_in_phenotype_lists = True
                            break
            else:
                phenotype_names.append(f"Unknown Phenotype ({hpo_id})")
        
        if not phenotype_names:
            return None
        
        # Filter out samples with only one phenotype
        # if len(phenotype_names) <= 1:
        #     return None
        
        # # Build phenotype-disease mapping text
        # phenotype_disease_text = ""
        # for phenotype_name, diseases in phenotype_disease_mappings.items():
        #     disease_list = "; ".join(diseases)
        #     phenotype_disease_text += f"\n- {phenotype_name}: {disease_list}"
        
        # if not phenotype_disease_text:
        #     phenotype_disease_text = "\n- No specific disease associations found for the phenotypes"
        
        # Convert phenotypes to natural language
        pheno_text = ", ".join(phenotype_names) if phenotype_names else "no specific phenotypes"
        
        # Sort phenotypes by the number of associated diseases (ascending order)
        # This helps prioritize more specific phenotypes that are associated with fewer diseases
        phenotype_disease_counts = []
        for hpo_id in phenotypes:
            disease_count = len(self.phenotype_to_diseases.get(hpo_id, set()))
            phenotype_disease_counts.append((hpo_id, disease_count))
        
        # Sort by disease count (ascending: fewer diseases = more specific phenotype)
        phenotype_disease_counts.sort(key=lambda x: x[1])
        
        # Extract sorted phenotype IDs
        sorted_phenotypes = [hpo_id for hpo_id, _ in phenotype_disease_counts]
        
        # Update phenotypes list to use sorted order
        phenotypes = sorted_phenotypes
        
        # Debug output: show sorted phenotype list with disease counts
        # print("Phenotype sorting results:")
        # for hpo_id, disease_count in phenotype_disease_counts:
        #     phenotype_name = self.get_phenotype_name(hpo_id) or hpo_id
        #     print(f"  {phenotype_name} ({hpo_id}): {disease_count} diseases")
        
        # Build detailed phenotype information in natural language
        phenotype_details_list = []
        pheno_num = 0
        for hpo_id in phenotypes:
            pheno_num += 1
            phenotype_name = self.get_phenotype_name(hpo_id)
            synonyms = self.get_phenotype_synonyms(hpo_id)
            definition = self.get_phenotype_definition(hpo_id)
            is_a_names = self.get_phenotype_is_a_names(hpo_id)
            
            # Get disease count for this phenotype
            disease_count = len(self.phenotype_to_diseases.get(hpo_id, set()))
            
            # Build natural language description with is_a information
            # "Reported in {disease_count} diseases" in prompt would bias model toward max disease_count phenotype; not intended
            # TODO: add which Phenotypic abnormality (HP:0000118) subcategory each phenotype belongs to
            # get phenotype's Phenotypic abnormality category
            phenotype_abnormal_category_ids = self.get_phenotype_abnormal_category(hpo_id)
            if phenotype_abnormal_category_ids:
                phenotype_abnormal_categories = []
                for category_id in phenotype_abnormal_category_ids:
                    category_name = self.get_phenotype_name(category_id)
                    if category_name:
                        phenotype_abnormal_categories.append(f"{category_id} ({category_name})")
                    else:
                        phenotype_abnormal_categories.append(category_id)
                phenotype_abnormal_category = " | ".join(phenotype_abnormal_categories)
            else:
                phenotype_abnormal_category = "Unknown category"

            if is_a_names:
                is_a_text = f"belongs to {', '.join(is_a_names)}"
                if synonyms and definition:
                    detailed_info = f"{pheno_num}. **{phenotype_name} ({hpo_id})**: {phenotype_abnormal_category}, {is_a_text}, also known as {', '.join(synonyms)}. {definition}"
                elif synonyms:
                    detailed_info = f"{pheno_num}. **{phenotype_name} ({hpo_id})**: {phenotype_abnormal_category}, {is_a_text}, also known as {', '.join(synonyms)}."
                elif definition:
                    detailed_info = f"{pheno_num}. **{phenotype_name} ({hpo_id})**: {phenotype_abnormal_category}, {is_a_text}, {definition}"
                else:
                    detailed_info = f"{pheno_num}. **{phenotype_name} ({hpo_id})**: {phenotype_abnormal_category}, {is_a_text}."
            else:
                # No is_a information available
                if synonyms and definition:
                    detailed_info = f"{pheno_num}. **{phenotype_name} ({hpo_id})**: {phenotype_abnormal_category}, also known as {', '.join(synonyms)}. {definition}"
                elif synonyms:
                    detailed_info = f"{pheno_num}. **{phenotype_name} ({hpo_id})**: {phenotype_abnormal_category}, also known as {', '.join(synonyms)}."
                elif definition:
                    detailed_info = f"{pheno_num}. **{phenotype_name} ({hpo_id})**: {phenotype_abnormal_category}, {definition}"
                else:
                    detailed_info = f"{pheno_num}. **{phenotype_name} ({hpo_id})**: {phenotype_abnormal_category}, no additional information available."
            
            phenotype_details_list.append(detailed_info)
        
        pheno_details_text = "\n".join(phenotype_details_list) if phenotype_details_list else "No detailed information available"

        # Build ranked diseases information for the prompt with frequency annotations
        ranked_diseases_text = ""
        if ranked_diseases:
            # For 'both' method, use all merged diseases; for other methods, use top_k
            diseases_to_include = ranked_diseases
            
            # when using real cases, no need to dedup
            if not use_samples:
                # Track seen disease names to avoid duplicates (same disease name with different IDs)
                seen_disease_names = set()
                unique_diseases = []
            
                # Filter out duplicate disease names while preserving order
                # This prevents the same disease from appearing multiple times in the prompt
                for disease_info in diseases_to_include:
                    disease_name = disease_info['disease_name']
                    if disease_name not in seen_disease_names:
                        seen_disease_names.add(disease_name)
                        unique_diseases.append(disease_info)
                
                # Update ranked_diseases to use unique diseases for consistency
                ranked_diseases = unique_diseases
                ranked_disease_names = [d['disease_name'] for d in ranked_diseases]
            
            # get parents phenotypes for each phenotype in phenotypes
            parents_phenotypes = {}
            for phenotype in phenotypes:
                all_parents = self.get_all_parent_phenotypes(phenotype)
                parents_phenotypes[phenotype] = list(all_parents)

            # Generate ranked diseases text with unique diseases
            for i, disease_info in enumerate(ranked_diseases, 1):
                disease_name = disease_info['disease_name']
                disease_id = disease_info['disease_id']
                # disease_description = self.get_disease_description(disease_id)
                matching_count = disease_info['matching_phenotype_count']
                matching_phenotypes = disease_info['matching_phenotypes']

                matching_phenotypes_without_frequency = []
                
                # Extract phenotype IDs without frequency information from matching_phenotypes
                for phenotype_with_freq in matching_phenotypes:
                    # Extract HPO ID from strings like "HP:0001234 (frequent)" or "HP:0001234"
                    if '(' in phenotype_with_freq and ')' in phenotype_with_freq:
                        # Extract HPO ID before the frequency information
                        hpo_id = phenotype_with_freq.split(' (')[0]
                        matching_phenotypes_without_frequency.append(hpo_id)
                    else:
                        # If no frequency info, use as is
                        matching_phenotypes_without_frequency.append(phenotype_with_freq)
                
                if prompt_steps == 2:
                    renum_id = 10 + i  # for step2 prompt ro renumber the id of cases.
                else:
                    renum_id = i  # for step3 prompt

                if matching_phenotypes:

                    # case 1: matching_phenotypes already contains frequency information for embedding method
                    # case 1.1: prompt without frequency information
                    phenotype_text = ", ".join(matching_phenotypes_without_frequency)
                    
                    # case 1.2: prompt with frequency information
                    # phenotype_text = "; ".join(matching_phenotypes)

                    # ranked_diseases_text += f"{renum_id}. **{disease_name}** - {phenotype_text}.\n"

                    # case 2: without phenotype information
                    # ranked_diseases_text += f"{renum_id}. **{disease_name}**.\n"
                    
                    # ranked_diseases_text += f"Patient {renum_id} has **{disease_name}** with phenotypes including but not limited to {phenotype_text}\n"
                    ranked_diseases_text += f"Patient {renum_id} has **{disease_name}**.\n" 
                else:
                    ranked_diseases_text += f"Patient {renum_id} has **{disease_name}**.\n"

        ranked_diseases_json = {}
        start_id = 1
        for i, d in enumerate(ranked_diseases):
            renum_id = i+start_id
            # disease_name = d.get('disease_name', '')
            disease_id = d.get('disease_id', '')
            disease_name = set(self.get_disease_name(_id) for _id in disease_id if isinstance(disease_id, list))
            disease_name = "; ".join(disease_name)
            synonyms = self.get_disease_synonyms(disease_id[0] if isinstance(disease_id, list) else disease_id)
            disease_type = self.disease_types.get(disease_id[0] if isinstance(disease_id, list) else disease_id, "")
            disease_description = self.get_disease_description(disease_id[0] if isinstance(disease_id, list) else disease_id)
            if disease_description == "":
                disease_description = "[Information is missing; please infer based on your memory.]"
            
            # get all phenotypes for disease including freq
            phenotypes = self.disease_to_phenotypes.get(disease_id[0] if isinstance(disease_id, list) else disease_id, [])
            
            # first collect all phenotypes with freq
            phenotype_list = []
            for phenotype in phenotypes:
                freq_key = (phenotype, disease_id[0]) if isinstance(disease_id, list) else (phenotype, disease_id)
                freq_info = self.phenotype_disease_frequency[freq_key]
                frequency_string = freq_info.get('frequency', '')
                # use get_max_frequency_from_frequency_string to number
                if freq_info['frequency_type'] == 'hpo_id':
                    frequency_string = self._convert_hpo_frequency_to_description(frequency_string)
                else:
                    frequency_string = frequency_string
                frequency_numeric = self.get_max_frequency_from_frequency_string(frequency_string)
                
                if frequency_numeric <= 0.17 and frequency_numeric > 0:
                    continue

                frequency_numeric = round(frequency_numeric, 3)
                phenotype_name = self.get_phenotype_name(phenotype)
                phenotype_list.append((phenotype_name, frequency_numeric))
            # sort by frequency_numeric descending
            phenotype_list.sort(key=lambda x: x[1], reverse=True)
            
            # if len(phenotype_list) > 30:
            #     phenotype_list = phenotype_list[:30]
            # concatenate phenotype text
            phenotypes_text = ""
            for phenotype_name, frequency_numeric in phenotype_list:
                if frequency_numeric > 0:
                    phenotypes_text += f"{phenotype_name}({frequency_numeric});"
                else:
                    phenotypes_text += f"{phenotype_name};"

            matching_phenotypes = d.get('matching_phenotypes', [])
            ranked_diseases_json[f"Case {renum_id}"] = {
                "Disease name": disease_name,
                # "Disease id": disease_id,
                # "Synonyms": synonyms,
                "Disease category": disease_type,
                # "Disease phenotypes (with frequency)": phenotypes_text,
                "Disease description": disease_description,
                # "Matching phenotypes": matching_phenotypes,
            }
        ranked_diseases_text = json.dumps(ranked_diseases_json, ensure_ascii=False, indent=2)
        # remove outermost braces
        ranked_diseases_text = ranked_diseases_text[1:-1]

# # ---------------------------------------------- v5.2 --------------------------------------------------------
# # Two-step reasoning prompt
        if prompt_steps == 2:
            step1_output = "<STEP1_OUTPUT>"
            step1_prompt = f"""
You are an expert in rare diseases.

A patient presents with the following detailed phenotypes:
{pheno_details_text}

Please follow the structured reasoning approach below to identify exactly 10 potential candidate rare diseases **based solely on the provided phenotypes**.

**REASONING APPROACH (Strict):**

1. Use ONLY the phenotypes provided.
2. Consider clusters of phenotypes by organ/system to detect disease patterns.
3. The patient may have multiple underlying conditions; not all phenotypes need to fit one disease.
4. Treat all phenotypes as potentially relevant; do not over-prioritize any single feature.
5. Do not exclude a disease if some phenotypes are missing.
6. Consider inferred patient factors (age, sex, onset, progression) from phenotypes.
7. Prioritize rare and well-documented diseases with high phenotype specificity.

**FINAL ANSWER FORMAT (strict):**  
List the 10 most likely candidate diseases in order, with no explanations, and the output must start with "FINAL ANSWER":  

FINAL ANSWER:
1. DISEASE_NAME
2. DISEASE_NAME
3. DISEASE_NAME
4. DISEASE_NAME
5. DISEASE_NAME
6. DISEASE_NAME
7. DISEASE_NAME
8. DISEASE_NAME
9. DISEASE_NAME
10. DISEASE_NAME
"""
            step2_output = "<STEP2_OUTPUT>"
            step2_prompt = f"""
You are an expert in rare diseases.

A patient presents with the following detailed phenotypes:
{pheno_details_text}

The following diseases are considered highly relevant candidates (**unordered list; no ranking implied**), list with the associated phenotypes: 

{ranked_diseases_text}{step1_output}

Please follow the structured reasoning approach below to identify the most likely diseases for this patient.  

---

**IMPORTANT ADDITIONAL INSTRUCTION:**  
For each candidate disease, you must not only rely on the explicitly listed phenotype associations above,  
but also incorporate your own medical and genetic knowledge base to identify additional potential phenotype-disease associations.  
This ensures that diseases are not unfairly excluded due to incomplete associations in the provided list.  
If your knowledge base indicates possible links between the candidate disease and the patient's phenotypes, you should include that in your reasoning.  

**REASONING APPROACH (Strict):**

1. **List all provided phenotypes.**
2. **Cluster phenotypes by organ/system**.  
3. **Identify key phenotype patterns/signatures** that indicate likely disease classes.  
4. **Cross-match phenotype clusters with the provided candidate diseases.**  
   - The patient may have multiple diseases; not all phenotypes must fit one disease. 
   - Do not over-prioritize any single phenotype; consider all phenotypes as relevant.
   - Do not exclude a disease solely because some provided phenotypes are missing, as not all phenotypes must appear in a single disease.
   - Diseases with higher overlap of phenotype patterns should be ranked higher.  
5. **Assess progression, age, and sex implications** inferred from the phenotypes.  
6. **Select at least 10 candidate diseases** that best fit the phenotype distribution.  

---

**FINAL ANSWER FORMAT (strict):**  
List the 10 most likely candidate diseases in order, with no explanations, and the output must start with "FINAL ANSWER":  

FINAL ANSWER:
1. DISEASE_NAME
2. DISEASE_NAME
3. DISEASE_NAME
4. DISEASE_NAME
5. DISEASE_NAME
6. DISEASE_NAME
7. DISEASE_NAME
8. DISEASE_NAME
9. DISEASE_NAME
10. DISEASE_NAME
"""
            # For 2-step, step3_prompt is None
            step3_prompt = None

# ---------------------------- v6 3steps in 3 inferences with detailed phenotype information with optimal prompt design--------------------------------
        
        elif prompt_steps == 3:
            step1_output = "<STEP1_OUTPUT>"
            step1_prompt = f"""
You are an expert in rare diseases.

A patient presents with the following detailed phenotypes:
{pheno_details_text}

**Task:** Identify exactly 10 potential candidate rare diseases **based solely on the phenotypes**.

**Rules for reasoning:**
1. Use ONLY the phenotypes provided.
2. Consider clusters of phenotypes by organ/system to detect disease patterns.
3. The patient may have multiple underlying conditions; not all phenotypes need to fit one disease.
4. Treat all phenotypes as potentially relevant; do not over-prioritize any single feature.
5. Do not exclude a disease if some phenotypes are missing.
6. Consider inferred patient factors (age, sex, onset, progression) from phenotypes.
7. Prioritize rare and well-documented diseases with high phenotype specificity.

**Output:** List exactly 10 candidate rare diseases in order of relevance, no explanations, and the output must start with "FINAL ANSWER:".

FINAL ANSWER:
1. DISEASE_NAME
2. DISEASE_NAME
3. DISEASE_NAME
4. DISEASE_NAME
5. DISEASE_NAME
6. DISEASE_NAME
7. DISEASE_NAME
8. DISEASE_NAME
9. DISEASE_NAME
10. DISEASE_NAME
"""
            step2_output = "<STEP2_OUTPUT>"
            step2_prompt = f"""
You are an expert in rare diseases.

A patient presents with the following detailed phenotypes:
{pheno_details_text}

The following diseases are considered highly relevant candidates (**unordered list; no ranking implied**), list with the associated phenotypes: 
{ranked_diseases_text}{step1_output}

Please follow the structured reasoning approach below to identify the most likely diseases for this patient.  

---

REASONING APPROACH (Strict):

1. Cluster phenotypes by organ/system
   Group all phenotypes into clinical systems (e.g., neurological, craniofacial, ophthalmologic, auditory, cardiovascular, respiratory, gastrointestinal, genitourinary, musculoskeletal, dermatologic, endocrine/metabolic). This helps reveal system-level patterns.

2. Identify cross-system phenotype patterns/signatures
   Find key combinations across systems that suggest specific disease classes or syndromes (e.g., skin abnormalities + immune deficiency; craniofacial anomalies + limb defects). These cross-system signatures should guide candidate selection.

3. Cross-match phenotype clusters with the provided candidate diseases
   - For each candidate disease, compare the phenotype clusters against the disease's listed features AND your own medical/genetic knowledge.
   - Include known, typical features of the disease that may not be listed in the prompt.
   - Do NOT require every phenotype to be explained by a single disease — multiple diseases may coexist.
   - Do NOT over-weight any single phenotype; evaluate the overall phenotype architecture.
   - Metabolic findings are supportive only: do NOT use metabolic phenotypes as the primary screening criterion.

4. Select at least 20 candidate diseases with the strongest overall fit
   - From the provided candidate list, choose ≥20 diseases that best match the main phenotype patterns.
   - For each selected disease, give a brief justification summarizing matched phenotypes.
   - The goal is to identify the most likely diseases even if no single disease perfectly fits all phenotypes.

---

**FINAL ANSWER FORMAT (strict):**  
List the 20 most likely candidate diseases in order, with no explanations, and the output must start with "FINAL ANSWER":  

FINAL ANSWER:
1. DISEASE_NAME
2. DISEASE_NAME
...
20. DISEASE_NAME
"""

            step3_prompt = f"""
You are an expert in rare diseases.

A patient presents with the following detailed phenotypes:
{pheno_details_text}

The following diseases are considered highly relevant candidates, list with the associated phenotypes: 
{step2_output}

Please follow the structured reasoning approach below to identify the most likely diseases for this patient.  

---

REASONING APPROACH (Strict):

1. Cluster phenotypes by organ/system
   Group all phenotypes into clinical systems (e.g., neurological, craniofacial, ophthalmologic, auditory, cardiovascular, respiratory, gastrointestinal, genitourinary, musculoskeletal, dermatologic, endocrine/metabolic). This helps reveal system-level patterns.

2. Identify cross-system phenotype patterns/signatures
   Find key combinations across systems that suggest specific disease classes or syndromes (e.g., skin abnormalities + immune deficiency; craniofacial anomalies + limb defects). These cross-system signatures should guide candidate selection.

3. Cross-match phenotype clusters with the provided candidate diseases
   - For each candidate disease, compare the phenotype clusters against the disease's listed features AND your own medical/genetic knowledge.
   - Include known, typical features of the disease that may not be listed in the prompt.
   - Do NOT require every phenotype to be explained by a single disease — multiple diseases may coexist.
   - Do NOT over-weight any single phenotype; evaluate the overall phenotype architecture.
   - Metabolic findings are supportive only: do NOT use metabolic phenotypes as the primary screening criterion.

4. Select at least 10 candidate diseases with the strongest overall fit
   - From the provided candidate list, choose ≥10 diseases that best match the main phenotype patterns.
   - For each selected disease, give a brief justification summarizing matched phenotypes.
   - The goal is to identify the most likely diseases even if no single disease perfectly fits all phenotypes.

---

**FINAL ANSWER FORMAT (strict):**  
List the 10 most likely candidate diseases in order, with no explanations, and the output must start with "FINAL ANSWER":  

FINAL ANSWER:
1. DISEASE_NAME
2. DISEASE_NAME
3. DISEASE_NAME
4. DISEASE_NAME
5. DISEASE_NAME
6. DISEASE_NAME
7. DISEASE_NAME
8. DISEASE_NAME
9. DISEASE_NAME
10. DISEASE_NAME
"""            

        # Generate answer format - use actual disease name
        if best_matched_disease:
            answer = f"{best_matched_disease}"
        else:
            # Use the first true disease if no match found
            answer = f"{all_true_disease_names[0] if all_true_disease_names else 'Unknown Disease'}"
        
        # Calculate prompt lengths based on steps
        step1_prompt_length = len(step1_prompt) if step1_prompt else 0
        step2_prompt_length = len(step2_prompt) if step2_prompt else 0
        step3_prompt_length = len(step3_prompt) if step3_prompt else 0
        
        return {
            "task_type": "phenotype_to_disease",
            "prompt_steps": prompt_steps,
            "step1_prompt": step1_prompt or "",
            "step2_prompt": step2_prompt or "",
            "step3_prompt": step3_prompt or "",
            "step1_prompt_length": step1_prompt_length,
            "step2_prompt_length": step2_prompt_length,
            "step3_prompt_length": step3_prompt_length,
            "answer": answer,
            "answer_length": len(answer),
            "phenotypes": phenotype_names,
            "pheno_details_text": pheno_details_text,
            "true_diseases": all_true_disease_names,  # Use all true diseases
            "true_disease_ids": all_true_diseases,  # Use all true diseases
            "target_disease_in_phenotype_lists": any_true_disease_in_phenotype_lists,
            "target_disease_associated_phenotypes": all_true_disease_associated_phenotypes,
            # Add ranking information
            "ranked_diseases": ranked_diseases,
            "ranked_disease_names": ranked_disease_names,
            "target_disease_rank": target_disease_rank,
            "target_disease_matching_count": best_similarity if best_similarity else 0.0, # Use best_similarity as count
            "target_disease_similarity": best_similarity if best_similarity else 0.0,
            "best_matched_disease": best_matched_disease,
            "top_k": top_k,
            # Add case extraction information
            "case_extraction_method": case_extraction_method,
            # Add advanced matching information
            "matching_method": "optimized_advanced_with_splitting",
            "similarity_threshold": similarity_threshold,
            "total_diseases_processed": total_diseases if 'total_diseases' in locals() else len(all_true_disease_names) + len(ranked_disease_names)
        }
    
    def load_input_data(self, input_file: str) -> List[List]:
        """Load phenotype-disease data from input JSON file"""
        print(f"Loading data from {input_file}...")
        
        try:
            with open(input_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            print(f"Loaded {len(data)} records from input file")
            return data
            
        except Exception as e:
            print(f"Error loading input data: {e}")
            return []
    
    def generate_dataset(self, input_file: str, num_samples: int = None, top_k: int = 20, 
                        use_frequency_weights: bool = False, case_extraction_method: str = "overlap", 
                        prompt_steps: int = 3, save_case_library_only: bool = False, use_IC_weights: bool = False, use_score: bool = False, use_samples: bool = False,
                        embedding_resort: bool = True, sample_indices: List[int] = None) -> List[Dict]:
        """Generate a dataset of phenotype-to-disease prompts from input data"""
        
        # Load input data
        input_data = self.load_input_data(input_file)
        if not input_data:
            print("No input data loaded, exiting...")
            return []
        
        print(f"Processing {len(input_data)} input records")
        print(f"Prompt steps: {prompt_steps}")
        print(f"Case extraction method: {case_extraction_method}")
        print(f"Top K: {top_k}")
        if case_extraction_method in ["embedding", "both"]:
            print(f"Embedding resort: {embedding_resort}")

        # if use_samples:
        #     print("Loading trueSamples data...")
        #     self._load_trueSamples()

        # If only saving case library, extract phenotype-disease mappings from the loaded data
        if save_case_library_only:
            print("Saving phenotype-disease case library from loaded mappings...")
            case_library = []
            
            # # Get all phenotype-disease mappings from the loaded data
            # for disease_id, phenotype_ids in self.disease_to_phenotypes.items():
            #     if phenotype_ids:  # Only include phenotypes that have associated diseases
            #         case_entry = {
            #             "Phenotype": list(phenotype_ids),
            #             "RareDisease": [disease_id],
            #             "Department": None
            #         }
            #         case_library.append(case_entry)
            
            for _, disease_ids in self.disease_name_to_ids.items():
                for disease_id in disease_ids:
                    phenotype_ids = self.disease_to_phenotypes[disease_id]
                    if not phenotype_ids:
                        continue
                    case_entry = {
                        "Phenotype": list(phenotype_ids),
                        "RareDisease": list(disease_ids),
                        "Department": None
                    }
                    case_library.append(case_entry)
                    break
                
            # define output dir, following graph_output pattern
            # if general_cases_output_dir in config, use it; else build from base_path
            if self.config.get("general_cases_output_dir"):
                general_cases_dir = self.config.get("general_cases_output_dir")
            elif self.config.get("base_path"):
                general_cases_dir = os.path.join(self.config.get("base_path"), "general_cases")
            else:
                general_cases_dir = "general_cases"
            os.makedirs(general_cases_dir, exist_ok=True)
            
            # Save to JSONL file
            output_file = os.path.join(general_cases_dir, "phenotype_disease_case_library.jsonl") if general_cases_dir else "phenotype_disease_case_library.jsonl"
            # Save original case library
            with open(output_file, 'w', encoding='utf-8') as f:
                for case in case_library:
                    f.write(json.dumps(case, ensure_ascii=False) + '\n')
            
            # Create high-frequency phenotype case library
            high_freq_case_library = []
            high_freq_count = 0
            freq_count = 0
            
            for case in case_library:
                disease_id = case["RareDisease"][0]
                phenotype_ids = case["Phenotype"]
                high_freq_phenotypes = []
                
                for phenotype_id in phenotype_ids:
                    # Get frequency information for this phenotype-disease pair
                    freq_info = self.get_frequency_info(phenotype_id, disease_id)
                    if freq_info:
                        # Convert frequency to numeric value
                        if freq_info['frequency_type'] == 'hpo_id':
                            frequency_display = self._convert_hpo_frequency_to_description(freq_info['frequency'])
                        else:
                            frequency_display = freq_info['frequency']
                        
                        if frequency_display:
                            try:
                                # Convert frequency to numeric value using get_max_frequency_from_frequency_string
                                frequency_value = self.get_max_frequency_from_frequency_string(frequency_display)
                                # Only include phenotypes with frequency >= 0.17 (17%)
                                if frequency_value >= 0.17:
                                    high_freq_phenotypes.append(phenotype_id)
                            except:
                                # If frequency conversion fails, skip this phenotype
                                continue
                
                # Only include cases that have high-frequency phenotypes
                if high_freq_phenotypes:
                    high_freq_case = {
                        "Phenotype": high_freq_phenotypes,
                        "RareDisease": case["RareDisease"],
                        "Department": None
                    }
                    high_freq_case_library.append(high_freq_case)
                    high_freq_count += len(high_freq_phenotypes)
            
            # Save high-frequency case library
            output_file_with_high_freq = os.path.join(general_cases_dir, "phenotype_disease_case_library_with_high_freq.jsonl") if general_cases_dir else "phenotype_disease_case_library_with_high_freq.jsonl"
            with open(output_file_with_high_freq, 'w', encoding='utf-8') as f:
                for case in high_freq_case_library:
                    f.write(json.dumps(case, ensure_ascii=False) + '\n')
                    
            # write json DB: standard_name - disease_id(s) - alias(es) - phenotypes (each with freq in parens)
            case_database_entries = self._build_case_database_entries()
            
            case_database_output = os.path.join(general_cases_dir, self.config.get("case_database_output_file", "phenotype_disease_case_database.json")) if general_cases_dir else self.config.get("case_database_output_file", "phenotype_disease_case_database.json")
            with open(case_database_output, 'w', encoding='utf-8') as f:
                json.dump(case_database_entries, f, ensure_ascii=False, indent=2)

            print(f"Saved {len(case_database_entries)} disease entries to {case_database_output}")
            
            
            # output json: disease_id (key) -> {standard_name, synonyms}
            # standard_names = {disease_id: (self.disease_names.get(disease_id, "") or "").strip() for disease_id in sorted_ids}
            # standard_names is dict; extract id and standard_name, id as key

            disease_ids_names = {}

            # from case_database_entries build {id: {standard_name, synonyms}}
            for entry in case_database_entries.values():
                standard_names = entry.get("standard_names", {})
                aliases = entry.get("aliases", []) or []
                for disease_id, std_name in standard_names.items():
                    disease_ids_names[disease_id] = {
                        "standard_name": (std_name or "").strip(),
                        "synonyms": aliases
                    }

            disease_ids_names_output = os.path.join(general_cases_dir, (self.config or {}).get("disease_ids_names_output_file", "disease_ids_names.json")) if general_cases_dir else (self.config or {}).get("disease_ids_names_output_file", "disease_ids_names.json")
            with open(disease_ids_names_output, 'w', encoding='utf-8') as f:
                json.dump(disease_ids_names, f, ensure_ascii=False, indent=2)

            print(f"Saved {len(disease_ids_names)} disease id-name mappings to {disease_ids_names_output}")
            
            # export graph: disease_nodes.csv (ID=aggregated D:X), publicDisease_nodes.csv (ID, link), phenotype_nodes.csv, disease_to_publicDisease_edges.csv (exact), phenotype_to_phenotype_edges.csv (is_a), disease_to_phenotype_edges.csv (has)
            # TODO: read maxo-annotations.tsv for treatment/literature, add to disease_to_phenotype_edges
            # TODO: read genes_to_phenotype.txt for disease-phenotype (most specific, no ancestors) genes; expand: 1) disease-phenotype -> n disease-subphenotypes, merge genes; 2) if phenotype is sole child of parent, extend disease-parent genes to disease-phenotype; output gene-phenotype in _export_graph_data (same aggregated disease ID)

            graph_paths = self._export_graph_data(case_database_entries)
            print("Saved graph CSV files:")
            for name, path in graph_paths.items():
                print(f"  {name}: {path}")

            print(f"Saved {len(case_library)} phenotype-disease mappings to {output_file}")
            print(f"Saved {len(high_freq_case_library)} high-frequency phenotype-disease mappings to {output_file_with_high_freq}")
            print(f"Total high-frequency phenotype associations: {high_freq_count}")
            
            return case_library
        
        dataset = []
        filtered_count = 0
        
        # Filter samples by indices if specified
        if sample_indices is not None:
            # Convert to 0-based indexing and filter
            filtered_data = []
            for idx in sample_indices:
                if 0 <= idx < len(input_data):
                    filtered_data.append(input_data[idx])
                else:
                    print(f"Warning: Sample index {idx} is out of range (0-{len(input_data)-1})")
            input_data = filtered_data
            print(f"Filtered to {len(input_data)} samples based on provided indices")
        
        # Limit samples if specified (after filtering by indices)
        if num_samples:
            input_data = input_data[:num_samples]
        
        for i, record in enumerate(input_data):
            # Show progress every 10 records or for the first few records
            if i < 5 or i % 10 == 0 or i == len(input_data) - 1:
                print(f"Processing record {i+1}/{len(input_data)} ({(i+1)/len(input_data)*100:.1f}%)")
            
            if len(record) != 2:
                print(f"  ⚠ Skipping record {i+1} (invalid format)")
                continue
                
            phenotypes = record[0]  # First element is phenotypes
            diseases = record[1]    # Second element is diseases
            
            if not phenotypes or not diseases:
                continue
            
            if use_samples:
                # when testing, exclude given sample's disease data
                prompt_data = self.generate_phenotype_to_disease_prompt(phenotypes, diseases, top_k, 
                    case_extraction_method, embedding_resort, use_frequency_weights, use_IC_weights, use_score, use_samples, prompt_steps, i+1)
            else:
                # using general case library, no need to exclude by sample
                prompt_data = self.generate_phenotype_to_disease_prompt(phenotypes, diseases, top_k, 
                    case_extraction_method, embedding_resort, use_frequency_weights, use_IC_weights, use_score, use_samples, prompt_steps)

            if prompt_data:
                # Create a new ordered dict with sample_id first
                ordered_prompt_data = {
                    'sample_id': i + 1,
                    'task_type': prompt_data['task_type'],
                    'step1_prompt': prompt_data['step1_prompt'],
                    'step2_prompt': prompt_data['step2_prompt'],
                    'step3_prompt': prompt_data['step3_prompt'],
                    'step1_prompt_length': prompt_data['step1_prompt_length'],
                    'step2_prompt_length': prompt_data['step2_prompt_length'],
                    'step3_prompt_length': prompt_data['step3_prompt_length'],
                    'answer': prompt_data['answer'],
                    'answer_length': prompt_data['answer_length'],
                    'phenotypes': prompt_data['phenotypes'],
                    'pheno_details_text': prompt_data['pheno_details_text'],
                    'true_diseases': prompt_data['true_diseases'],
                    'true_disease_ids': prompt_data['true_disease_ids'],
                    'target_disease_in_phenotype_lists': prompt_data['target_disease_in_phenotype_lists'],
                    'target_disease_associated_phenotypes': prompt_data['target_disease_associated_phenotypes'],
                    'ranked_diseases': prompt_data['ranked_diseases'],
                    'ranked_disease_names': prompt_data['ranked_disease_names'],
                    'target_disease_rank': prompt_data['target_disease_rank'],
                    'target_disease_matching_count': prompt_data['target_disease_matching_count'],
                    'target_disease_similarity': prompt_data['target_disease_similarity'],
                    'best_matched_disease': prompt_data['best_matched_disease'],
                    'top_k': prompt_data['top_k'],
                    'case_extraction_method': prompt_data['case_extraction_method']
                }
                # No length filtering - include all prompts
                dataset.append(ordered_prompt_data)
                disease_name = prompt_data['true_diseases'][0] if prompt_data['true_diseases'] else "Unknown"
                num_phenotypes = len(prompt_data['phenotypes'])
                step1_prompt_length = prompt_data['step1_prompt_length']
                step2_prompt_length = prompt_data['step2_prompt_length']
                # print(f"  ✓ Generated {prompt_steps}-step prompt for {disease_name} ({num_phenotypes} phenotypes)")
            else:
                print(f"  ✗ Failed to generate prompt for record {i+1}")
        
        print(f"Generated {len(dataset)} {prompt_steps}-step phenotype-to-disease prompts from input data")
        
        return dataset


def analyze_case_distribution(all_prompts: List[Dict]) -> Dict:
    """
    Analyze the distribution of case counts for 'both' method.
    
    Args:
        all_prompts: List of prompt data dictionaries
        
    Returns:
        Dictionary containing case distribution statistics
    """
    case_counts = []
    
    for prompt_data in all_prompts:
        ranked_diseases = prompt_data.get('ranked_diseases', [])
        case_counts.append(len(ranked_diseases))
    
    if not case_counts:
        return {
            'min_cases': 0,
            'max_cases': 0,
            'avg_cases': 0,
            'median_cases': 0,
        }
    
    return {
        'min_cases': min(case_counts),
        'max_cases': max(case_counts),
        'avg_cases': sum(case_counts) / len(case_counts),
        'median_cases': sorted(case_counts)[len(case_counts)//2],
    }

def analyze_ranking_performance(all_prompts: List[Dict], top_k: int = 20, case_extraction_method: str = "overlap") -> Dict:
    """
    Analyze the performance of disease ranking based on phenotype associations.
    
    Args:
        all_prompts: List of prompt data dictionaries
        top_k: The top_k used for ranking
        case_extraction_method: The case extraction method used
        
    Returns:
        Dictionary containing ranking performance statistics
    """
    # Rank-based statistics
    ranks = []
    top_k_count = 0
    
    for prompt_data in all_prompts:
        # Rank-based logic
        target_rank = prompt_data.get('target_disease_rank')
        if target_rank is not None:
            ranks.append(target_rank)
            if target_rank <= top_k:
                top_k_count += 1
    
    total_samples = len(all_prompts)
    
    # For 'both' method, calculate actual ranking performance without top_k limit
    if case_extraction_method == "both":
        # Count how many samples have target disease ranked at any position
        actual_ranked_count = len(ranks)
        actual_ranked_pct = (actual_ranked_count / total_samples * 100) if total_samples > 0 else 0
        
        return {
            # Actual ranking statistics for 'both' method
            'hit': actual_ranked_count,
            'hit_rate': actual_ranked_pct,
            'avg_rank': sum(ranks) / len(ranks) if ranks else 0,
            'median_rank': sorted(ranks)[len(ranks)//2] if ranks else 0,
            'total_ranked': len(ranks),
            'total_samples': total_samples,
            'unranked_samples': total_samples - len(ranks),
            'method': 'both'
        }
    else:
        # Standard top_k statistics for other methods
        top_k_pct = (top_k_count / total_samples * 100) if total_samples > 0 else 0
        
        # Calculate top1, top5, top10, top20, top30 hit rates
        top1_count = sum(1 for rank in ranks if rank <= 1)
        top5_count = sum(1 for rank in ranks if rank <= 5)
        top10_count = sum(1 for rank in ranks if rank <= 10)
        top20_count = sum(1 for rank in ranks if rank <= 20)
        top30_count = sum(1 for rank in ranks if rank <= 30)
        
        top1_pct = (top1_count / total_samples * 100) if total_samples > 0 else 0
        top5_pct = (top5_count / total_samples * 100) if total_samples > 0 else 0
        top10_pct = (top10_count / total_samples * 100) if total_samples > 0 else 0
        top20_pct = (top20_count / total_samples * 100) if total_samples > 0 else 0
        top30_pct = (top30_count / total_samples * 100) if total_samples > 0 else 0
        
        return {
            # Rank-based statistics
            f'top_{top_k}': top_k_count,
            f'top_{top_k}_pct': top_k_pct,
            'top1': top1_count,
            'top1_pct': top1_pct,
            'top5': top5_count,
            'top5_pct': top5_pct,
            'top10': top10_count,
            'top10_pct': top10_pct,
            'top20': top20_count,
            'top20_pct': top20_pct,
            'top30': top30_count,
            'top30_pct': top30_pct,
            'avg_rank': sum(ranks) / len(ranks) if ranks else 0,
            'median_rank': sorted(ranks)[len(ranks)//2] if ranks else 0,
            'total_ranked': len(ranks),
            'total_samples': total_samples,
            'unranked_samples': total_samples - len(ranks),
            'method': case_extraction_method
        }


def load_config(config_file: str = "config.json") -> dict:
    """Load configuration from JSON file with parameter substitution"""
    try:
        with open(config_file, 'r', encoding='utf-8') as f:
            config = json.load(f)
        
        # Replace {base_path} only (use replace to avoid breaking other placeholders like ${timestamp})
        if 'base_path' in config:
            base_path = config['base_path']
            for key, value in config.items():
                if isinstance(value, str) and '{base_path}' in value:
                    config[key] = value.replace("{base_path}", base_path)
                elif isinstance(value, dict):
                    # Handle nested dictionaries (e.g. output_config, orphanet_files)
                    for subkey, subvalue in value.items():
                        if isinstance(subvalue, str) and '{base_path}' in subvalue:
                            config[key][subkey] = subvalue.replace("{base_path}", base_path)
        
        print(f"Loaded configuration from {config_file}")
        return config
    except FileNotFoundError:
        print(f"Configuration file {config_file} not found, using default values")
        return {}
    except json.JSONDecodeError as e:
        print(f"Error parsing configuration file {config_file}: {e}")
        return {}


def _build_parser() -> argparse.ArgumentParser:
    """Build and return the argument parser."""
    parser = argparse.ArgumentParser(description="Generate Phenotype to Disease Prompts from phenotype-disease data")
    parser.add_argument("--config", type=str, default="prompt_config.json",
                       help="Path to configuration file (default: prompt_config.json)")
    required_group = parser.add_argument_group('Required Input Files')
    required_group.add_argument("--input_file", type=str, default=None,
                               help="Path to input JSON file (phenotype-disease data: [[phenotypes],[diseases]], ...)")
    required_group.add_argument("--disease_mapping", type=str, default=None, help="Path to disease_mapping.json file")
    required_group.add_argument("--obo_file", type=str, default=None, help="Path to HPO OBO file")
    required_group.add_argument("--phenotype_hpoa", type=str, default=None, help="Path to phenotype.hpoa file")
    optional_files_group = parser.add_argument_group('Optional Input Files')
    optional_files_group.add_argument("--phenotype_to_genes", type=str, default=None)
    optional_files_group.add_argument("--genes_to_phenotype", type=str, default=None)
    optional_files_group.add_argument("--genes_to_disease", type=str, default=None)
    embedding_group = parser.add_argument_group('Case Extraction & Embedding Files')
    embedding_group.add_argument("--case_extraction_method", type=str, default=None, choices=["overlap", "embedding", "both"],
                                help="Case extraction method: 'overlap', 'embedding', or 'both'")
    embedding_group.add_argument("--embedding_file", type=str, default=None)
    embedding_group.add_argument("--ic_file", type=str, default=None)
    embedding_group.add_argument("--use_samples", action="store_true", default=None,
                                help="Use phenotype-disease associations from the true case library (--case_library) for ranking; current sample excluded to avoid leakage")
    embedding_group.add_argument("--case_library", type=str, default=None,
                                help="Path to the true case library JSONL. For use_samples: source of phenotype-disease associations. For embedding/both: similar-case pool. Required when use_samples or when case_extraction_method is embedding/both")
    embedding_group.add_argument("--disease_descriptions_file", type=str, default=None)
    embedding_group.add_argument("--embedding_resort", action="store_true", default=None)
    model_group = parser.add_argument_group('Model & Processing Parameters')
    model_group.add_argument("--sentence_transformer_model", type=str, default="FremyCompany/BioLORD-2023")
    model_group.add_argument("--gpu_id", type=int, default=None)
    model_group.add_argument("--prompt_steps", type=int, default=3, choices=[1, 2, 3, 4])
    output_group = parser.add_argument_group('Output & Generation Parameters')
    output_group.add_argument("--max_samples", type=int, default=None)
    output_group.add_argument("--sample_indices", type=str, default=None,
                             help="Comma-separated indices (e.g. '0,5,10' or '0-5,10-15'). Default: process all")
    output_group.add_argument("--top_k", type=int, default=None)
    output_group.add_argument("--output_file", type=str, default=None)
    output_group.add_argument("--use_frequency_weights", action="store_true", default=None)
    output_group.add_argument("--use_IC_weights", action="store_true", default=None)
    output_group.add_argument("--use_score", action="store_true", default=None)
    output_group.add_argument("--save_case_library_only", action="store_true", default=None)
    return parser


def _resolve_params(args: argparse.Namespace, config: dict) -> dict:
    """Merge CLI args and config; return a params dict. Handles 'None' strings and prompt_steps special case."""
    def _get(k: str, default=None):
        v = getattr(args, k, None)
        return v if v is not None else config.get(k, default)

    params = {
        "input_file": _get("input_file"),
        "disease_mapping": _get("disease_mapping"),
        "obo_file": _get("obo_file"),
        "phenotype_hpoa": _get("phenotype_hpoa"),
        "phenotype_to_genes": _get("phenotype_to_genes"),
        "genes_to_phenotype": _get("genes_to_phenotype"),
        "genes_to_disease": _get("genes_to_disease"),
        "use_samples": _get("use_samples", False),
        "case_extraction_method": _get("case_extraction_method", "overlap"),
        "embedding_file": _get("embedding_file"),
        "ic_file": _get("ic_file"),
        "case_library": _get("case_library"),
        "disease_descriptions_file": _get("disease_descriptions_file"),
        "embedding_resort": _get("embedding_resort", False),
        "sentence_transformer_model": _get("sentence_transformer_model", "FremyCompany/BioLORD-2023"),
        "gpu_id": _get("gpu_id"),
        "prompt_steps": args.prompt_steps if getattr(args, "prompt_steps", 3) != 3 else config.get("prompt_steps", 3),
        "max_samples": _get("max_samples"),
        "top_k": _get("top_k", 20),
        "output_file": _get("output_file"),
        "use_frequency_weights": _get("use_frequency_weights", False),
        "use_IC_weights": _get("use_IC_weights", False),
        "use_score": _get("use_score", False),
        "save_case_library_only": _get("save_case_library_only", False),
    }
    if params.get("phenotype_to_genes") == "None":
        params["phenotype_to_genes"] = None
    if params.get("genes_to_phenotype") == "None":
        params["genes_to_phenotype"] = None
    return params


def _validate_params(params: dict) -> None:
    """Validate required and conditional params; raise ValueError on failure."""
    if params["case_extraction_method"] not in ("overlap", "embedding", "both"):
        raise ValueError(f"Invalid case_extraction_method: '{params['case_extraction_method']}'. Valid: overlap, embedding, both")
    if not params["input_file"]:
        raise ValueError("input_file is required (--input_file or config).")
    if not params["obo_file"]:
        raise ValueError("obo_file is required (--obo_file or config).")
    if not params["phenotype_hpoa"]:
        raise ValueError("phenotype_hpoa is required (--phenotype_hpoa or config).")
    if params["case_extraction_method"] in ("embedding", "both"):
        if not params["case_library"]:
            raise ValueError(f"case_library is required when case_extraction_method='{params['case_extraction_method']}' (--case_library or config).")
        if not params["embedding_file"]:
            raise ValueError(f"embedding_file is required when case_extraction_method='{params['case_extraction_method']}'.")
        if not params["ic_file"]:
            raise ValueError(f"ic_file is required when case_extraction_method='{params['case_extraction_method']}'.")
    if params["use_samples"] and not params["case_library"]:
        raise ValueError("case_library is required when use_samples is True (--case_library or config).")


def _build_disease_matching_summary(all_prompts: List[Dict], config: dict) -> List[Dict]:
    """Build the disease_matching_summary list for JSON output."""
    threshold = config.get("evaluation_config", {}).get("similarity_threshold", 0.8)
    out = []
    for i, prompt in enumerate(all_prompts, 1):
        sim = prompt.get("target_disease_similarity", 0.0)
        out.append({
            "sample_id": i,
            "true_diseases": prompt["true_diseases"],
            "best_matched_disease": prompt.get("best_matched_disease"),
            "similarity_score": sim,
            "rank": prompt.get("target_disease_rank"),
            "is_matched": sim >= threshold if sim is not None else False,
        })
    return out


def _build_output_metadata(all_prompts: List[Dict], params: dict, config: dict) -> dict:
    """Build the metadata dict for JSON output."""
    p = params
    metadata = {
        "basic_info": {
            "total_samples": len(all_prompts),
            "generated_at": pd.Timestamp.now().isoformat(),
            "source": os.path.basename(p["input_file"]) if p["input_file"] else "input",
            "task_type": "phenotype_to_disease",
            "prompt_steps": p["prompt_steps"],
            "output_file": p["output_file"],
        },
        "input_files": {
            "input_file": p["input_file"],
            "disease_mapping": p["disease_mapping"],
            "phenotype_hpoa": p["phenotype_hpoa"],
            "obo_file": p["obo_file"],
            "phenotype_to_genes": p["phenotype_to_genes"],
            "genes_to_phenotype": p["genes_to_phenotype"],
            "embedding_file": p["embedding_file"],
            "ic_file": p["ic_file"],
            "case_library": p["case_library"],
        },
        "processing_params": {
            "case_extraction_method": p["case_extraction_method"],
            "top_k": p["top_k"],
            "max_samples": p["max_samples"],
            "use_frequency_weights": p["use_frequency_weights"],
            "use_IC_weights": p["use_IC_weights"],
            "use_samples": p["use_samples"],
        },
        "model_config": {
            "sentence_transformer_model": p["sentence_transformer_model"],
            "gpu_id": p["gpu_id"],
        },
    }
    if p["case_extraction_method"] == "both":
        metadata["case_distribution"] = analyze_case_distribution(all_prompts)
    if all_prompts:
        n = len(all_prompts)
        cnt = sum(1 for x in all_prompts if x.get("target_disease_in_phenotype_lists"))
        metadata["performance_stats"] = {
            "target_diseases_in_phenotype_lists": {"count": cnt, "total": n, "percentage": (cnt / n * 100) if n else 0},
            "ranking_performance": analyze_ranking_performance(all_prompts, p["top_k"], p["case_extraction_method"]),
        }
    return metadata


def _print_config_summary(params: dict, config_path: str) -> None:
    """Print the final configuration summary."""
    p = params
    print(f"\n{'='*60}\nCONFIGURATION SUMMARY\n{'='*60}")
    print(f"📋 Configuration source: {config_path}")
    print("\n📁 INPUT FILES:")
    for k in ("input_file", "disease_mapping", "phenotype_hpoa", "phenotype_to_genes", "genes_to_phenotype", "genes_to_disease", "obo_file", "embedding_file", "ic_file", "case_library"):
        print(f"  • {k}: {p.get(k)}")
    print("\n🤖 MODEL CONFIGURATION:")
    print(f"  • Sentence transformer model: {p['sentence_transformer_model']}\n  • GPU ID: {p['gpu_id']}")
    print("\n⚙️  PROCESSING PARAMETERS:")
    print(f"  • Top K: {p['top_k']}\n  • Max samples: {p['max_samples']}\n  • Prompt steps: {p['prompt_steps']}\n  • Case extraction method: {p['case_extraction_method']}\n  • Use samples: {p['use_samples']}")
    print("\n⚖️  WEIGHT CONFIGURATION:")
    print(f"  • Use frequency weights: {p['use_frequency_weights']}\n  • Use IC weights: {p['use_IC_weights']}")
    print("\n📤 OUTPUT CONFIGURATION:")
    print(f"  • Output file: {p['output_file']}\n{'='*60}")


def main():
    args = _build_parser().parse_args()
    config = load_config(args.config)
    params = _resolve_params(args, config)

    # Parse sample_indices
    if args.sample_indices:
        try:
            params["sample_indices"] = parse_sample_indices(args.sample_indices)
            print(f"Selected sample indices: {params['sample_indices']}")
        except ValueError as e:
            print(f"Error parsing sample indices: {e}")
            return
    else:
        params["sample_indices"] = None

    _validate_params(params)

    if not params["output_file"]:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        fname = f"phenotype_to_disease_prompts_{params['prompt_steps']}steps_out_with_top_{params['top_k']}_{params['case_extraction_method']}_{ts}.json"
        if config.get("base_path"):
            params["output_file"] = os.path.join(config["base_path"], "pho2disease", "prompt", fname)
        else:
            params["output_file"] = os.path.join("prompt", fname)

    # Header and run info
    print("\n" + "="*60)
    print(f"GENERATING {params['prompt_steps']}-STEP PROMPTS FROM INPUT DATA")
    print("="*60)
    print(f"Case extraction method: {params['case_extraction_method']}")
    if params["case_extraction_method"] in ("embedding", "both"):
        print(f"Embedding resort: {params['embedding_resort']}")
    if params["use_samples"]:
        print(f"  Use samples: {params['use_samples']}\n  Case library: {params['case_library']}")

    print("Initializing Phenotype to Disease Prompt Generator...")
    
    set_gpu_id(params["gpu_id"])

    generator = PhenotypeToDiseasePromptGenerator(
        params["disease_mapping"], params["phenotype_hpoa"], params["phenotype_to_genes"],
        params["genes_to_phenotype"], params["genes_to_disease"],
        params["embedding_file"], params["ic_file"], params["case_library"],
        params["disease_descriptions_file"],
        params["sentence_transformer_model"], config,
    )
    print("Loading phenotype names...")
    generator.load_phenotype_names_from_obo(params["obo_file"])
    generator.load_phenotype_disease_mappings()
    generator.merge_disease_phenotypes_by_name()
    generator.merge_disease_synonyms_by_name()
    generator.load_disease_descriptions()
    generator.integrate_frequency_numeric()

    all_prompts = generator.generate_dataset(
        params["input_file"], params["max_samples"], params["top_k"],
        params["use_frequency_weights"], params["case_extraction_method"],
        params["prompt_steps"], params["save_case_library_only"],
        params["use_IC_weights"], params["use_score"], params["use_samples"],
        params["embedding_resort"], params["sample_indices"],
    )

    if params["save_case_library_only"]:
        print("Case library saved, exiting...")
        return

    disease_matching_summary = _build_disease_matching_summary(all_prompts, config)
    metadata = _build_output_metadata(all_prompts, params, config)
    output_data = {"metadata": metadata, "disease_matching_summary": disease_matching_summary, "samples": all_prompts}

    out_dir = os.path.dirname(params["output_file"])
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    with open(params["output_file"], "w", encoding="utf-8") as f:
        json.dump(output_data, f, ensure_ascii=False, indent=2)

    print(f"\n{'='*60}\nFINAL SUMMARY\n{'='*60}")
    print(f"Total {params['prompt_steps']}-step prompts generated: {len(all_prompts)}")
    print(f"Results saved to {params['output_file']}")

    _print_config_summary(params, args.config)
    


if __name__ == "__main__":
    main()