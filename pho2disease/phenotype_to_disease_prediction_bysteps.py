#!/usr/bin/env python3
"""
Phenotype-to-disease prediction: evaluate diagnostic accuracy with single-step,
multi-step (two/three-step) chain-of-thought reasoning.

Backends
--------
- Local Qwen: load model directly (e.g. "Qwen/Qwen3-8B"). Use --model_name and --gpu_id.
- OpenRouter: call OpenRouter API. Use --model_name "openrouter" and optionally --api_model
  to pick the routed model (e.g. "qwen/qwen3-8b:free"). API key and proxy come from
  inference_config openrouter_config or --openrouter_*.

Config (inference_config.json)
---------------------------------------------------
- model_config.default_model_name: default --model_name.
- model_config.enable_thinking: default for thinking/CoT mode (or override with --enable_thinking).

Few-shot
--------
- --use_few_shot: enable similar-case retrieval from a case library.
- --case_library: path to case library JSONL (or set in config file_paths.case_library).
- --k_shot: number of similar samples to inject.

Main CLI
--------
  --prompts_file     Path to prompts JSON (required)
  --output_file      Results JSON path
  --model_name       "Qwen/Qwen3-8B" or "openrouter" (default from config)
  --api_model        OpenRouter model override (e.g. "qwen/qwen3-8b:free")
  --gpu_id           GPU for local Qwen (default 0)
  --enable_thinking  Turn on thinking/CoT
  --sample_indices   Subset to run: "0,5,10" or "0-5,10-15" or "0,5-7,10,20-25"
  --num_samples      Cap number of samples (default: all)
  --max_retries      Retries on failure (default from evaluation_config)

Examples
--------
  # Local Qwen
  python phenotype_to_disease_prediction_bysteps.py --prompts_file prompts_config.json --model_name "Qwen/Qwen3-8B" --gpu_id 2

  # Local Qwen + thinking
  python phenotype_to_disease_prediction_bysteps.py --prompts_file prompts_config.json --model_name "Qwen/Qwen3-8B" --gpu_id 2 --enable_thinking

  # OpenRouter (config default model)
  python phenotype_to_disease_prediction_bysteps.py --prompts_file prompts_config.json --model_name "openrouter"

  # OpenRouter + explicit model
  python phenotype_to_disease_prediction_bysteps.py --prompts_file prompts_config.json --model_name "openrouter" --api_model "qwen/qwen3-8b:free"

  # Subset of samples
  python phenotype_to_disease_prediction_bysteps.py --prompts_file prompts_config.json --sample_indices "0,5-7,10,20-25"

"""

import os
import sys
import json
import torch
import argparse
import re
import time
import pandas as pd
import requests
from typing import Dict, List, Optional, Union
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModelForMaskedLM
from dataclasses import dataclass
from sentence_transformers import SentenceTransformer
import numpy as np
import openai
from openai import OpenAI
from collections import defaultdict
from analysis.disease_scraper.disease_scraper import get_disease_description as disease_description_scraper
import torch
import torch.nn.functional as F  

from generate_prompts_bysteps import (
    parse_sample_indices,
    find_file_path,
    PhenotypeToDiseasePromptGenerator,
)


def load_config(config_file: str = "inference_config.json") -> Dict:
    """Load configuration from JSON file with variable resolution"""
    try:
        from config_utils import load_config_with_variables
        config = load_config_with_variables(config_file)
        print(f"Loaded configuration from {config_file} with variable resolution")
        return config
    except ImportError:
        # Fallback to original method if config_utils is not available
        config_path = os.path.join(os.path.dirname(__file__), config_file)
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Configuration file not found: {config_path}")
        
        with open(config_path, 'r', encoding='utf-8') as f:
            config = json.load(f)
        
        print(f"Loaded configuration from {config_path} (fallback method)")
        return config

@dataclass
class PromptData:
    """Data structure for phenotype-to-disease prompt"""
    task_type: str
    prompt: str
    answer: str
    prompt_length: int
    answer_length: int
    phenotypes: List[str]
    true_diseases: List[str]  # true disease list
    true_disease_ids: List[str]  # true disease id list
    ranked_diseases: List[Dict]  # ranked disease list
    phenotype_disease_mappings: Dict[str, List[str]] = None  # built from ranked_diseases
    sample_id: str = ""  # unique sample id
    # multi-step support
    step1_prompt: str = ""  # step 1 prompt
    step2_prompt: str = ""  # step 2 prompt
    step3_prompt: str = ""  # step 3 prompt
    is_multi_step: bool = False  # whether multi-step reasoning
    # memory reasoning support
    memory_mode: str = None  # memory mode: full, previous, accumulative
    step_prompts: List[str] = None  # step prompts list of arbitrary length

class PhenotypeToDiseasePredictor:
    """Phenotype to Disease Predictor using local Qwen with thinking mode or OpenRouter API"""
    
    def __init__(self, model_name: str = None, cache_dir: str = None, config: Dict = None, gpu_id: int = None, api_key: str = None, proxy_url: str = None, 
                 api_model: str = None, enable_thinking: bool = False):
        """Initialize predictor"""
        # Load configuration
        if config is None:
            config = load_config()
        self.config = config
        self._prompt_generator = None
        
        # Use config values if not provided
        if model_name is None:
            model_name = config['model_config']['default_model_name']
        if cache_dir is None:
            cache_dir = config['model_config']['default_cache_dir']
        
        # Store model name for later use
        self.model_name = model_name
        
        # Store enable_thinking parameter
        self.enable_thinking = enable_thinking
        
        # Check if this is an OpenRouter API model
        self.is_openrouter = model_name.startswith('openrouter') or 'openrouter' in model_name.lower()
        
        # Determine actual model name for OpenRouter API calls
        if self.is_openrouter:
            if api_model:
                self.actual_model_name = api_model
            else:
                openrouter_config = config.get('openrouter_config', {})
                self.actual_model_name = openrouter_config.get('model_name', 'qwen/qwen3-8b:free')
                print(f"Using OpenRouter model: {self.actual_model_name}")
        else:
            self.actual_model_name = model_name

        if self.is_openrouter:
            # Initialize OpenRouter API client
            self._init_openai_model(api_key, proxy_url)
        else:
            # Initialize Qwen model (original logic)
            self._init_qwen_model(cache_dir, gpu_id)
        
        # # Load sentence transformer for semantic similarity
        # print("Loading sentence transformer for semantic similarity...")
        # sentence_model_name = config['evaluation_config']['sentence_transformer_model']
        # # Set cache directory for sentence transformer models from config
        # cache_dir = config.get("sentence_transformer_cache_dir", "../../model_weight")
        # cache_dir = os.path.abspath(cache_dir)
        # print(f"Sentence transformer cache directory: {cache_dir}")
        # self.sentence_model = SentenceTransformer(sentence_model_name, cache_folder=cache_dir, local_files_only=True)
        
        # # # Load BioBERT model and tokenizer
        # # print("Loading BioBERT model...")
        # # try:
        # #     self.biobert_tokenizer = AutoTokenizer.from_pretrained("dmis-lab/biobert-base-cased-v1.2")
        # #     self.biobert_model = AutoModelForMaskedLM.from_pretrained("dmis-lab/biobert-base-cased-v1.2")
        # #     print("BioBERT model loaded successfully!")
        # # except Exception as e:
        # #     print(f"Warning: Failed to load BioBERT model: {e}")
        # #     self.biobert_tokenizer = None
        # #     self.biobert_model = None
        
        # # Load BioLORD-2023 model for biomedical semantic similarity
        # print("Loading BioLORD-2023 model for biomedical semantic similarity...")
        
        # if sentence_model_name != "FremyCompany/BioLORD-2023":
        #     try:
        #         self.biolord_model = SentenceTransformer('FremyCompany/BioLORD-2023', cache_folder=cache_dir)
        #         self.biolord_tokenizer = None  # Will use sentence-transformers tokenizer
        #         print("BioLORD-2023 model loaded successfully!")
        #     except Exception as e:
        #         print(f"Warning: Failed to load BioLORD-2023 model: {e}")
        #         print("Falling back to standard sentence transformer for biomedical similarity...")
        #         self.biolord_model = self.sentence_model
        #         self.biolord_tokenizer = None
        # else:
        #     self.biolord_model = self.sentence_model
        #     self.biolord_tokenizer = None
        
        # Initialize shared prompt generator resources
        print("Initializing phenotype/disease resources...")
        generator = self.prompt_generator  # Ensure generator is created and metadata loaded

        self.sentence_model = generator.sentence_model
        self.biolord_model = self.sentence_model

        # Load phenotype metadata and mappings similarly to prompt generation pipeline
        obo_file = self._resolve_path_option(self.config.get('file_paths', {}).get('obo_file'))
        if obo_file:
            generator.load_phenotype_names_from_obo(obo_file)

        generator.load_phenotype_disease_mappings()
        generator.merge_disease_phenotypes_by_name()
        generator.merge_disease_synonyms_by_name()
        generator.load_disease_descriptions()
        generator.integrate_frequency_numeric()
        
        # per-step output records
        self.insert_step_outputs = {}
        
        # init conversation history for multi-turn
        self.history = []
        
        # pre-encoded disease name embeddings (lazy)
        self._disease_names_embeddings_cache = None
        self._disease_names_cache = None  # name list per disease id for display
        
        print("Model loaded successfully!")
    
    def _init_openai_model(self, api_key: str = None, proxy_url: str = None):
        """Initialize OpenAI-compatible client for OpenRouter API"""
        print(f"Initializing OpenAI client for OpenRouter: {self.model_name}")
        
        config_key = 'openrouter_config'
        model_config = self.config.get(config_key, {})
        
        if api_key is None:
            api_key = model_config.get('api_key', '') or os.getenv('OPENROUTER_API_KEY')
            if not api_key:
                raise ValueError("OpenRouter API key not provided. Set it in config file, OPENROUTER_API_KEY environment variable, or pass api_key parameter.")
        
        if proxy_url is None:
            proxy_url = model_config.get('proxy_url', '') or os.getenv('OPENROUTER_PROXY_URL', 'https://openrouter.ai/api/v1')
        
        if not proxy_url.endswith('/v1'):
            proxy_url = proxy_url.rstrip('/') + '/api/v1'
        
        self.client = OpenAI(api_key=api_key, base_url=proxy_url)
        self.config.setdefault(config_key, {})['api_key'] = api_key
        self.config.setdefault(config_key, {})['proxy_url'] = proxy_url
        print(f"Successfully initialized OpenAI client for OpenRouter proxy: {proxy_url}")

    def _init_qwen_model(self, cache_dir: str, gpu_id: int = None):
        """Initialize Qwen model (original logic)"""
        # Device is already set by CUDA_VISIBLE_DEVICES in main()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # self.device = torch.device("cpu") # force to use cpu
        print(f"Using device: {self.device}")
        
        # Ensure cache_dir is absolute path
        cache_dir = os.path.abspath(cache_dir)
        print(f"Model cache directory: {cache_dir}")
        
        # Check if model exists in HuggingFace cache format
        text_model_dir = os.path.join(cache_dir, f"models--{self.model_name.replace('/', '--')}")
        print(f"Checking for cached model at: {text_model_dir}")
        
        if os.path.exists(text_model_dir):
            print(f"Found cached model at: {text_model_dir}")
            # Use the original model name, transformers will find it in cache
            text_model_to_use = self.model_name
        else:
            print(f"Cached model not found at: {text_model_dir}")
            print(f"Will download model to: {cache_dir}")
            text_model_to_use = self.model_name
        
        # Load model and tokenizer
        print(f"Loading {self.model_name} model with thinking mode...")
        self.tokenizer = AutoTokenizer.from_pretrained(
            text_model_to_use, 
            trust_remote_code=True,
            cache_dir=cache_dir
        )
        
        # Set pad token
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        self.model = AutoModelForCausalLM.from_pretrained(
            text_model_to_use,
            trust_remote_code=True,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            # torch_dtype=torch.float32, # force to use float32
            cache_dir=cache_dir
        ).to(self.device)
    
    def load_prompts(self, prompts_file: str) -> List[PromptData]:
        """Load prompts from JSON file"""
        print(f"Loading prompts from: {prompts_file}")
        
        if not os.path.exists(prompts_file):
            raise FileNotFoundError(f"Prompts file not found: {prompts_file}")
        
        with open(prompts_file, 'r', encoding='utf-8') as f:
            prompts_data = json.load(f)
        
        # Handle both old format (list) and new format (with metadata)
        if isinstance(prompts_data, dict) and 'samples' in prompts_data:
            prompts_list = prompts_data['samples']
        else:
            prompts_list = prompts_data
        
        prompts = []
        for item in prompts_list:
            # Handle missing keys gracefully with default values
            true_diseases = item.get('true_diseases', [])
            true_disease_ids = item.get('true_disease_ids', [])
            ranked_diseases = item.get('ranked_diseases', [])
            
            # build phenotype_disease_mappings from ranked_diseases
            phenotype_disease_mappings = {}
            for ranked_disease in ranked_diseases:
                disease_id = ranked_disease.get('disease_id', '')
                # if disease_id is list, take first
                if isinstance(disease_id, list):
                    disease_id = disease_id[0]
                disease_name = ranked_disease.get('disease_name', '')
                if disease_id and disease_name:
                    phenotype_disease_mappings[disease_id] = [disease_name]
            
            # try to extract sample id from raw data
            # prefer id, then sample_id, then first disease_id
            sample_id = item.get('id', item.get('sample_id', true_disease_ids[0] if true_disease_ids else 'UNKNOWN'))
            
            # check if multi-step prompt
            step1_prompt = item.get('step1_prompt', '')
            step2_prompt = item.get('step2_prompt', '')
            step3_prompt = item.get('step3_prompt', '')
            
            # check for step_prompts list (memory reasoning)
            step_prompts = item.get('step_prompts', [])
            memory_mode = item.get('memory_mode', None)

            # print(f"step1_prompt: {step1_prompt}")
            # print(f"step2_prompt: {step2_prompt}")
            # print(f"step3_prompt: {step3_prompt}")

            is_multi_step = bool(step1_prompt and step2_prompt) or bool(step_prompts)
            
            prompt = PromptData(
                task_type=item.get('task_type', 'phenotype_to_disease'),
                prompt=item.get('prompt', ''),
                answer=item.get('answer', ''),
                prompt_length=item.get('prompt_length', 0),
                answer_length=item.get('answer_length', 0),
                phenotypes=item.get('phenotypes', []),
                true_diseases=true_diseases,
                true_disease_ids=true_disease_ids,
                ranked_diseases=ranked_diseases,
                phenotype_disease_mappings=phenotype_disease_mappings,
                sample_id=sample_id,
                step1_prompt=step1_prompt,
                step2_prompt=step2_prompt,
                step3_prompt=step3_prompt,
                is_multi_step=is_multi_step,
                # memory reasoning support
                memory_mode=memory_mode,
                step_prompts=step_prompts
            )
            prompts.append(prompt)
        
        print(f"Loaded {len(prompts)} prompts")
        return prompts
    
    def load_case_library(self, case_library: str) -> List[Dict]:
        """Load few-shot samples from case library JSONL file"""
        print(f"Loading case library from: {case_library}")
        
        if not os.path.exists(case_library):
            raise FileNotFoundError(f"Case library file not found: {case_library}")
        
        case_samples = []
        with open(case_library, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                if line.strip():
                    try:
                        sample = json.loads(line.strip())
                        case_samples.append(sample)
                    except json.JSONDecodeError as e:
                        print(f"Warning: Invalid JSON at line {line_num}: {e}")
                        continue
        
        print(f"Loaded {len(case_samples)} samples from case library")
        return case_samples
    
    def find_similar_samples(self, current_phenotypes: List[str], case_samples: List[Dict], k_shot: int = 3, 
                           use_embeddings: bool = True) -> List[Dict]:
        """
        Get top-k most similar samples from case library by current phenotypes
        Ref: generate_dynamic_few_shot_id similarity
        Args:
            current_phenotypes: list of phenotypes for current sample
            case_samples: case library sample list
            k_shot: number of similar samples needed
            use_embeddings: whether to use embedding if available
        Returns:
            similar_samples: list of top-k most similar samples
        """
        if not case_samples:
            return []
        
        # try embedding method if available
        if use_embeddings:
            try:
                similar_samples, _ = self._find_similar_samples_with_embeddings(current_phenotypes, case_samples, k_shot)
                return similar_samples
            except Exception as e:
                print(f"Warning: Failed to use embedding method: {e}")
                print("Falling back to Jaccard similarity method")
                return []
        else:
            return []
            
    def _find_similar_samples_with_embeddings(self, current_phenotypes: List[str], case_samples: List[Dict], k_shot: int = 3) -> tuple[List[Dict], List[float]]:
        """
        Compute similarity via embedding (ref: generate_dynamic_few_shot_id)
        """
        # get embedding file path from config
        embedding_file = find_file_path(self.config['file_paths']['mapping_files']['phe2embedding'])
        ic_file = find_file_path(self.config['file_paths']['mapping_files']['ic_dict'])
        
        if not embedding_file or not ic_file:
            raise FileNotFoundError(f"Embedding or IC files not found. Checked paths: {self.config['file_paths']['mapping_files']['phe2embedding']} and {self.config['file_paths']['mapping_files']['ic_dict']}")
        
        # load embedding and IC data
        try:
            phe2embedding = json.load(open(embedding_file, "r", encoding="utf-8-sig"))
            ic_dict = json.load(open(ic_file, "r", encoding="utf-8-sig"))
        except Exception as e:
            raise FileNotFoundError(f"Failed to load embedding or IC files: {e}")
        
        # compute embedding for current sample
        current_embeddings = []
        current_ic_values = []
        
        print(f"Current phenotypes: {current_phenotypes}")
        for phe in current_phenotypes:
            try:
                # extract HP code
                hp_code = self._extract_hp_code(phe)
                print(f"  Extracted HP code: {hp_code} from {phe}")
                
                if hp_code in phe2embedding and hp_code in ic_dict:
                    current_embeddings.append(np.array(phe2embedding[hp_code]))
                    current_ic_values.append(ic_dict[hp_code])
                else:
                    print(f"  Warning: HP code {hp_code} not found in embedding or IC dictionary")
            except Exception as e:
                print(f"  Warning: Error processing phenotype {phe}: {e}")
                continue
        
        if not current_embeddings:
            raise ValueError("No valid phenotypes found in embedding dictionary")
        
        # compute weighted average embedding
        try:
            current_embeddings = np.array(current_embeddings)
            current_ic_values = np.array(current_ic_values)
            current_embedding = np.sum(current_embeddings * current_ic_values.reshape(-1, 1), axis=0) / np.sum(current_ic_values)
        except Exception as e:
            print(f"Warning: Error calculating weighted embedding, using mean: {e}")
            current_embedding = np.mean(current_embeddings, axis=0)
        
        # compute embedding for all case library samples
        candidate_embeddings = []
        valid_samples = []
        
        for sample in case_samples:
            try:
                sample_phenotypes = sample.get('Phenotype', [])
                sample_embeddings = []
                sample_ic_values = []
                
                for phe in sample_phenotypes:
                    try:
                        # extract HP code (same as current sample)
                        hp_code = self._extract_hp_code(phe)
                        
                        if hp_code in phe2embedding and hp_code in ic_dict:
                            sample_embeddings.append(np.array(phe2embedding[hp_code]))
                            sample_ic_values.append(ic_dict[hp_code])
                    except Exception as e:
                        print(f"Warning: Error processing sample phenotype {phe}: {e}")
                        continue
                
                if sample_embeddings:
                    try:
                        sample_embeddings = np.array(sample_embeddings)
                        sample_ic_values = np.array(sample_ic_values)
                        sample_embedding = np.sum(sample_embeddings * sample_ic_values.reshape(-1, 1), axis=0) / np.sum(sample_ic_values)
                        candidate_embeddings.append(sample_embedding)
                        valid_samples.append(sample)
                    except Exception as e:
                        print(f"Warning: Error calculating sample embedding: {e}")
                        continue
            except Exception as e:
                print(f"Warning: Error processing sample: {e}")
                continue
        
        if not candidate_embeddings:
            raise ValueError("No valid samples found with embeddings")
        
        # compute cosine similarity
        try:
            candidate_embeddings = np.array(candidate_embeddings)
            
            # normalize vector
            current_embedding_norm = np.linalg.norm(current_embedding)
            if current_embedding_norm > 0:
                current_embedding_normalized = current_embedding / current_embedding_norm
            else:
                current_embedding_normalized = current_embedding
                
            # normalize candidate vectors
            candidate_norms = np.linalg.norm(candidate_embeddings, axis=1, keepdims=True)
            candidate_embeddings_normalized = np.where(candidate_norms > 0, 
                                                      candidate_embeddings / candidate_norms, 
                                                      candidate_embeddings)
            
            # compute cosine similarity
            cosine_similarities = np.dot(candidate_embeddings_normalized, current_embedding_normalized)

            # sort by similarity descending
            sorted_indices = np.argsort(cosine_similarities)[::-1]
            sorted_cosine_similarities = cosine_similarities[sorted_indices]
            
            print(f"Sorted cosine similarities: {sorted_cosine_similarities[:5]}")  # print top 5 only
            if valid_samples and sorted_indices.size > 0:
                print(f"top1 sample: {valid_samples[sorted_indices[0]]}")
            
            # select top k_shot most similar samples
            similar_samples = []
            similar_values = []
            for i, idx in enumerate(sorted_indices[:k_shot]):
                if idx < len(valid_samples):
                    similar_samples.append(valid_samples[idx])
                    similar_values.append(cosine_similarities[idx])  # use raw similarity array
            
            return similar_samples, similar_values
            
        except Exception as e:
            print(f"Error in similarity calculation: {e}")
            # Fallback: return first k_shot samples
            return valid_samples[:k_shot], [1.0] * min(k_shot, len(valid_samples))
    
    def _write_result_to_file(self, output_file: str, result: Dict, completed_count: int, total_count: int, 
                             correct_count: int, total_time: float):
        """
        Write single result to file and update metadata
        Args:
            output_file: output file path
            result: single result data
            completed_count: number completed
            total_count: total count
            correct_count: number of correct predictions
            total_time: total time
        """
        try:
            # read existing file
            with open(output_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # append to results list
            data['results'].append(result)
            
            # update metadata
            data['metadata']['completed_count'] = completed_count
            data['metadata']['correct_count'] = correct_count
            data['metadata']['total_time'] = total_time
            
            # compute current stats
            if completed_count > 0:
                data['metadata']['average_time'] = total_time / completed_count
                
                # compute top1, top5, top10 accuracy
                valid_results = [r for r in data['results'] if 'evaluation' in r and r['evaluation'].get('best_rank', -1) > 0]
                data['metadata']['valid_count'] = len(valid_results)  # update valid_count
                
                if valid_results:
                    # Top1: use top1_correct
                    top1_correct_count = sum(1 for r in valid_results if r["evaluation"].get("top1_correct", False))
                    data["metadata"]["top1_accuracy"] = top1_correct_count / len(valid_results)
                    
                    # Top5: use top5_correct
                    top5_correct_count = sum(1 for r in valid_results if r["evaluation"].get("top5_correct", False))
                    data["metadata"]["top5_accuracy"] = top5_correct_count / len(valid_results)
                    
                    # Top10: use top10_correct
                    top10_correct_count = sum(1 for r in valid_results if r["evaluation"].get("top10_correct", False))
                    data["metadata"]["top10_accuracy"] = top10_correct_count / len(valid_results)
                    
                else:
                    data['metadata']['top1_accuracy'] = 0.0
                    data['metadata']['top5_accuracy'] = 0.0
                    data['metadata']['top10_accuracy'] = 0.0
            else:
                data['metadata']['average_time'] = 0.0
                data['metadata']['valid_count'] = 0
                data['metadata']['top1_accuracy'] = 0.0
                data['metadata']['top5_accuracy'] = 0.0
                data['metadata']['top10_accuracy'] = 0.0
            
            # update step statistics
            if 'step_statistics' in data['metadata']:
                # detect steps to compute stats for
                steps_to_calculate = self._identify_steps_for_statistics(data['results'])
                
                if steps_to_calculate:
                    print(f"  Auto-detected steps for statistics: {steps_to_calculate}")
                
                # compute step stats dynamically
                for step_num in steps_to_calculate:
                    step_evaluation_key = f"step{step_num}_evaluation"
                    step_stats = self._calculate_step_statistics(data['results'], step_evaluation_key)
                    data['metadata']['step_statistics'][f'step{step_num}'] = step_stats
            
            # check if completed
            if completed_count >= total_count:
                data['metadata']['status'] = 'completed'
            
            # write file (ensure dir exists)
            out_dir = os.path.dirname(output_file)
            if out_dir:
                os.makedirs(out_dir, exist_ok=True)
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
                
        except Exception as e:
            print(f"Warning: Failed to write result to file: {e}")
            import traceback
            traceback.print_exc()
    
    def _extract_hp_code(self, phenotype: str) -> str:
        """
        Extract HP code from phenotype string
        Args:
            phenotype: string, e.g. 'Description (HP:0000000)' or 'HP:0000000'
        Returns:
            hp_code: extracted HP code
        """
        if '(' in phenotype and ')' in phenotype:
            # format: 'Description (HP:0000000)'
            return phenotype.split('(')[-1].split(')')[0]
        elif phenotype.startswith('HP:'):
            # format: 'HP:0000000'
            return phenotype
        else:
            # other formats, use as-is
            return phenotype
    
    def _resolve_path_option(self, option):
        """Resolve path configuration entries that can be strings, lists, or dictionaries."""
        if not option:
            return None
        if isinstance(option, str):
            return option
        if isinstance(option, list):
            resolved = find_file_path(option)
            if resolved:
                return resolved
            return option[0] if option else None
        if isinstance(option, dict):
            for key in ("default_path", "path", "file", "value"):
                if key in option:
                    resolved = self._resolve_path_option(option[key])
                    if resolved:
                        return resolved
            if "paths" in option:
                resolved = self._resolve_path_option(option["paths"])
                if resolved:
                    return resolved
            for value in option.values():
                resolved = self._resolve_path_option(value)
                if resolved:
                    return resolved
            return None
        return None

    def _build_prompt_generator_config(self) -> Dict:
        """Build configuration dictionary compatible with PhenotypeToDiseasePromptGenerator."""
        file_paths = self.config.get('file_paths', {})
        orphanet_config = file_paths.get('orphanet_files', {})

        generator_config: Dict = {}

        mondo_file = self._resolve_path_option(file_paths.get('mondo_file'))
        if mondo_file:
            generator_config['mondo_file'] = mondo_file

        orphanet_files_resolved = {}
        if isinstance(orphanet_config, dict):
            for key in ('alignment_json', 'categorization_csv', 'metadata_tsv', 'phenotype_disease_json'):
                resolved = self._resolve_path_option(orphanet_config.get(key))
                if resolved:
                    orphanet_files_resolved[key] = resolved
        if orphanet_files_resolved:
            generator_config['orphanet_files'] = orphanet_files_resolved

        sentence_cache_dir = self.config.get("sentence_transformer_cache_dir")
        if sentence_cache_dir:
            generator_config["sentence_transformer_cache_dir"] = sentence_cache_dir

        return generator_config

    def _ensure_prompt_generator(self) -> PhenotypeToDiseasePromptGenerator:
        """Create (if needed) and return a shared prompt generator for mapping data."""
        if self._prompt_generator is not None:
            return self._prompt_generator

        file_paths = self.config.get('file_paths', {})
        mapping_files = file_paths.get('mapping_files', {})

        disease_mapping_file = self._resolve_path_option(mapping_files.get('disease_mapping'))
        phenotype_hpoa_file = self._resolve_path_option(file_paths.get('hpoa_file'))
        phenotype_to_genes_file = self._resolve_path_option(mapping_files.get('phenotype_to_genes'))
        genes_to_phenotype_file = self._resolve_path_option(mapping_files.get('genes_to_phenotype'))
        embedding_file = self._resolve_path_option(mapping_files.get('phe2embedding'))
        ic_file = self._resolve_path_option(mapping_files.get('ic_dict'))

        case_library_path = file_paths.get('case_library')
        case_library = self._resolve_path_option(case_library_path) if case_library_path else None

        disease_descriptions_file = self._resolve_path_option(file_paths.get('disease_descriptions_file'))

        sentence_model_name = self.config.get('evaluation_config', {}).get('sentence_transformer_model', "FremyCompany/BioLORD-2023")

        generator_config = self._build_prompt_generator_config()

        self._prompt_generator = PhenotypeToDiseasePromptGenerator(
            disease_mapping_file=disease_mapping_file,
            phenotype_hpoa_file=phenotype_hpoa_file,
            phenotype_to_genes_file=phenotype_to_genes_file,
            genes_to_phenotype_file=genes_to_phenotype_file,
            embedding_file=embedding_file,
            ic_file=ic_file,
            case_library=case_library,
            disease_descriptions_file=disease_descriptions_file,
            sentence_transformer_model=sentence_model_name,
            config=generator_config,
        )
        return self._prompt_generator

    @property
    def prompt_generator(self) -> PhenotypeToDiseasePromptGenerator:
        return self._ensure_prompt_generator()

    @property
    def disease_mapping(self) -> Dict:
        return getattr(self.prompt_generator, "disease_names", {})

    @property
    def disease_mapping_with_synonyms(self) -> Dict:
        return getattr(self.prompt_generator, "disease_mapping_with_synonyms", {})

    @property
    def disease_name_to_ids(self) -> Dict:
        return getattr(self.prompt_generator, "disease_name_to_ids", {})

    @property
    def phenotype_mapping(self) -> Dict:
        return getattr(self.prompt_generator, "phenotype_names", {})

    @property
    def hpo_synonyms(self) -> Dict:
        return getattr(self.prompt_generator, "hpo_synonyms", {})

    @property
    def hpo_definitions(self) -> Dict:
        return getattr(self.prompt_generator, "hpo_definitions", {})

    @property
    def hpo_comments(self) -> Dict:
        return getattr(self.prompt_generator, "hpo_comments", {})

    @property
    def hpo_is_a(self) -> Dict:
        return getattr(self.prompt_generator, "hpo_is_a", {})

    @property
    def parent_to_children(self) -> Dict:
        return getattr(self.prompt_generator, "parent_to_children", {})

    @property
    def phenotype_to_diseases(self) -> Dict:
        return getattr(self.prompt_generator, "phenotype_to_diseases", {})

    @property
    def disease_to_phenotypes(self) -> Dict:
        return getattr(self.prompt_generator, "disease_to_phenotypes", {})

    @property
    def disease_phenotype_counts(self) -> Dict:
        return getattr(self.prompt_generator, "disease_phenotype_counts", {})

    @property
    def phenotype_disease_frequency(self) -> Dict:
        return getattr(self.prompt_generator, "phenotype_disease_frequency", {})

    @property
    def disease_phenotype_frequency(self) -> Dict:
        return getattr(self.prompt_generator, "disease_phenotype_frequency", {})

    @property
    def hpo_frequency_descriptions(self) -> Dict:
        return getattr(self.prompt_generator, "hpo_frequency_descriptions", {})

    @property
    def hpo_frequency_description_to_id(self) -> Dict:
        return getattr(self.prompt_generator, "hpo_frequency_description_to_id", {})

    @property
    def hpo2freq_dict(self):
        return getattr(self.prompt_generator, "hpo2freq_dict", None)

    @property
    def word2freq_dict(self):
        return getattr(self.prompt_generator, "word2freq_dict", None)

    @property
    def disease_descriptions(self) -> Dict:
        return getattr(self.prompt_generator, "disease_descriptions", {})

    @property
    def disease_types(self) -> Dict:
        return getattr(self.prompt_generator, "disease_types", {})

    @property
    def rare_disease_types(self) -> Dict:
        return getattr(self.prompt_generator, "rare_disease_types", {})

    def _is_rare_disease(self, disease_id: str) -> bool:
        """
        Check if disease is rare by disease_id
        
        Args:
            disease_id: disease ID
            
        Returns:
            bool: whether rare disease
        """
        if disease_id == 'UNKNOWN' or not disease_id:
            return False
            
        # check in rare disease type dict
        if self.rare_disease_types:
            return disease_id in self.rare_disease_types
        
        # if dict missing, return False
        return False
    
    def get_orphanet_statistics(self) -> Dict:
        """
        Get Orphanet disease statistics
        Returns:
            statistics: dict with disease counts
        """
        # count Orphanet (ORPHA: prefix only)
        orphanet_diseases = 0
        for disease_id in self.disease_mapping.keys():
            if disease_id.startswith('ORPHA:'):
                orphanet_diseases += 1
        
        # count mappings from files
        csv_mappings = 0
        tsv_mappings = 0
        
        # compare OrphaNumber across files
        # provide total when source unclear
        
        return {
            "total_orphanet_diseases": orphanet_diseases,
            "total_disease_mappings": len(self.disease_mapping),
            "source_files": ["categorization_of_orphanet_diseases.csv", "orphanet_final_disease_metadata.tsv"]
        }
    
    def _print_mapping_statistics(self):
        """Print mapping statistics."""
        print(f"Loaded {len(self.disease_mapping)} diseases, {len(self.phenotype_mapping)} phenotypes")
        print(f"Phenotype-disease mappings: {len(self.phenotype_to_diseases)}")
        print(f"Frequency annotations: {len(self.phenotype_disease_frequency)}")

    def _categorize_frequency(self, frequency: str) -> str:
        """
        Classify frequency value to type
        
        Args:
            frequency: value from HPOA
            
        Returns:
            one of: fraction, percentage, hpo_id, empty, other
        """
        
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
    
    def _get_phenotype_description(self, hp_code: str) -> str:
        """
        Get full phenotype description
        Args:
            hp_code: HP code
        Returns:
            description: full desc or code if not found
        """
        if hp_code in self.phenotype_mapping:
            return self.phenotype_mapping[hp_code]
        return hp_code
    
    def _convert_hpo_frequency_to_description(self, hpo_id: str) -> str:
        """
        Convert HPO freq ID to human-readable description
        
        Args:
            hpo_id: HPO freq ID or multiple sep by '; '
            
        Returns:
            human-readable frequency description
        """
        # handle multiple HPO IDs separated by '; '
        if '; ' in hpo_id:
            hpo_ids = hpo_id.split('; ')
            descriptions = []
            for single_hpo_id in hpo_ids:
                description = self.hpo_frequency_descriptions.get(single_hpo_id.strip(), single_hpo_id.strip())
                if description not in descriptions:  # avoid duplicate
                    descriptions.append(description)
            return '; '.join(descriptions)
        else:
            return self.hpo_frequency_descriptions.get(hpo_id, hpo_id)  # return original ID if not found
    
    def get_phenotype_name(self, hpo_id: str) -> str:
        """Get phenotype name."""
        return self.phenotype_mapping.get(hpo_id, hpo_id)
    
    def get_phenotype_synonyms(self, hpo_id: str) -> List[str]:
        """Get phenotype synonyms."""
        return self.hpo_synonyms.get(hpo_id, [])
    
    def get_phenotype_definition(self, hpo_id: str) -> str:
        """Get phenotype definition."""
        return self.hpo_definitions.get(hpo_id, "")
    
    def get_phenotype_comment(self, hpo_id: str) -> str:
        """Get phenotype annotation."""
        return self.hpo_comments.get(hpo_id, "")
    
    def get_phenotype_full_info(self, hpo_id: str) -> Dict:
        """Get full phenotype info."""
        return {
            'id': hpo_id,
            'name': self.get_phenotype_name(hpo_id),
            'synonyms': self.get_phenotype_synonyms(hpo_id),
            'definition': self.get_phenotype_definition(hpo_id),
            'comment': self.get_phenotype_comment(hpo_id)
        }
    
    def get_disease_name(self, disease_id: str) -> str:
        """Get disease name."""
        return self.disease_mapping.get(disease_id, disease_id)
    
    def get_disease_ids_by_name(self, disease_name: str) -> List[str]:
        """Get disease ID list by name (case-insensitive)"""
        disease_name_lower = disease_name.lower().strip()
        return list(self.disease_name_to_ids.get(disease_name_lower, set()))
    
    def get_phenotype_diseases(self, phenotype_id: str) -> List[str]:
        """Get disease list for phenotype."""
        return list(self.phenotype_to_diseases.get(phenotype_id, set()))
    
    def get_mapping_summary(self) -> Dict:
        """Get mapping summary."""
        # count frequency annotation types
        frequency_type_counts = {}
        for freq_info in self.phenotype_disease_frequency.values():
            freq_type = freq_info['frequency_type']
            frequency_type_counts[freq_type] = frequency_type_counts.get(freq_type, 0) + 1
        
        return {
            "total_diseases": len(self.disease_mapping),
            "total_phenotypes": len(self.phenotype_mapping),
            "unique_disease_names": len(self.disease_name_to_ids),
            "phenotype_to_disease_mappings": len(self.phenotype_to_diseases),
            "disease_to_phenotype_mappings": len(self.disease_to_phenotypes),
            "frequency_annotations": {
                "total": len(self.phenotype_disease_frequency),
                "by_type": frequency_type_counts
            },
            "duplicate_disease_names": sum(1 for ids in self.disease_name_to_ids.values() if len(ids) > 1)
        }
    
    def _get_disease_description(self, disease_id: str) -> str:
        """
        Get full disease description
        Args:
            disease_id: disease ID
        Returns:
            description: full desc or ID if not found
        """
        return self.get_disease_name(disease_id)
    
    def _enrich_sample_info(self, sample: Dict) -> Dict:
        """
        Convert IDs in sample to names, ID in parens
        Args:
            sample: raw sample data
        Returns:
            enriched_sample: sample with names and IDs
        """
        phenotypes = sample.get('Phenotype', [])
        diseases = sample.get('RareDisease', [])
        
        # convert phenotype ID to name, ID in parens
        phenotype_with_ids = []
        for phe in phenotypes:
            hp_code = self._extract_hp_code(phe)
            name = self._get_phenotype_description(hp_code)
            phenotype_with_ids.append(f"{name} ({hp_code})")
        
        # convert disease ID to name, ID in parens
        disease_with_ids = []
        for disease in diseases:
            name = self._get_disease_description(disease)
            disease_with_ids.append(f"{name} ({disease})")
        
        return {
            'phenotypes': phenotype_with_ids,
            'diseases': disease_with_ids
        }
       
    def create_few_shot_prompt(self, current_prompt: str, similar_samples: List[Dict]) -> str:
        """
        Build few-shot prompt from similar samples, append cases in natural language
        Args:
            current_prompt: current prompt
            similar_samples: list of similar samples
        Returns:
            enhanced_prompt: enhanced prompt
        """
        if not similar_samples:
            return current_prompt
        
        # build few-shot examples in natural language
        few_shot_examples = []
        for i, sample in enumerate(similar_samples, 1):
            # use converted names
            enriched_sample = self._enrich_sample_info(sample)
            phenotypes = enriched_sample['phenotypes']
            diseases = enriched_sample['diseases']
            
            # build natural language description
            if phenotypes and diseases:
                phenotype_text = ", ".join(phenotypes)
                disease_text = ", ".join(diseases)
                example = f"The {i}th case has a rare disease {disease_text}, and his/her phenotypes are as follows: {phenotype_text}."
            
            few_shot_examples.append(example)
        
        # combine few-shot examples
        few_shot_text = " ".join(few_shot_examples)
        
        # append few-shot examples to prompt
        enhanced_prompt = f"{current_prompt}\nThe following cases are provided for consideration only:\n{few_shot_text}"
        
        return enhanced_prompt
    
    def _check_reasoning_result(self, content: str, patient_phenotypes: dict = None) -> str:
        """Check if reasoning is invalid; return True if invalid
        
        Args:
            content: model output JSON string
            patient_phenotypes: patient phenotype dict
            
        Returns:
            bool: True=invalid, False=valid
        """
        import json
        
        try:
            # content is multiple JSON objects separated by double newline
            # split by double newline, parse each block
            
            # store extracted data
            phenotype_category = {}
            anchor_clues = []
            key_phenotypic_clusters = []
            supportive_metabolic_laboratory_phenotypes = []
            hypothesis = ""
            final_answer_data = {}
            
            # split JSON blocks by double newline
            json_blocks = content.strip().split('\n\n')
            
            for block in json_blocks:
                block = block.strip()
                if not block:
                    continue
                
                try:
                    # parse each JSON block
                    json_obj = json.loads(block)
                    
                    # extract phenotype classification
                    if 'Functional/System Categories' in json_obj:
                        phenotype_category = json_obj['Functional/System Categories']
                    
                    # extract anchor clues
                    if 'anchor clues' in json_obj:
                        anchor_clues = json_obj['anchor clues']
                    
                    # extract key phenotypic clusters
                    if 'key phenotypic clusters' in json_obj:
                        key_phenotypic_clusters = json_obj['key phenotypic clusters']
                    
                    # extract supportive metabolic/lab phenotypes
                    if 'supportive metabolic/laboratory phenotypes' in json_obj:
                        supportive_metabolic_laboratory_phenotypes = json_obj['supportive metabolic/laboratory phenotypes']
                    
                    # extract hypothesis
                    if 'Hypothesis' in json_obj:
                        hypothesis = json_obj['Hypothesis']
                    
                    # extract final answer
                    if 'FINAL ANSWER' in json_obj:
                        final_answer_data = json_obj['FINAL ANSWER']
                        
                except json.JSONDecodeError as e:
                    # on parse error skip block
                    print(f"Failed to parse JSON block: {e}")
                    continue
            
            # if required data missing, treat as invalid
            if not final_answer_data:
                print("No final answer data found!")
                return "No final answer data found!"
            
            
            # if no patient phenotypes, treat as invalid
            if not patient_phenotypes:
                print("No patient phenotypes provided!")
                return "No patient phenotypes provided!"
            
            # extract disease names from FINAL ANSWER
            disease_names = set()
            for disease_name, description in final_answer_data.items():
                if isinstance(disease_name, str):
                    disease_names.add(disease_name.strip())
            
            # count phenotype occurrences in FINAL ANSWER
            phenotype_occurrence_count = {}
            
            for phenotype, phenotype_id in patient_phenotypes.items():
                count = 0
                
                # search phenotype in disease description
                for disease_name, description in final_answer_data.items():
                    if isinstance(description, str) and phenotype in description.lower():
                        count += 1
                
                phenotype_occurrence_count[phenotype] = count
            
            # check if reasoning is valid
            for phenotype, occurrence_count in phenotype_occurrence_count.items():
                # if phenotype in FINAL ANSWER >= 5 times
                if occurrence_count >= 5:
                    # check if phenotype linked to >100 diseases
                    phenotype_disease_count = len(self.phenotype_to_diseases.get(phenotype_id, set()))
                    # if >100 and >=5 in FINAL ANSWER, treat invalid
                    if phenotype_disease_count > 100:
                        return f"Phenotype {phenotype} is not a typical phenotype!"
            
            return None
            
        except Exception as e:
            # on any parse error, treat invalid
            print(f"Error in check reasoning result: {e}")
            return f"Error in check reasoning result: {e}"

    def add_adjustment_info(self, thinking_content: str, content: str, patient_phenotypes: dict = None) -> str:
        """Add adjustment to prompt from check; extract Matched/Unmatched phenotypes per disease
        
        Args:
            thinking_content: thinking content
            content: model output
            patient_phenotypes: patient phenotype dict
            
        Returns:
            str: adjusted prompt
        """

        # extract disease-phenotype pairs from thinking
        error_in_thinking_prompt = ""
        if thinking_content.strip() != "":
            error_in_thinking = self._check_thinking(thinking_content)
            if error_in_thinking:
                error_in_thinking_prompt = f"""
Issues identified in your thinking process of last step:
-------------------------
\n{error_in_thinking}\n
-------------------------
Please avoid making the same mistake in subsequent reasoning steps.
""" 

        # extract Matched/Unmatched phenotypes per disease
        disease_names, candidate_diseases_validation = self.validate_candidate_diseases(content)
        candidate_diseases_validation_prompt = ""
        if disease_names:
            if candidate_diseases_validation["error_prompts"]:
                candidate_diseases_validation_prompt = f"""                
Issues identified in the **20 Candidate Rare Diseases** list:
-------------------------
\n{"\n".join(candidate_diseases_validation["error_prompts"])}\n
-------------------------
Please provide a revised and more complete list of 20 reasonable candidate diseases in the **20 Candidate Rare Diseases** section.
The candidate diseases in the **20 Candidate Rare Diseases** list **must be specific rare diseases, not broad rare disease types**.
"""

        disease_phenotype_data = self._extract_disease_phenotype_data(content)
        error_matches_prompt = ""
        if disease_phenotype_data:
            error_matches = self.evaluate_phenotype_similarity(disease_phenotype_data, similarity_threshold=0.5)
            if error_matches:
                # error_matches_json = json.dumps(error_matches, ensure_ascii=False, indent=2)
                error_matches_prompt = f"""
Issues identified in the **FINAL ANSWER** disease-phenotype matching:
-------------------------
\n{"\n".join(error_matches)}\n
-------------------------
Please provide a more reliable **FINAL ANSWER** in the last step of the diagnostic process based on the issues pointed out by the expert.
Based on the expert evaluation, please correct the **Matched** and **Unmatched** phenotypes and **Reasoning** information for each candidate disease in the original diagnostic results. 
You can also introduce new diagnoses that have higher phenotype relevance to the patient's manifestations, and remove diseases from the original list that show lower association.
The diagnosed diseases **must be specific rare diseases, not broad rare disease types**.
**You must reorder the diagnosed diseases based on how well each can explain the patient's phenotypes, placing the disease that accounts for the most symptoms at the top.**
"""

        adjustment_prompt = ""
        if error_in_thinking_prompt or candidate_diseases_validation_prompt or error_matches_prompt:
            
            adjustment_prompt = f"""
\nExpert evaluation of the above diagnostic results is as follows:
{error_in_thinking_prompt}\n{candidate_diseases_validation_prompt}\n{error_matches_prompt}
"""
        
        # format prompt from disease_phenotype_data
#         adjustment_prompt = """
# The following is the phenotype association information of the predicted diagnostic results in the known database. 
# Please judge whether the **matched phenotypes** of each disease in the diagnostic list can all be matched to the **known associated phenotypes** in the database.
# If any disease name in the diagnostic list does not appear in the database, or its standard name is 'UNKNOWN', or there are incorrectly associated phenotypes in the **matched phenotypes** section, please reanalyze and output 10 more accurate diagnostic results.
# Please boldly think of other rare diseases with higher phenotype matching degrees, and do not be limited by the previous diagnosis results.
# """
#         if disease_phenotype_data:
#             # build JSON
#             json_data = {}
            
#             for disease_name, data in disease_phenotype_data.items():
#                 all_phenotypes = data.get('all_phenotypes', {})
#                 # extract phenotype name list
#                 phenotype_names_list = list(all_phenotypes.values())
                
#                 json_data[disease_name] = {
#                     "standard_name": data.get('disease_name_standard', disease_name),
#                     "all known associated phenotypes": phenotype_names_list
#                 }
            
#             adjustment_prompt += json.dumps(json_data, ensure_ascii=False, indent=2)
#         else:
#             adjustment_prompt += json.dumps({"error": "Failed to extract disease-phenotype data"}, ensure_ascii=False, indent=2)
        
        return adjustment_prompt

    def _check_thinking(self, thinking_content: str) -> dict:
        """Extract disease-phenotype pairs from thinking
        
        Args:
            thinking_content: thinking content
        """
        # Use self.biobert_tokenizer and self.biobert_model
        # which are loaded during class initialization
        thinking_content = thinking_content.replace('<think>','').replace('</think>','')
        try:
            teacher_evaluation = self._evaluate_with_teacher_openrouter(
                step_label="check_thinking",
                model_thinking=thinking_content,
                teacher_model_name="qwen/qwen3-235b-a22b:free",
            )
        except Exception as e:
            teacher_evaluation = f"[Teacher evaluation failed: {e}]"

        return teacher_evaluation

    def _extract_disease_phenotype_data(self, content: str) -> dict:
        """Extract Matched/Unmatched phenotypes per disease from content
        
        Args:
            content: JSON string with FINAL ANSWER
            
        Returns:
            dict: disease->phenotype match data
        """
        import json
        
        disease_phenotype_data = {}
        
        try:
            # get disease name via existing func
            disease_names = self.extract_disease_names_from_prediction(content)
            
            # extract FINAL ANSWER:{} part
            start = content.find('FINAL ANSWER')
            if start == -1:
                print("    Warning: No FINAL ANSWER: found in content")
                return disease_phenotype_data
            
            # find first { after FINAL ANSWER:
            start_brace = content.find('{', start)
            if start_brace == -1:
                print("No { found after FINAL ANSWER")
                return disease_phenotype_data
            
            # find matching }
            brace_count = 0
            for i in range(start_brace, len(content)):
                if content[i] == '{':
                    brace_count += 1
                elif content[i] == '}':
                    brace_count -= 1
                    if brace_count == 0:
                        # extract FINAL ANSWER:{} and wrap in {}
                        inner_content = content[start_brace:i+1]
                        json_str = '{"FINAL ANSWER":' + inner_content + '}'
                        final_answer_data = json.loads(json_str)
                        break
            else:
                print("No } found after FINAL ANSWER:")
                return disease_phenotype_data
            
            # print(final_answer_data)

            if 'FINAL ANSWER' in final_answer_data:
                for disease_name in disease_names:
                    # find entry with pure disease name in FINAL ANSWER keys
                    for key, disease_info in final_answer_data['FINAL ANSWER'].items():
                        if disease_name in key and isinstance(disease_info, dict):
                            matched = disease_info.get('Matched', [])
                            unmatched = disease_info.get('Unmatched', [])
                            
                            # step 2: get standard name and id
                            disease_info_standard = self._find_best_matching_disease_two_stage(disease_name)
                            disease_id = disease_info_standard.get('disease_id', 'UNKNOWN')
                            
                            # step 3: get all phenotypes for disease id
                            all_phenotypes = {}
                            if disease_id != 'UNKNOWN':
                                phenotype_ids = self.disease_to_phenotypes.get(disease_id, set())
                                for phenotype_id in phenotype_ids:
                                    phenotype_name = self.get_phenotype_name(phenotype_id)
                                    all_phenotypes[phenotype_id] = phenotype_name
                            
                            disease_phenotype_data[disease_name] = {
                                'matched': matched,
                                'unmatched': unmatched,
                                'disease_name_standard': disease_info_standard.get('standard_name', "UNKNOWN"),
                                'disease_id': disease_id,
                                'similarity': disease_info_standard.get('similarity', 0.0),
                                'all_phenotypes': all_phenotypes
                            }
                            break
        
        except json.JSONDecodeError as e:
            print(f"    ERROR: Failed to parse JSON content: {e}")
        except Exception as e:
            print(f"    ERROR: Error extracting disease phenotype data: {e}")
        
        return disease_phenotype_data

    def validate_candidate_diseases(self, content: str) -> Dict:
        """
        Validate 20 candidate diseases from content
        
        Args:
            content: JSON string containing FINAL ANSWER
            
        Returns:
            dict: Dictionary containing validation results
        """
        
        result = {
            "total_diseases": 0,
            "expected_diseases": 20,
            "is_complete": False,
            "disease_validation": [],
            "unmatched_diseases": [],
            "error_prompts": []
        }
        
        try:
            # Extract disease names list
            disease_names = self.extract_candidate_diseases_json(content)
            if not disease_names:
                return disease_names, None

            result["total_diseases"] = len(disease_names)
            result["is_complete"] = len(disease_names) == 20
            
            # print(f"Extracted {len(disease_names)} candidate diseases")
            
            # Validate each disease against standard disease database
            for i, disease_name in enumerate(disease_names, 1):
                # print(f"Validating disease {i}: {disease_name}")
                
                # Check if disease name contains "Sjögren" - skip matching and use original name
                if "sjögren" in disease_name.lower():
                    disease_validation = {
                        "index": i,
                        "original_name": disease_name,
                        "standard_name": disease_name,  # Use original name as standard_name
                        "disease_id": 'OMIM:270150',
                        "similarity": 1.0,  # Set similarity to 1.0 since we're using original name
                        "is_matched": True,  # Not matched in database
                        "is_rare_disease": True  # Cannot determine without disease_id
                    }
                else:
                    # Use existing two-stage matching function
                    matched_disease_info = self._find_best_matching_disease_two_stage(disease_name)
                    
                    # check if rare disease
                    disease_id = matched_disease_info.get('disease_id', 'UNKNOWN')
                    is_rare_disease = self._is_rare_disease(disease_id)
                    
                    disease_validation = {
                        "index": i,
                        "original_name": disease_name,
                        "standard_name": matched_disease_info.get('standard_name', 'UNKNOWN'),
                        "disease_id": disease_id,
                        "similarity": matched_disease_info.get('similarity', 0.0),
                        "is_matched": disease_id != 'UNKNOWN',
                        "is_rare_disease": is_rare_disease
                    }
                
                result["disease_validation"].append(disease_validation)
                
                # If not matched, add to unmatched list
                if not disease_validation["is_matched"]:
                    result["unmatched_diseases"].append(disease_name)
                #     print(f"  ❌ Not matched: {disease_name}")
                # else:
                #     print(f"  ✅ Matched: {matched_disease_info.get('standard_name')} (similarity: {matched_disease_info.get('similarity', 0.0):.3f})")
            
            # Generate summary prompt for all unmatched diseases
            if result["unmatched_diseases"]:
                unmatched_list = "; ".join(result["unmatched_diseases"])
                summary_prompt = f"The following diseases in the predicted **20 Candidate Rare Diseases** are not standalone rare diseases or part of broader categories:\n**{unmatched_list}**\nPlease remove these candidate diseases from the list."
                result["error_prompts"].append(summary_prompt)  
            if not result["is_complete"]:
                summary_prompt = f"The predicted list of **20 Candidate Rare Diseases** contains fewer than 20 diseases."
                result["error_prompts"].append(summary_prompt)  
                
        except Exception as e:
            print(f"Error occurred while validating candidate diseases: {e}")
            result["error"] = str(e)
        
        return disease_names, result

    def generate_prediction(self, prompt: str, max_length: int = 2048, temperature: float = None, max_retries: int = 3, patient_phenotypes: dict = None, final_step: bool = False, use_history: bool = False, addTo_history: bool = False, enable_thinking: Optional[bool] = None, replace_last_step: bool = False) -> Dict:
        """Generate prediction result using Qwen3 format with thinking mode or API models via OpenAI-compatible proxy
        
        Args:
            prompt: Input prompt for the model
            max_length: Maximum length for generation (currently not used)
            temperature: Temperature for generation
            max_retries: Maximum number of retry attempts
            patient_phenotypes: List of patient phenotypes for reasoning validation
            final_step: Whether this is the final step
            use_history: Whether to use conversation history for multi-turn dialogue
            addTo_history: Whether to add this step to conversation history
            enable_thinking: Whether to enable thinking mode (default: self.enable_thinking)
            replace_last_step: Whether to replace the last step in history
        """
        if self.is_openrouter:
            return self._generate_prediction_openai(prompt, temperature, max_retries, patient_phenotypes, final_step, use_history, addTo_history, enable_thinking, replace_last_step)
        else:
            return self._generate_prediction_qwen(prompt, temperature, max_retries=3, patient_phenotypes=patient_phenotypes, final_step=final_step, use_history=use_history, addTo_history=addTo_history, enable_thinking=enable_thinking, replace_last_step=replace_last_step)
    
    def _generate_prediction_openai(self, prompt: str, temperature: float = None, max_retries: int = 3, patient_phenotypes: dict = None, final_step: bool = False, use_history: bool = False, addTo_history: bool = False, enable_thinking: Optional[bool] = None, replace_last_step: bool = False) -> Dict:
        """Generate prediction result using OpenRouter API with thinking mode and retry mechanism
        
        Args:
            prompt: Input prompt for the model
            temperature: Temperature for generation
            max_retries: Maximum number of retry attempts
            patient_phenotypes: List of patient phenotypes for reasoning validation
            final_step: Whether this is the final step
            use_history: Whether to use conversation history for multi-turn dialogue
            addTo_history: Whether to add this step to conversation history
            enable_thinking: Whether to enable thinking mode (default: self.enable_thinking)
            replace_last_step: Whether to replace the last step in history
        """
        start_time = time.time()

        # Use config values if not provided
        if temperature is None:
            temperature = self.config['model_config']['temperature']
        
        # Use instance default if enable_thinking not specified
        if enable_thinking is None:
            enable_thinking = self.enable_thinking

        messages = []
        if not enable_thinking:
            messages += [{"role": "system", "content": "/no_think"}] # stop thinking, required to disable
            print(f"    INFO: Thinking mode is disabled.")
        else:
            messages += [{"role": "system", "content": "/think"}] # start thinking, required to enable
            print(f"    INFO: Thinking mode is enabled.")
        
        # Prepare messages for conversation history support
        # if use_history and history: history+prompt; else prompt only
        if use_history and hasattr(self, 'history') and self.history:
            messages += self.history + [{"role": "user", "content": prompt}]
        else:
            messages += [{"role": "user", "content": prompt}]
        
        model_config = self.config.get('openrouter_config', {})
        retry_delay = model_config.get('retry_delay', 30)  # Start with configured delay
        
        for attempt in range(max_retries):
            try:
                # for Sonoma API: openrouter/sonoma-dusk-alpha, openrouter/sonoma-sky-alpha
                # max_tokens = input_tokens + output_tokens, but `max_new_tokens` is the max tokens for the output
                # TODO: calculate the length of input_tokens
                
                # Prepare request parameters, for OpenAI-compatible API
                request_params = {
                    "model": self.model_name,
                    "messages": messages,
                    "temperature": temperature,
                    "max_tokens": self.config['model_config']['max_new_tokens'],
                    "top_p": self.config['model_config']['top_p']
                }
                
                # Initialize variables
                main_content = ""
                thinking_content = ""
                
                # Generate response using OpenRouter API
                if self.is_openrouter:
                    # Check if OpenRouter payload mode is enabled
                    # Payload mode uses direct HTTP requests instead of OpenAI client
                    # This can be useful for better control over requests or when the OpenAI client has issues
                    openrouter_config = self.config.get('openrouter_config', {})
                    use_payload_mode = openrouter_config.get('use_payload_mode', False)
                    
                    if use_payload_mode:
                        # Use direct HTTP requests for OpenRouter payload mode
                        api_key = openrouter_config.get('api_key', '')
                        if not api_key:
                            raise ValueError("OpenRouter API key not found in config")
                        
                        # Ensure API key has Bearer prefix
                        if not api_key.startswith("Bearer "):
                            api_key = f"Bearer {api_key}"
                        
                        # Get proxy URL
                        proxy_url = openrouter_config.get('proxy_url', 'https://openrouter.ai')
                        if not proxy_url.endswith('/v1'):
                            proxy_url = proxy_url.rstrip('/') + '/v1'
                        
                        think_intervention = """
                            <think> 
                            1. I should analyze the patient's phenotypes provided by the user, giving particular attention to phenotypes that are linked to fewer diseases, as these are usually more diagnostically significant.
                            2. I should use my knowledge of rare diseases to identify which combinations of phenotypes point strongly to specific diseases, rather than evaluating each phenotype in isolation.
                            """
                            # 3. During the diagnostic process, I should thoughtfully recall and provide a diverse set of rare diseases that fit the given combination of phenotypes, rather than simply listing one or two rare diseases that match closely with a specific phenotype combination.
                            # 4. If the user provides reference disease cases, I should analyze the key distinguishing features of each candidate disease and evaluate how well these features match the patient's phenotypes under diagnosis.
                            # 5. In the final diagnostic ranking stage, for each candidate disease, I should fully consider: (a) whether the disease's key distinguishing features are present among the patient's phenotypes, and (b) how many of the patient's phenotypes belong to the common clinical manifestations of this candidate disease. The most reliable diagnosis is one for which the patient's phenotypes include the distinguishing features of the disease, and the majority of the patient's phenotypes can be explained by the disease.

                        # Provider preferences:
                        # - If config gives a string like "fireworks", build {"order": ["fireworks"], "allow_fallbacks": False}
                        # - If config gives an object, pass through.
                        raw_provider = openrouter_config.get('provider')
                        provider_config = {}
                        if isinstance(raw_provider, str) and raw_provider.strip():
                            provider_config = {
                                "order": [raw_provider.strip()],
                                "allow_fallbacks": False,
                            }
                        elif isinstance(raw_provider, dict):
                            provider_config = raw_provider

                        # Prepare payload for OpenRouter API                        
                        # {"role": "assistant", "content": think_intervention}]

                        payload = {
                            "model": self.actual_model_name,
                            "messages": messages,
                            "temperature": temperature,
                            # "max_tokens": self.config['model_config']['max_new_tokens'],
                            "top_p": self.config['model_config']['top_p'],
                            "top_k": self.config['model_config']['top_k'], # supported by some APIs
                            "enable_thinking": enable_thinking, # supported by some APIs
                            # some APIs can disable thinking via below
                            # stop thinking, optional to disable
                            "reasoning": {
                                "enabled": enable_thinking
                            },
                        }

                        # Attach provider settings from config when provided
                        if provider_config:
                            payload['provider'] = provider_config

                        # print(payload)
                        
                        # Make direct HTTP request to OpenRouter API
                        headers = {
                            "Authorization": api_key,
                            "Content-Type": "application/json"
                        }
                        
                        print(f"    INFO: Making OpenRouter payload request to: {proxy_url}/chat/completions")
                        print(f"    INFO: Model: {self.actual_model_name}")
                        
                        response = requests.post(
                            f"{proxy_url}/chat/completions",
                            headers=headers,
                            json=payload,
                            timeout=600  # 1 minutes timeout for complex tasks
                        )

                        # print(f"Time: {response.elapsed.total_seconds():.2f}s | Status: {response.status_code} | Model: {self.actual_model_name} | Response: {response.json()}", file=open("response.log", "a"))
                        
                        if response.status_code == 200:
                            response_data = response.json()
                            # Convert to OpenAI-compatible format
                            class MockResponse:
                                def __init__(self, data):
                                    self.choices = [MockChoice(data.get('choices', [{}])[0])]
                            
                            class MockChoice:
                                def __init__(self, choice_data):
                                    self.message = MockMessage(choice_data.get('message', {}))
                            
                            class MockMessage:
                                def __init__(self, message_data):
                                    self.content = message_data.get('content', '')
                                    # OpenRouter doesn't typically return reasoning content in payload mode
                                    self.reasoning = message_data.get('reasoning', '') or message_data.get('reasoning_content', '')
                            
                            response = MockResponse(response_data)
                        else:
                            # Enhanced error reporting
                            error_details = f"Status: {response.status_code}"
                            try:
                                error_json = response.json()
                                error_details += f", Code: {error_json.get('code', 'N/A')}, Message: {error_json.get('message', 'N/A')}"
                            except:
                                response_text = response.text if response.text is not None else "No response text available"
                                error_details += f", Response: {response_text}"
                            
                            print(f"OpenRouter API Error Details: {error_details}")
                            # print(f"Payload: {json.dumps(payload, indent=2, ensure_ascii=False)}")
                            response_text = response.text if response.text is not None else "No response text available"
                            raise Exception(f"API request failed with status {response.status_code}: {response_text}")
                    else:
                        # Use standard OpenAI client for OpenRouter
                        # Update model name for the request
                        request_params["model"] = self.actual_model_name

                        # OpenRouter works with standard OpenAI client without extra headers
                        print(f"Making OpenAI API request to {self.model_name} with model: {self.actual_model_name}")
                        # print(f"Request params: {json.dumps(request_params, indent=2, ensure_ascii=False)}")
                        response = self.client.chat.completions.create(**request_params)
                
                # print(f"Response: {response}")
                # Extract content from response
                if response is not None:
                    # print(f"Response type: {type(response)}")
                    # print(f"Response: {response}")
                    
                    # Check if response is valid
                    if not hasattr(response, 'choices'):
                        raise Exception(f"Invalid response object: {type(response)}. Expected OpenAI response object with 'choices' attribute.")
                    
                    if response.choices and len(response.choices) > 0:
                        # OpenRouter: extract content and reasoning from message
                        msg = response.choices[0].message
                        main_content = msg.content or ""
                        if hasattr(msg, 'reasoning') and msg.reasoning is not None:
                            thinking_content = msg.reasoning
                        else:
                            thinking_content = ""
                    else:
                        thinking_content = ""
                        main_content = "Error: No response generated"
                else:
                    # Response is None (error case)
                    thinking_content = ""
                    main_content = "Error: No response generated"
                
                # update history before final check, only when addToHistory
                if addTo_history and not replace_last_step and hasattr(self, 'history'):
                    streamlined_prompt = self.remove_arrows_block(prompt)
                    self.history.append({"role": "user", "content": streamlined_prompt})
                    self.history.append({"role": "assistant", "content": main_content})
                # if replace_last_step, replace last assistant reply
                elif addTo_history and replace_last_step and hasattr(self, 'history') and len(self.history) >= 1:
                    self.history.pop()  # pop assistant reply
                    self.history.append({"role": "assistant", "content": main_content})
                    
                if replace_last_step:
                    thinking_content = prompt + "\n--------------------------------\n" + (thinking_content if thinking_content is not None else "")
                    
                # Check if content contains "final answer" keyword
                if main_content.strip() != "":
                    # Success: found final answer or not final step, break out of retry loop
                    break
                else:
                    # No final answer found, retry if not the last attempt
                    if attempt < max_retries - 1:
                        print(f"Attempt {attempt + 1}: No 'content' found in output, retrying...")
                        # pop added history for retry when addToHistory
                        if addTo_history and not replace_last_step and hasattr(self, 'history') and len(self.history) >= 2:
                            self.history.pop()  # pop assistant reply
                            self.history.pop()  # pop user message
                        elif addTo_history and replace_last_step and hasattr(self, 'history') and len(self.history) >= 1:
                            self.history.pop()  # pop assistant reply
                        continue
                    else:
                        print(f"Final attempt {attempt + 1}: No 'content' found, using current output")
                        break
                    
            except Exception as e:
                error_msg = str(e)
                print(f"Attempt {attempt + 1}/{max_retries} failed: {error_msg}")
                                
                # Check if it's a rate limit error
                if "429" in error_msg or "quota" in error_msg.lower() or "rate limit" in error_msg.lower():
                    if attempt < max_retries - 1:  # Don't sleep on last attempt
                        print(f"Rate limit detected. Waiting {retry_delay} seconds before retry...")
                        time.sleep(retry_delay)
                        retry_delay *= 2  # Exponential backoff
                    else:
                        print("Max retries reached for rate limit.")                
                # If this was the last attempt, set error content
                if attempt == max_retries - 1:
                    thinking_content = ""
                    main_content = f"Error after {max_retries} attempts: {error_msg}"
                    # Set response to None to avoid accessing choices attribute
                    response = None
        
        end_time = time.time()
        generation_time = end_time - start_time
        
        return {
            "prediction": main_content,
            "thinking_content": thinking_content,
            "generation_time": generation_time
        }
    
    def _generate_prediction_qwen(self, prompt: str, temperature: float = None, max_retries: int = 3, patient_phenotypes: dict = None, final_step: bool = False, use_history: bool = False, addTo_history: bool = False, enable_thinking: Optional[bool] = None, replace_last_step: bool = False) -> Dict:
        """Generate prediction result using Qwen3 format with thinking mode and retry mechanism
        
        Args:
            prompt: Input prompt for the model
            temperature: Temperature for generation
            max_retries: Maximum number of retry attempts
            patient_phenotypes: List of patient phenotypes for reasoning validation
            final_step: Whether this is the final step
            use_history: Whether to use conversation history for multi-turn dialogue
            addTo_history: Whether to add this step to conversation history
            enable_thinking: Whether to enable thinking mode (default: self.enable_thinking)
        """
        start_time = time.time()

        # Use config values if not provided
        if temperature is None:
            temperature = self.config['model_config']['temperature']
        
        # Use instance default if enable_thinking not specified
        if enable_thinking is None:
            enable_thinking = self.enable_thinking

        if not enable_thinking:
            print(f"    INFO: Thinking mode is disabled.")
        else:
            print(f"    INFO: Thinking mode is enabled.")
        
        # Qwen3 format, multi-turn
        # if use_history and history: history+prompt; else prompt only
        if use_history and hasattr(self, 'history') and self.history:
            messages = self.history + [{"role": "user", "content": prompt}]
        else:
            messages = [{"role": "user", "content": prompt}]
        
        for attempt in range(max_retries):
            try:
                # print(f"Using model: {self.config['model_config']['default_model_name']}")
                if self.config['model_config']['default_model_name'] == "Qwen/Qwen2.5-7B":
                    # For Qwen2.5-7B, use a simpler approach without system message
                    text = self.tokenizer.apply_chat_template(
                        messages,
                        tokenize=False,
                        add_generation_prompt=True,
                    )
                else: # Qwen3 models
                    # Apply chat template with thinking mode enabled
                    text = self.tokenizer.apply_chat_template(
                        messages,
                        tokenize=False,
                        add_generation_prompt=True,
                        enable_thinking=enable_thinking # Enable thinking mode for Qwen3-4B, only used for Qwen3-4B; for Qwen3-4B-Thinking-2507, thinking mode is enabled by default
                    )
                
                # Tokenize input
                model_inputs = self.tokenizer([text], return_tensors="pt").to(self.device)
                max_new_tokens = self.config['model_config']['max_new_tokens']
                
                # Generate answer with thinking mode
                with torch.no_grad():
                    generated_ids = self.model.generate(
                        **model_inputs,
                        max_new_tokens=max_new_tokens,
                        temperature=temperature,
                        top_p=self.config['model_config']['top_p'],
                        top_k=self.config['model_config']['top_k'],
                        do_sample=True,
                        pad_token_id=self.tokenizer.eos_token_id,
                        eos_token_id=self.tokenizer.eos_token_id
                    )                
                
                # Decode output with error handling
                try:
                    output_ids = generated_ids[0][len(model_inputs.input_ids[0]):].tolist()
                    
                    # Different parsing logic for different models
                    if self.config['model_config']['default_model_name'] == "Qwen/Qwen2.5-7B":
                        # Qwen2.5-7B doesn't support thinking mode, treat all output as content
                        try:
                            content = self.tokenizer.decode(output_ids, skip_special_tokens=True).strip("\n")
                            print(f"Qwen2.5-7B decoded content: {content}")
                            thinking_content = ""
                        except Exception as e:
                            print(f"Warning: Error decoding Qwen2.5-7B output: {e}")
                            content = ""
                            thinking_content = ""
                    else:
                        # Parse thinking content for Qwen3 models with thinking mode
                        try:
                            # Find the last occurrence of </think> token (151668)
                            if 151668 in output_ids:
                                # Find the last occurrence
                                index = len(output_ids) - output_ids[::-1].index(151668)
                            else:
                                # If no </think> token found, treat all as content
                                index = 0
                        except (ValueError, IndexError) as e:
                            print(f"Warning: Error parsing thinking content: {e}")
                            index = 0

                        # Safely decode thinking content and main content
                        try:
                            thinking_content = self.tokenizer.decode(output_ids[:index], skip_special_tokens=True).strip("\n")
                        except Exception as e:
                            print(f"Warning: Error decoding thinking content: {e}")
                            thinking_content = ""
                        
                        try:
                            content = self.tokenizer.decode(output_ids[index:], skip_special_tokens=True).strip("\n")
                        except Exception as e:
                            print(f"Warning: Error decoding main content: {e}")
                            content = ""
                        
                        # Ensure we have some content
                        if not content and thinking_content:
                            content = thinking_content
                            thinking_content = ""
                    
                except Exception as e:
                    print(f"Error in output decoding: {e}")
                    # Fallback: try to decode the entire output
                    try:
                        full_output = self.tokenizer.decode(generated_ids[0], skip_special_tokens=True)
                        # Try to extract content after the input prompt
                        if text in full_output:
                            content = full_output.split(text, 1)[1].strip()
                        else:
                            content = full_output
                        thinking_content = ""
                    except Exception as decode_error:
                        print(f"Critical error in fallback decoding: {decode_error}")
                        content = "Error: Failed to decode model output"
                        thinking_content = ""
                
                # update history before final check, only when addToHistory
                if addTo_history and not replace_last_step and hasattr(self, 'history') :
                    streamlined_prompt = self.remove_arrows_block(prompt)
                    self.history.append({"role": "user", "content": streamlined_prompt})
                    self.history.append({"role": "assistant", "content": content})
                # if replace_last_step, replace last assistant reply
                elif addTo_history and replace_last_step and hasattr(self, 'history') and len(self.history) >= 1:
                    self.history.pop()  # pop assistant reply
                    self.history.append({"role": "assistant", "content": content})
                    
                if replace_last_step:
                    thinking_content = prompt + "\n--------------------------------\n" + (thinking_content if thinking_content is not None else "")
                    
                # Check if content contains "final answer" keyword
                if content.strip() != "":
                        break
                    
                    # Success: found final answer, break out of retry loop
                    # break
                
                else:
                    # No final answer found, retry if not the last attempt
                    if attempt < max_retries - 1:
                        print(f"Attempt {attempt + 1}: No 'content' found in output, retrying...")
                        # pop added history for retry when addToHistory
                        if addTo_history and not replace_last_step and hasattr(self, 'history') and len(self.history) >= 2:
                            self.history.pop()  # pop assistant reply
                            self.history.pop()  # pop user message
                        elif addTo_history and replace_last_step and hasattr(self, 'history') and len(self.history) >= 1:
                            self.history.pop()  # pop assistant reply

                        # Clear GPU memory before retry
                        if torch.cuda.is_available():
                            del generated_ids, model_inputs
                            torch.cuda.empty_cache()
                        continue
                    else:
                        print(f"Final attempt {attempt + 1}: No 'content' found, using current output")
                        break
                        
            except Exception as e:
                print(f"Error in generation attempt {attempt + 1}: {e}")
                if attempt < max_retries - 1:
                    print(f"Retrying... (attempt {attempt + 2}/{max_retries})")
                    # Clear GPU memory before retry
                    if torch.cuda.is_available():
                        if 'generated_ids' in locals():
                            del generated_ids
                        if 'model_inputs' in locals():
                            del model_inputs
                        torch.cuda.empty_cache()
                    continue
                else:
                    # Final attempt failed, return error
                    content = f"Error: Failed to generate prediction after {max_retries} attempts: {e}"
                    thinking_content = ""
                    break
        
        end_time = time.time()
        generation_time = end_time - start_time
        
        # Clear GPU memory
        if torch.cuda.is_available():
            if 'generated_ids' in locals():
                del generated_ids
            if 'model_inputs' in locals():
                del model_inputs
            torch.cuda.empty_cache()
        
        return {
            "prediction": content,
            "thinking_content": thinking_content,
            "generation_time": generation_time
        }
    

    def _extract_thinking_content(self, full_content: str) -> tuple[str, str]:
        """
        Extract thinking and main content from full response
        Args:
            full_content: full model response
        Returns:
            tuple: (thinking_content, main_content)
        """
        if not full_content:
            return "", ""
        
        # method 2: find structured thinking format
        structured_patterns = [
            r"<REASONING>(.*?)</REASONING>",
            r"REASONING(.*?)/REASONING",
            r"REASONING(.*?)FINAL ANSWER",
            r"<REASONING>(.*?)FINAL ANSWER:",
        ]
        
        # try structured pattern match
        for pattern in structured_patterns:
            match = re.search(pattern, full_content, re.DOTALL | re.IGNORECASE)
            if match:
                thinking_content = match.group(1).strip()
                main_content = full_content.replace(match.group(0), "").strip()
                return thinking_content, main_content
        
        # if cannot split, return empty thinking and full content
        return "", full_content
    
    def _execute_single_step(self, step_number: int, prompt: str, previous_output: str = None, 
                           placeholder_patterns: List[str] = None, target_phenotypes: List[str] = None, 
                           temperature: float = None) -> Dict:
        """
        Generic single reasoning step
        Args:
            step_number: step number
            prompt: current step prompt
            previous_output: previous output for placeholder
            placeholder_patterns: list of placeholder patterns
            target_phenotypes: target phenotype list
            temperature: temperature
        Returns:
            step_result: step result
        """
        print(f"  Step {step_number}: Generating reasoning...")
        
        # if previous output, replace placeholder
        enhanced_prompt = prompt
        if previous_output and placeholder_patterns:
            placeholder_found = False
            for placeholder in placeholder_patterns:
                if placeholder in prompt:
                    enhanced_prompt = prompt.replace(placeholder, previous_output)
                    print(f"  Found and replaced placeholder: {placeholder}")
                    placeholder_found = True
                    break
            
            if not placeholder_found:
                print(f"  Warning: No placeholder found in step{step_number}_prompt. Available placeholders: {placeholder_patterns}")
        
        # generate prediction
        try:
            phe_id_dict = {phenotype.split('(')[0].strip().lower(): phenotype.split('(')[1].split(')')[0].strip() for phenotype in target_phenotypes}
            step_result = self.generate_prediction(enhanced_prompt, temperature=temperature, patient_phenotypes=phe_id_dict)
            step_prediction = step_result["prediction"]
            step_thinking = step_result["thinking_content"]
            step_time = step_result["generation_time"]
        except Exception as e:
            print(f"  Error in step {step_number} prediction: {e}")
            # return default
            step_prediction = f"Error in step {step_number}: {str(e)}"
            step_thinking = ""
            step_time = 0.0
        
        # extract disease names
        # step_prediction = step_prediction.replace('\"', '')
        step_diseases = self.extract_disease_names_from_prediction(step_prediction)
        # dedup step_diseases, keep order, first occurrence only
        step_diseases = list(dict.fromkeys(step_diseases))

        print(f"  Step {step_number} extracted {len(step_diseases)} disease candidates")
        
        # format output if needed
        step_diseases_str = ""
        if target_phenotypes and step_diseases:
            step_diseases_str = self._format_step_diseases_output(step_diseases, target_phenotypes, step_number)
            # add newline only when formatted non-empty
            # if step_diseases_str:
            #     step_diseases_str = step_diseases_str + "\n"
        
        return {
            "prediction": step_prediction,
            "thinking": step_thinking,
            "diseases": step_diseases,
            "diseases_str": step_diseases_str,
            "time": step_time,
            "enhanced_prompt": enhanced_prompt
        }

    def _execute_single_step_v2(self, step_number: int, prompt: str, previous_output: str = None, 
                           placeholder_patterns: List[str] = None, target_phenotypes: List[str] = None, 
                           temperature: float = None) -> Dict:
        """
        Simplified step: generate and extract after FINAL ANSWER only
        Args:
            step_number: step number
            prompt: current step prompt
            previous_output: previous output for placeholder
            placeholder_patterns: list of placeholder patterns
            target_phenotypes: unused (compat)
            temperature: temperature
        Returns:
            dict: raw pred, thinking, FINAL ANSWER only, time, enhanced prompt
        """
        print(f"  Step {step_number}: Generating reasoning (v2, extract FINAL ANSWER only)...")

        # replace placeholders (same as v1)
        enhanced_prompt = prompt
        if previous_output and placeholder_patterns:
            placeholder_found = False
            for placeholder in placeholder_patterns:
                if placeholder in prompt:
                    enhanced_prompt = prompt.replace(placeholder, previous_output)
                    print(f"  Found and replaced placeholder: {placeholder}")
                    placeholder_found = True
                    break
            if not placeholder_found:
                print(f"  Warning: No placeholder found in step{step_number}_prompt. Available placeholders: {placeholder_patterns}")

        # generate prediction
        try:
            step_result = self.generate_prediction(enhanced_prompt, temperature=temperature)
            step_prediction = step_result.get("prediction", "")
            step_thinking = step_result.get("thinking_content", "")
            step_time = step_result.get("generation_time", 0.0)
        except Exception as e:
            print(f"  Error in step {step_number} prediction (v2): {e}")
            step_prediction = f"Error in step {step_number}: {str(e)}"
            step_thinking = ""
            step_time = 0.0

        # extract text after FINAL ANSWER (case-insensitive)
        try:
            match = re.search(r"(FINAL ANSWER:|Final Answer:)", step_prediction, re.IGNORECASE)
            if match:
                final_answer_text = step_prediction[match.end():].strip()
            else:
                # if no marker, return full prediction
                final_answer_text = step_prediction.strip()
        except Exception as e:
            print(f"  Warning: Error extracting FINAL ANSWER content: {e}")
            final_answer_text = step_prediction.strip()

        return {
            "prediction": step_prediction,
            "thinking": step_thinking,
            "final_answer": final_answer_text,
            # compat v1 return structure, no processing
            "diseases": [],
            "diseases_str": "",
            "time": step_time,
            "enhanced_prompt": enhanced_prompt
        }

    def _execute_single_step_v3(self, step_number: int, prompt: str, previous_output: str = None, 
                           placeholder_patterns: List[str] = None, target_phenotypes: List[str] = None, 
                           temperature: float = None) -> Dict:
        """
        Enhanced step: generate, then process and format output
        Args:
            step_number: step number
            prompt: current step prompt
            previous_output: previous output for placeholder
            placeholder_patterns: list of placeholder patterns
            target_phenotypes: target phenotype list
            temperature: temperature
        Returns:
            dict: raw pred, thinking, formatted disease-phenotype list, time, enhanced prompt
        """
        print(f"  Step {step_number}: Generating reasoning (v3, with enhanced formatting)...")

        # replace placeholders
        enhanced_prompt = prompt
        if previous_output and placeholder_patterns:
            placeholder_found = False
            for placeholder in placeholder_patterns:
                if placeholder in prompt:
                    enhanced_prompt = prompt.replace(placeholder, previous_output)
                    print(f"  Found and replaced placeholder: {placeholder}")
                    placeholder_found = True
                    break
            if not placeholder_found:
                print(f"  Warning: No placeholder found in step{step_number}_prompt. Available placeholders: {placeholder_patterns}")

        # generate prediction
        try:
            step_result = self.generate_prediction(enhanced_prompt, temperature=temperature)
            step_prediction = step_result.get("prediction", "")
            step_thinking = step_result.get("thinking_content", "")
            step_time = step_result.get("generation_time", 0.0)
        except Exception as e:
            print(f"  Error in step {step_number} prediction (v3): {e}")
            step_prediction = f"Error in step {step_number}: {str(e)}"
            step_thinking = ""
            step_time = 0.0

        # extract text after FINAL ANSWER (case-insensitive)
        try:
            match = re.search(r"(FINAL ANSWER:|Final Answer:)", step_prediction, re.IGNORECASE)
            if match:
                final_answer_text = step_prediction[match.end():].strip()
            else:
                # if no marker, return full prediction
                final_answer_text = step_prediction.strip()
        except Exception as e:
            print(f"  Warning: Error extracting FINAL ANSWER content: {e}")
            final_answer_text = step_prediction.strip()

        # further process and format output
        formatted_diseases_str = ""
        extracted_diseases = []
        
        if target_phenotypes and final_answer_text:
            try:
                # parse output, extract disease names and phenotype
                parsed_diseases = self._parse_disease_phenotype_output(final_answer_text)
                
                if parsed_diseases:
                    # get associated phenotypes per disease for patient
                    enhanced_diseases = self._enhance_diseases_with_phenotype_associations(
                        parsed_diseases, target_phenotypes
                    )
                    
                    # build formatted output
                    formatted_diseases_str = self._build_formatted_disease_phenotype_list(
                        enhanced_diseases, step_number
                    )
                    
                    # extract disease names list (compat)
                    extracted_diseases = [disease['disease_name'] for disease in enhanced_diseases]
                    
                    print(f"  Step {step_number}: Enhanced {len(enhanced_diseases)} diseases with phenotype associations")
                else:
                    print(f"  Step {step_number}: No diseases parsed from output")
                    
            except Exception as e:
                print(f"  Warning: Error processing step {step_number} output: {e}")
                formatted_diseases_str = f"Error processing step {step_number} output: {str(e)}"

        return {
            "prediction": step_prediction,
            "thinking": step_thinking,
            "final_answer": final_answer_text,
            "diseases": extracted_diseases,
            "diseases_str": formatted_diseases_str,
            "time": step_time,
            "enhanced_prompt": enhanced_prompt
        }
    
    def _parse_disease_phenotype_output(self, output_text: str) -> List[Dict]:
        """
        Parse disease-phenotype output, extract names and phenotypes
        Args:
            output_text: model output, format e.g.:
                "1. **Schaaf-Yang syndrome**-HP:0000750;HP:0002033;HP:0008897;HP:0008947
                 2. **White-Sutton syndrome**-HP:0002033;HP:0008947;HP:0008897"
                 3. **Schaaf-Yang syndrome** - HP:0001319; HP:0000750; HP:0001612; HP:0002033
        Returns:
            parsed_diseases: list of {name, phenotypes}
        """
        parsed_diseases = []
        
        if not output_text:
            return parsed_diseases
        
        # split by line
        lines = output_text.strip().split('\n')
                
        for line in lines:
            line = line.strip()
            if not line:
                continue
            
            # skip line if not starting with digit
            if not re.match(r'^\d+\.', line):
                continue
            
            try:
                # regex for multiple formats
                patterns = [
                    # format 1: num. **name** - phenotypes (with space)
                    r'^\d+\.\s*\*\*([^*]+)\*\*\s*-\s*(.*)$',
                    r'^\d+\.\s*\*\*([^*]+)\*\*\s*–\s*(.*)$',
                    # format 2: num. **name**-phenotypes (no space)
                    r'^\d+\.\s*\*\*([^*]+)\*\*\s*-\s*([^\s]*)$',    
                    r'^\d+\.\s*\*\*([^*]+)\*\*\s*–\s*([^\s]*)$'
                ]
                
                disease_name = None
                phenotype_list = []
                
                # try to match formats
                for pattern in patterns:
                    match = re.match(pattern, line)
                    if match:
                        disease_name = match.group(1).strip()
                        phenotype_part = match.group(2).strip()
                        
                        # parse phenotype list, semicolon sep
                        if phenotype_part:
                            # clean phenotype ID format
                            phenotype_list = []
                            for p in phenotype_part.split(';'):
                                p = p.strip()
                                if p:
                                    # ensure phenotype ID starts with HP:
                                    if not p.startswith('HP:'):
                                        # if not HP:, try extract HP:xxxxx
                                        hp_match = re.search(r'HP:\d+', p)
                                        if hp_match:
                                            p = hp_match.group()
                                        else:
                                            continue
                                    phenotype_list.append(p)
                        else:
                            phenotype_list = []
                        
                        break
                
                if disease_name:
                    # record disease even without phenotypes
                    parsed_diseases.append({
                        'disease_name': disease_name,
                        'phenotypes': phenotype_list
                    })
                    if phenotype_list:
                        print(f"    Parsed: {disease_name} -> {len(phenotype_list)} phenotypes")
                    else:
                        print(f"    Parsed: {disease_name} -> no phenotypes")
                else:
                    print(f"    Warning: Could not parse line: {line}")
                    
            except Exception as e:
                print(f"    Error parsing line '{line}': {e}")
                continue
        
        return parsed_diseases
    
    # Helper function to recursively get all parent phenotypes
    def get_all_parent_phenotypes(self, phenotype_id: str, visited: set = None, max_depth: int = 1) -> set:
        """
        Recursively get all parent phenotypes (including grandparents, great-grandparents, etc.)
        
        Args:
            phenotype_id: The HPO ID to get parents for
            visited: Set to track visited phenotypes to avoid infinite loops
            max_depth: Maximum depth to explore (default: 1)
            
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
        
        # Recursively get parents of parents (with reduced depth)
        for parent in direct_parents:
            all_parents.update(self.get_all_parent_phenotypes(parent, visited.copy(), max_depth - 1))
        
        return all_parents

    def _is_hp_with_only_child(self, parent_phenotype: str) -> str:
        """Check if a parent phenotype has only one child, return the child if so"""
        # use prebuilt parent-child map for lookup
        children = self.parent_to_children.get(parent_phenotype, [])
        if len(children) == 1:
            return children[0]
        
        # if no prebuilt map, use legacy (backward compat)
        children = []
        for phenotype, parent_nodes in self.hpo_is_a.items():
            if parent_phenotype in parent_nodes:
                children.append(phenotype)
        
        return children[0] if len(children) == 1 else None

    def _enhance_diseases_with_phenotype_associations(self, parsed_diseases: List[Dict], target_phenotypes: List[str]) -> List[Dict]:
        """
        Get associated phenotypes per disease for patient and merge with existing
        Get phenotype info by disease name
        Args:
            parsed_diseases: parsed disease list
            target_phenotypes: target patient phenotypes
        Returns:
            enhanced_diseases: list with merged phenotype info
        """
        enhanced_diseases = []
        
        for disease_info in parsed_diseases:
            disease_name = disease_info['disease_name']
            existing_phenotypes = disease_info['phenotypes']
            
            try:
                # Get phenotype info by disease name
                # use _merge_disease_phenotypes_by_name merged data
                associated_phenotypes = []
                disease_ids = []
                
                # lookup in name-to-id map
                disease_name_lower = disease_name.lower().strip()
                if disease_name_lower in self.disease_name_to_ids:
                    disease_ids = list(self.disease_name_to_ids[disease_name_lower])
                    
                    # get phenotype from disease_phenotype_frequency
                    for disease_id in disease_ids:
                        if disease_id in self.disease_phenotype_frequency:
                            # get all phenotypes and freq for disease
                            disease_phenotypes = self.disease_phenotype_frequency[disease_id]

                            # get parent phenotype map for patient
                            patient_phenotypes_parents = {}
                            for patient_phenotype in target_phenotypes:
                                hp_code = self._extract_hp_code(patient_phenotype)
                                all_parents = self.get_all_parent_phenotypes(hp_code)
                                patient_phenotypes_parents[hp_code] = list(all_parents)

                            # check association: direct and via parent
                            for patient_phenotype in target_phenotypes:
                                hp_code = self._extract_hp_code(patient_phenotype)
                                
                                # direct match
                                if hp_code in disease_phenotypes:
                                    # do not show freq, add phenotype ID
                                    associated_phenotypes.append(f"{hp_code}")
                                elif hp_code in patient_phenotypes_parents:
                                    parent_list = patient_phenotypes_parents[hp_code]
                                    if parent_list:
                                        for parent in parent_list:
                                            if parent in disease_phenotypes:
                                                associated_phenotypes.append(f"{hp_code}(by parent)")
                                                break
                                
                                # check if parent with single child when no match
                                if hp_code not in [p.split(' (')[0] for p in associated_phenotypes if p.startswith('HP:')]:
                                    child_phenotype = self._is_hp_with_only_child(hp_code)
                                    if child_phenotype and child_phenotype in disease_phenotypes:
                                        associated_phenotypes.append(f"{hp_code}(by child)")

                # merge existing and associated, dedup
                # existing=generated, associated=from KB
                # extract HP code, dedup, then merge
                all_phenotypes = []
                used_hp_codes = set()
                
                # 1. add KB-associated phenotypes
                for phenotype in associated_phenotypes:
                    # extract HP code, handle formats:
                    # "HP:0000750" -> "HP:0000750"
                    # "HP:0000750 (by parent)" -> "HP:0000750"
                    # "HP:0000750 (by child)" -> "HP:0000750"
                    # "HP:0000750(Very frequent)" -> "HP:0000750"
                    if phenotype.startswith('HP:'):
                        if ' (' in phenotype:
                            hp_code = phenotype.split(' (')[0]
                        elif '(' in phenotype:
                            hp_code = phenotype.split('(')[0]
                        else:
                            hp_code = phenotype
                    else:
                        hp_code = self._extract_hp_code(phenotype)
                    
                    if hp_code not in used_hp_codes:
                        all_phenotypes.append(phenotype)
                        used_hp_codes.add(hp_code)
                
                # 2. add generated phenotypes (dedup)
                for phenotype in existing_phenotypes:
                    # for HP:xxx format, code is HP:xxx
                    if phenotype.startswith('HP:'):
                        hp_code = phenotype
                    else:
                        hp_code = self._extract_hp_code(phenotype)
                    
                    if hp_code not in used_hp_codes:
                        all_phenotypes.append(phenotype + "(by predict)")
                        used_hp_codes.add(hp_code)
                
                enhanced_diseases.append({
                    'disease_name': disease_name,
                    'disease_ids': disease_ids if disease_ids else ['UNKNOWN'],
                    'existing_phenotypes': existing_phenotypes,
                    'associated_phenotypes': associated_phenotypes,
                    'all_phenotypes': all_phenotypes
                })
                
                # print(f"    Enhanced {disease_name}: {len(existing_phenotypes)} existing + {len(associated_phenotypes)} associated = {len(all_phenotypes)} total")
                # if disease_ids:
                #     print(f"      Found disease IDs: {disease_ids}")
                
            except Exception as e:
                print(f"    Error enhancing disease {disease_name}: {e}")
                # on error keep original
                enhanced_diseases.append({
                    'disease_name': disease_name,
                    'disease_ids': ['UNKNOWN'],
                    'existing_phenotypes': existing_phenotypes,
                    'associated_phenotypes': [],
                    'all_phenotypes': existing_phenotypes
                })
        
        return enhanced_diseases
    
    def _build_formatted_disease_phenotype_list(self, enhanced_diseases: List[Dict], step_number: int) -> str:
        """
        Build formatted disease-phenotype list output
        Support multiple disease IDs
        Args:
            enhanced_diseases: enhanced disease list
            step_number: step number
        Returns:
            formatted_output: formatted output string
        """
        if not enhanced_diseases:
            return f"No diseases identified in Step {step_number}."
        
        formatted_lines = []
        
        for i, disease_info in enumerate(enhanced_diseases, 1):
            disease_name = disease_info['disease_name']
            all_phenotypes = disease_info['all_phenotypes']
            disease_ids = disease_info.get('disease_ids', [])
            
            # build base line: num. **disease name**
            base_line = f"{i}. **{disease_name}**"
            
            # # if disease ID and not UNKNOWN, add ID
            # if disease_ids and disease_ids != ['UNKNOWN']:
            #     # handle multiple disease IDs
            #     if len(disease_ids) == 1:
            #         base_line += f" ({disease_ids[0]})"
            #     else:
            #         # multiple IDs comma-sep
            #         id_str = ", ".join(disease_ids)
            #         base_line += f" ({id_str})"
            
            # add phenotype info
            if all_phenotypes:
                phenotype_str = "; ".join(all_phenotypes)
                base_line += f" - {phenotype_str}"
            
            formatted_lines.append(base_line)
        
        result = "\n".join(formatted_lines)
        result = result + "\n"
        
        return result
   
    def generate_two_step_prediction(self, step1_prompt: str, step2_prompt: str, target_phenotypes: List[str] = None, temperature: float = None) -> Dict:
        """Generate prediction using two-step reasoning approach"""
        print("  Starting multi-step reasoning...")
        
        # Step 1: three strategies for step1 output:
        # v1: extract disease-phenotype, format, optionally add phenotypes
        # v2: extract after FINAL ANSWER only, no processing
        # v3: extract after FINAL ANSWER, then KB merge+format for step2
        step1_result = self._execute_single_step(
            step_number=1,
            prompt=step1_prompt,
            target_phenotypes=target_phenotypes,
            temperature=temperature
        )
        
        # Step 2: Use step 1 output to enhance step 2 prompt
        step2_placeholder_patterns = ["{STEP1_OUTPUT}", "<STEP1_OUTPUT>", "{{STEP1_OUTPUT}}", "[STEP1_OUTPUT]"]
        # Use final_answer from step1 since v2 doesn't populate diseases_str
        step1_output = step1_result.get("diseases_str", step1_result.get("final_answer", ""))
        step2_result = self._execute_single_step(
            step_number=2,
            prompt=step2_prompt,
            # previous_output=step1_result["diseases_str"],
            previous_output=step1_output,
            placeholder_patterns=step2_placeholder_patterns,
            target_phenotypes=target_phenotypes,
            temperature=temperature
        )
        
        total_time = step1_result["time"] + step2_result["time"]
        
        return {
            "step1_prediction": step1_result["prediction"],
            "step1_thinking": step1_result["thinking"],
            "step1_diseases": step1_result["diseases"],
            "step1_diseases_str": step1_result["diseases_str"],
            "step1_time": step1_result["time"],
            "step2_prediction": step2_result["prediction"],
            "step2_thinking": step2_result["thinking"],
            "final_diseases": step2_result["diseases"],
            "step2_time": step2_result["time"],
            "total_time": total_time,
            "enhanced_step2_prompt": step2_result["enhanced_prompt"]
        }
    
    def generate_three_step_prediction(self, step1_prompt: str, step2_prompt: str, step3_prompt: str, target_phenotypes: List[str] = None, temperature: float = None) -> Dict:
        """Generate prediction using three-step reasoning approach, using step1 results as input for step2 and 
        step2 results as input for step3
        """
        print("  Starting three-step reasoning, using step1 results as input for step2 and step2 results as input for step3...")
        
        # Step 1: Generate initial reasoning
        step1_result = self._execute_single_step(
            step_number=1,
            prompt=step1_prompt,
            target_phenotypes=target_phenotypes,
            temperature=temperature
        )
        
        # Step 2: Use step 1 output to enhance step 2 prompt
        step2_placeholder_patterns = ["{STEP1_OUTPUT}", "<STEP1_OUTPUT>", "{{STEP1_OUTPUT}}", "[STEP1_OUTPUT]"]
        step2_result = self._execute_single_step(
            step_number=2,
            prompt=step2_prompt,
            previous_output=step1_result["diseases_str"],
            placeholder_patterns=step2_placeholder_patterns,
            target_phenotypes=target_phenotypes,
            temperature=temperature
        )
        
        # Step 3: Use step 2 output to enhance step 3 prompt
        step3_placeholder_patterns = ["{STEP2_OUTPUT}", "<STEP2_OUTPUT>", "{{STEP2_OUTPUT}}", "[STEP2_OUTPUT]"]
        step3_result = self._execute_single_step(
            step_number=3,
            prompt=step3_prompt,
            previous_output=step2_result["diseases_str"],
            placeholder_patterns=step3_placeholder_patterns,
            target_phenotypes=target_phenotypes,
            temperature=temperature
        )
        
        total_time = step1_result["time"] + step2_result["time"] + step3_result["time"]
        
        return {
            "step1_prediction": step1_result["prediction"],
            "step1_thinking": step1_result["thinking"],
            "step1_diseases": step1_result["diseases"],
            "step1_diseases_str": step1_result["diseases_str"],
            "step1_time": step1_result["time"],
            "step2_prediction": step2_result["prediction"],
            "step2_thinking": step2_result["thinking"],
            "step2_diseases": step2_result["diseases"],
            "step2_diseases_str": step2_result["diseases_str"],
            "step2_time": step2_result["time"],
            "step3_prediction": step3_result["prediction"],
            "step3_thinking": step3_result["thinking"],
            "final_diseases": step3_result["diseases"],
            "step3_time": step3_result["time"],
            "total_time": total_time,
            "enhanced_step2_prompt": step2_result["enhanced_prompt"],
            "enhanced_step3_prompt": step3_result["enhanced_prompt"]
        }
    
    def generate_three_step_prediction_v1(self, step1_prompt: str, step2_prompt: str, step3_prompt: str, target_phenotypes: List[str] = None, temperature: float = None) -> Dict:
        """Generate prediction using three-step reasoning approach, using step1 and step2 results as input for step3"""
        print("  Starting three-step reasoning, using step1 and step2 results as input for step3...")
        
        # Step 1: three strategies for step1 output:
        # v1: extract disease-phenotype, format, optionally add phenotypes
        # v2: extract after FINAL ANSWER only, no processing
        # v3: extract after FINAL ANSWER, then KB merge+format for step2
        step1_result = self._execute_single_step(
            step_number=1,
            prompt=step1_prompt,
            target_phenotypes=target_phenotypes,
            temperature=temperature
        )

        # enhance step2 prompt
        enhanced_step2_prompt = step2_prompt

        # Replace STEP1_OUTPUT placeholder
        step1_placeholders = ["<STEP1_OUTPUT>", "{STEP1_OUTPUT}", "{{STEP1_OUTPUT}}", "[STEP1_OUTPUT]"]
        for placeholder in step1_placeholders:
            if placeholder in enhanced_step2_prompt:
                # Use diseases_str if available (contains formatted phenotype info), otherwise use prediction
                step1_content = step1_result["diseases_str"] if step1_result["diseases_str"] else step1_result["prediction"]
                step1_content = self._remove_numbering(step1_content)
                enhanced_step2_prompt = enhanced_step2_prompt.replace(placeholder, step1_content)
                print(f"  Found and replaced placeholder: {placeholder}")
                break
        
        # Step 2: Generate refined reasoning
        step2_result = self._execute_single_step(
            step_number=2,
            prompt=enhanced_step2_prompt,
            target_phenotypes=target_phenotypes,
            temperature=temperature
        )
        
        # Step 3: Use step 2 outputs to enhance step 3 prompt
        # Create enhanced step 3 prompt by replacing both placeholders
        enhanced_step3_prompt = step3_prompt
        
        # Replace STEP2_OUTPUT placeholder
        step2_placeholders = ["<STEP2_OUTPUT>", "{STEP2_OUTPUT}", "{{STEP2_OUTPUT}}", "[STEP2_OUTPUT]"]
        for placeholder in step2_placeholders:
            if placeholder in enhanced_step3_prompt:
                # Use diseases_str if available (contains formatted phenotype info), otherwise use prediction
                step2_content = step2_result["diseases_str"] if step2_result["diseases_str"] else step2_result["prediction"]
                step2_content = self._remove_numbering(step2_content)
                enhanced_step3_prompt = enhanced_step3_prompt.replace(placeholder, step2_content)
                print(f"  Found and replaced placeholder: {placeholder}")
                break
        
        step3_result = self._execute_single_step(
            step_number=3,
            prompt=enhanced_step3_prompt,
            target_phenotypes=target_phenotypes,
            temperature=temperature
        )
        
        total_time = step1_result["time"] + step2_result["time"] + step3_result["time"]
        
        return {
            "step1_prediction": step1_result["prediction"],
            "step1_thinking": step1_result["thinking"],
            "step1_diseases": step1_result["diseases"],
            "step1_diseases_str": step1_result["diseases_str"],
            "step1_time": step1_result["time"],
            "step2_prediction": step2_result["prediction"],
            "step2_thinking": step2_result["thinking"],
            "step2_diseases": step2_result["diseases"],
            "step2_diseases_str": step2_result["diseases_str"],
            "step2_time": step2_result["time"],
            "step3_prediction": step3_result["prediction"],
            "step3_thinking": step3_result["thinking"],
            "final_diseases": step3_result["diseases"],
            "step3_time": step3_result["time"],
            "total_time": total_time,
            "enhanced_step2_prompt": step2_result["enhanced_prompt"],  # add for compat
            "enhanced_step3_prompt": step3_result["enhanced_prompt"]
        }
    
    def generate_memory_reasoning_prediction(self, step_prompts: List[str], target_phenotypes: List[str] = None, 
                                           temperature: float = None) -> Dict:
        """
        Multi-step with memory, full mode
        
        Args:
            step_prompts: list of step prompts
            target_phenotypes: target phenotype list
            temperature: temperature
        
        Returns:
            dict with all step results
        """
        num_steps = len(step_prompts)
        print(f"  Starting {num_steps}-step memory reasoning (full memory mode)...")
        
        # store all step results
        step_results = []
        total_time = 0.0
        
        for step_num in range(num_steps):
            current_prompt = step_prompts[step_num]
            # step_label: first non-empty Insert Step N or Step N line
            step_label = ""
            for line in current_prompt.split('\n'):
                if line.strip() and (
                    re.match(r"^\**\s*(Insert Step\s*\d+|Step\s*\d+)", line.strip())
                ):
                    step_label = line.strip()
                    break
            
            # full memory: all prior inputs and outputs
            enhanced_prompt = self._build_full_memory_prompt(current_prompt, step_results, step_num, step_prompts, target_phenotypes)
            
            # check if last step
            is_final_step = (step_num == num_steps - 1)

            # run current step
            step_result = self._execute_single_step_with_memory(
                step_number=step_num+1,
                prompt=enhanced_prompt,
                target_phenotypes=target_phenotypes,
                temperature=temperature,
                is_final_step=is_final_step,
                step_label=step_label,
            )

            # add original prompt to result
            step_result["original_prompt"] = current_prompt
            step_results.append(step_result)
            total_time += step_result["time"]

            # (opt) use teacher model to check step for errors
            # try:
            #     teacher_evaluation = self._evaluate_with_teacher_openrouter(
            #         step_label=step_label,
            #         enhanced_prompt=enhanced_prompt,
            #         model_thinking=step_result.get("thinking", ""),
            #         model_prediction=step_result.get("prediction", ""),
            #         teacher_model_name="qwen/qwen3-235b-a22b:free",
            #     )
            # except Exception as e:
            #     teacher_evaluation = f"[Teacher evaluation failed: {e}]"

            # (opt) build new prompt from step and teacher eval, re-reason
            # reinfer_prompt = self._build_reinference_prompt(
            #     base_prompt=enhanced_prompt,
            #     teacher_evaluation=teacher_evaluation,
            #     model_thinking=step_result.get("thinking", ""),
            #     model_prediction=step_result.get("prediction", ""),
            #     step_label=step_label
            # )

            # try:
            #     reinfer_result = self._execute_single_step_with_memory(
            #         step_number=step_num + 1,
            #         prompt=reinfer_prompt,
            #         target_phenotypes=target_phenotypes,
            #         temperature=temperature,
            #         is_final_step=is_final_step,
            #         step_label=f"Reinference of {step_label}",
            #     )
            # except Exception as e:
            #     print(f"Teacher evaluation failed: {str(e)}")
            #     reinfer_result = {
            #         "error": str(e),
            #         "enhanced_prompt": reinfer_prompt,
            #         "prediction": "",
            #         "thinking": "",
            #         "diseases": [],
            #         "diseases_str": "",
            #         "time": 0.0,
            #     }

            # (opt) record teacher eval and re-reason result
            # step_results.append(reinfer_result)
            # total_time += reinfer_result["time"]
        
        # build return result
        result = {
            "total_steps": len(step_results),
            "total_time": total_time,
            "memory_mode": "full",
            "final_diseases": step_results[-1]["diseases"] if step_results else []
        }
        
        # add each step detail
        for i, step_result in enumerate(step_results):
            step_key = f"step{i+1}"
            result.update({
                f"{step_key}_enhanced_prompt": step_result["enhanced_prompt"],
                f"{step_key}_prediction": step_result["prediction"],
                f"{step_key}_thinking": step_result["thinking"],
                f"{step_key}_diseases": step_result["diseases"],
                f"{step_key}_diseases_str": step_result["diseases_str"],
                f"{step_key}_time": step_result["time"]
            })
        
        return result

    def _evaluate_with_teacher_openrouter(self, step_label: str, model_thinking: str, teacher_model_name: str) -> str:
        """
        Evaluate step thinking via OpenRouter teacher model, return text evaluation.
        """
        openrouter_config = self.config.get('openrouter_config', {})
        api_key = openrouter_config.get('api_key') or os.environ.get('OPENROUTER_API_KEY', '')
        if not api_key:
            raise ValueError("OpenRouter API key not found for teacher evaluation")

        proxy_url = openrouter_config.get('proxy_url', 'https://openrouter.ai')
        endpoint = f"{proxy_url.rstrip('/')}/chat/completions"

        system_prompt = (
            "You are an expert medical reasoning evaluator. Review the model's reasoning process. "
            "Identify knowledge errors, logical fallacies, unsupported assumptions and hallucinations"
            "Provide concise, actionable feedback. Keep medical safety in mind."
        )
        user_prompt = (
            f"The following is a disease diagnosis reasoning process provided by a medical expert:\n{model_thinking}\n\n"
            "Please evaluate and return a structured critique (in JSON) with: "
            "1) has_error (yes/no), 2) error_types, 3) critique, 4) suggestions"
        )

        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": teacher_model_name,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "temperature": 0.6,
        }

        try:
            resp = requests.post(endpoint, headers=headers, data=json.dumps(payload), timeout=60)
            resp.raise_for_status()
            data = resp.json()
            content = (
                data.get("choices", [{}])[0]
                    .get("message", {})
                    .get("content", "")
            )
            
            # check and handle <think></think> tag
            if content and "<think>" in content and "</think>" in content:
                # find </think>, keep content after
                think_end = content.find("</think>")
                if think_end != -1:
                    content = content[think_end + len("</think>"):].strip()
            
            return content or "[Empty evaluation from teacher model]"
        except Exception as e:
            raise RuntimeError(f"Teacher model request failed: {e}")

    def _build_reinference_prompt(self, base_prompt: str, teacher_evaluation: str, model_thinking: str, model_prediction: str, step_label: str) -> str:
        """
        Build re-reasoning prompt from teacher eval and prior thinking, guide model to fix conclusion.
        """
        parts = [
            # "\n\n--- Original Task ---\n",
            # base_prompt,
            "\n\n--- Initial Diagnostic Reasoning Process ---\n",
            f"Your previous diagnostic reasoning process was as follows:\n{model_thinking}\n",
            "\n\n--- Expert Evaluation of the Initial Diagnostic Reasoning ---\n",
            f"The following is detailed feedback from the authoritative medical expert:\n{teacher_evaluation}\n",
            "\n\n--- New Task: Re-reason based on the expert’s evaluation ---\n",
            "\n\nPlease revise your previous reasoning process according to the evaluation of authoritative expert. "
            "Be precise, avoid hallucinations, and keep outputs in the exact required format.",
        ]
        return "".join(parts)
    
    def generate_memory_reasoning_prediction_v2(self, step_prompts: List[str], target_phenotypes: List[str] = None, 
                                               temperature: float = None, use_history: bool = True) -> Dict:
        """
        Multi-turn reasoning (Qwen3-8B style), optional history
        
        Args:
            step_prompts: list of step prompts
            target_phenotypes: target phenotype list
            temperature: temperature
            use_history: use conversation history (True=multi, False=single)
        
        Returns:
            dict with all step results
        """
        num_steps = len(step_prompts)
        print(f"  Starting {num_steps}-turn dialogue reasoning v2 (use_history={use_history})...")
        
        # use_history passed via generate_prediction
        
        # store all step results
        step_results = []
        total_time = 0.0
        insert_step_count = 0
        
        for step_num in range(num_steps):
            current_prompt = step_prompts[step_num]
            
            # extract step label
            step_label = ""
            for line in current_prompt.split('\n'):
                if line.strip() and (
                    re.match(r"^\**\s*(Insert Step\s*\d+|Step\s*\d+)", line.strip())
                ):
                    step_label = line.strip()
                    break
            
            # check if insert step
            is_insert_step = self._is_insert_step(current_prompt)
            
            # independent insert: first line has Insert Step and <>
            is_independent_insert_step = False
            if current_prompt:
                lines = current_prompt.strip().split('\n')
                for line in lines:
                    if line.strip():  # first non-empty line
                        if 'Insert Step' in line and '<>' in line:
                            is_independent_insert_step = True
                        break
            
            # if insert step, adjust step number
            if is_insert_step:
                insert_step_count += 1
            
            # check if last step
            is_final_step = (step_num == num_steps - 1)

            # replace placeholders in prompt
            processed_prompt = self._replace_placeholders(current_prompt)

            # run step (existing func, multi-turn)
            step_result = self._execute_single_step_with_memory(
                step_number=step_num+1,
                prompt=processed_prompt,  # use processed prompt
                target_phenotypes=target_phenotypes,
                temperature=temperature,
                is_final_step=is_final_step,
                step_label=step_label,
                use_history=use_history and not is_independent_insert_step,  # independent_insert_step does not use history
                addTo_history=use_history,
                # enable_thinking=True if "Generating and Ranking the Differential Diagnosis" in step_label else False
            )
            
            # if insert step, process output and store
            if is_insert_step:
                insert_step_key = f"<INSERTSTEP{insert_step_count}_OUTPUT>"
                formatted_output = self._process_insert_step_output(
                    step_result["prediction"], 
                    step_result["diseases_str"],
                    insert_step_key, 
                    target_phenotypes, 
                    step_num+1 
                )
                print(f"    INFO: Insert Step {insert_step_count} processed and stored as {insert_step_key}")
            
            # add original prompt to result
            step_result["original_prompt"] = current_prompt
            step_results.append(step_result)
            total_time += step_result["time"]

            # result check
            # TODO: if invalid, append hint and re-reason
            # if "Generating Candidate Rare Diseases" in step_label or "Generating and Ranking the Differential Diagnosis" in step_label:
            #     adjustment_prompt = self.add_adjustment_info(step_result['thinking'], step_result['prediction'], target_phenotypes)
            #     if adjustment_prompt:
            #         print("    Warning: The diagnostic results contain knowledge errors. Re-analyzing and outputting more reliable diagnostic results...")
            #         # run step (existing func, multi-turn)
            #         step_label = step_label + " (Adjustment)"
            #         step_result = self._execute_single_step_with_memory(
            #             step_number=step_num+1,
            #             prompt=adjustment_prompt,  # use processed prompt
            #             target_phenotypes=target_phenotypes,
            #             temperature=temperature,
            #             is_final_step=is_final_step,
            #             step_label=step_label,
            #             use_history=use_history,  # correction must use history; TODO: for independent step only need prev
            #             addTo_history=use_history and not is_insert_step,  # insert step not added to history
            #             replace_last_step=True, # correction only replaces last step output
            #             # enable_thinking=True if "Generating and Ranking the Differential Diagnosis" in step_label else False
            #         )

            #         if is_insert_step:
            #             insert_step_key = f"<INSERTSTEP{insert_step_count}_OUTPUT>"
            #             self.insert_step_outputs.pop(insert_step_key, None)
            #             formatted_output = self._process_insert_step_output(
            #                 step_result["prediction"], 
            #                 step_result["diseases_str"],
            #                 insert_step_key, 
            #                 target_phenotypes, 
            #                 step_num+1 
            #             )
            #             print(f"    INFO:Insert Step {insert_step_count} processed and stored as {insert_step_key}")

            #         step_result["original_prompt"] = adjustment_prompt
            #         step_results.append(step_result)
            #         total_time += step_result["time"]
            #     else:
            #         empty_result = {
            #             "original_prompt": "",
            #             "enhanced_prompt": "",
            #             "prediction": "",
            #             "thinking": "",
            #             "diseases": [],
            #             "diseases_str": "",
            #             "time": 0.0,
            #         }
            #         step_results.append(empty_result)

            # insert step does not save history; clear after
            if is_insert_step and len(self.history) >= 2:
                self.history.pop()
                self.history.pop()
        
            # if "Identifying Key Diagnostic Clues from Observable Symptoms" in step_label:
            # (opt) use teacher to check step for knowledge/logic errors
            #     think_content = step_result.get("thinking", "").replace('<think>','').replace('</think>','')
            #     try:
            #         teacher_evaluation = self._evaluate_with_teacher_openrouter(
            #             step_label=step_label,
            #             enhanced_prompt=processed_prompt,
            #             model_thinking=think_content,
            #             model_prediction=step_result.get("prediction", ""),
            #             teacher_model_name="qwen/qwen3-235b-a22b:free",
            #         )
            #     except Exception as e:
            #         teacher_evaluation = f"[Teacher evaluation failed: {e}]"
            #         print(teacher_evaluation)

            # (opt) build new prompt from step and teacher eval, re-reason
            #     reinfer_prompt = self._build_reinference_prompt(
            #         base_prompt=processed_prompt,
            #         teacher_evaluation=teacher_evaluation,
            #         model_thinking=think_content,
            #         model_prediction=step_result.get("prediction", ""),
            #         step_label=step_label
            #     )

            #     try:
            #         reinfer_result = self._execute_single_step_with_memory(
            #             step_number=step_num + 1,
            #             prompt=reinfer_prompt,
            #             target_phenotypes=target_phenotypes,
            #             temperature=temperature,
            #             is_final_step=is_final_step,
            #             step_label=f"Reinference of {step_label}",
            #             use_history=use_history and not is_independent_insert_step,  # independent_insert_step does not use history
            #             addTo_history=use_history and not is_insert_step  # insert step not added to history
            #         )
            #     except Exception as e:
            #         print(f"Teacher evaluation failed: {str(e)}")
            #         reinfer_result = {
            #             "error": str(e),
            #             "enhanced_prompt": reinfer_prompt,
            #             "prediction": "",
            #             "thinking": "",
            #             "diseases": [],
            #             "diseases_str": "",
            #             "time": 0.0,
            #         }

            # (opt) record teacher eval and re-reason result
            #     reinfer_result["original_prompt"] = reinfer_prompt
            #     step_results.append(reinfer_result)
            #     total_time += reinfer_result["time"]
        
        # build return result
        result = {
            "total_steps": len(step_results),
            "total_time": total_time,
            "memory_mode": "use chat label",
            "final_diseases": step_results[-1]["diseases"] if step_results else []
        }
        
        # add each step detail
        for i, step_result in enumerate(step_results):
            step_key = f"step{i+1}"
            result.update({
                f"{step_key}_prompt": step_result["original_prompt"],  # use original prompt
                f"{step_key}_enhanced_prompt": step_result["enhanced_prompt"],
                f"{step_key}_prediction": step_result["prediction"],
                f"{step_key}_thinking": step_result["thinking"],
                f"{step_key}_diseases": step_result["diseases"],
                f"{step_key}_diseases_str": step_result["diseases_str"],
                f"{step_key}_time": step_result["time"]
            })
        
        return result
    
    def _is_insert_step(self, prompt: str) -> bool:
        """
        Check if insert step
        
        Args:
            prompt: prompt to check
            
        Returns:
            True if is insert step, False otherwise
        """
        if not prompt:
            return False
        
        lines = prompt.strip().split('\n')
        for line in lines:
            if line.strip():  # first non-empty line
                return 'Insert Step' in line
        return False
    
    def _process_insert_step_output(self, step_output: str, diseases_str: str, insert_step_key: str, target_phenotypes: List[str] = None, step_num: int = 1) -> str:
        """
        Process insert step output, extract and format
        
        Args:
            step_output: insert step output
            insert_step_key: insert step key
            target_phenotypes: target phenotype list
            step_num: step number
            
        Returns:
            formatted output string
        """
        # avoid duplicate processing of insert step
        if insert_step_key in self.insert_step_outputs:
            return self.insert_step_outputs[insert_step_key]
        
        # first run: extract disease list from FINAL ANSWER
        if 'FINAL ANSWER' in step_output or 'final answer' in step_output:
            # insert_step_diseases = self.extract_disease_names_from_prediction(step_output)
            # insert_step_diseases = list(dict.fromkeys(insert_step_diseases))
            # format disease list
            # formatted_insert_output = self._format_step_diseases_output(insert_step_diseases, target_phenotypes, step_num)
            formatted_insert_output = diseases_str
        else:
            # extract outermost {} as dict
            match = re.search(r"\{.*\}", step_output, flags=re.DOTALL)
            if match:
                out_block = match.group(0)
                try:
                    out_dict = json.loads(out_block)
                except Exception as e:
                    # if JSON fails, try lenient fix for trailing comma, quotes
                    try:
                        out_block_fixed = out_block.replace("'", '"').replace('\n', '').replace(',}', '}').replace(',]', ']')
                        out_dict = json.loads(out_block_fixed)
                    except Exception as e2:
                        out_dict = {}
            else:
                out_dict = {}
            # convert dict to string for replace
            formatted_insert_output = json.dumps(out_dict, ensure_ascii=False, indent=2)
        
        # cache formatted output
        self.insert_step_outputs[insert_step_key] = formatted_insert_output
        return formatted_insert_output
    
    def _replace_placeholders(self, prompt: str) -> str:
        """
        Replace placeholders in prompt
        
        Args:
            prompt: prompt with placeholders
            
        Returns:
            prompt after replacement
        """
        result_prompt = prompt
        
        # find and replace all insert step placeholders
        for placeholder_key, formatted_output in self.insert_step_outputs.items():
            if placeholder_key in result_prompt:
                result_prompt = result_prompt.replace(placeholder_key, formatted_output)
        
        return result_prompt
    
    # remove <- ... -> and delimiters from original_prompt
    def remove_arrows_block(self, text:str):
        """
        Remove all <- ... -> blocks including delimiters. Multi-line OK.
        """
        # non-greedy match <- ... -> across lines
        return re.sub(r'<-.*?->', '', text, flags=re.DOTALL)

    def _build_full_memory_prompt(self, current_prompt: str, step_results: List[Dict], step_num: int, step_prompts: List[str], target_phenotypes: List[str] = None) -> str:
        """
        Build full-memory prompt with all prior inputs and outputs
        
        Args:
            current_prompt: current step prompt
            step_results: prior step results
            step_num: current step index (0-based)
            step_prompts: all step prompts
        
        Returns:
            enhanced prompt
        """
        if step_num == 0:
            return current_prompt
        
        # build memory context, full prompt
        # Step N prompt = prior prompts (templates replaced) + current
        
        # get all raw step prompts
        all_original_prompts = step_prompts
        
        # build full prompt
        enhanced_prompt = ""
        
        insert_step_count = 0
        # add prior prompts with templates replaced
        for i in range(len(step_results)):
            step_num = i + 1 - insert_step_count
            step_output = step_results[i].get('prediction', 'N/A')
            
            # get raw prompt
            original_prompt = all_original_prompts[i]

            # check if insert step
            # insert step if first non-empty has Insert Step
            is_insert_step = False
            is_independent_insert_step = False
            if original_prompt:
                lines = original_prompt.strip().split('\n')
                for line in lines:
                    if line.strip():  # first non-empty line
                        is_insert_step = 'Insert Step' in line
                        # if Insert Step and <>, step is independent
                        if is_insert_step and '<>' in line:
                            is_independent_insert_step = True
                        break
            
            # if temp insert, do not add to later prompts; normalize and put in placeholder
            if is_insert_step:
                insert_step_count += 1
                # avoid duplicate processing of insert step
                insert_step_key = f"<INSERTSTEP{insert_step_count}_OUTPUT>"
                
                # if processed, use cache
                # note: clear insert_step_outputs per sample
                if insert_step_key in self.insert_step_outputs:
                    formatted_insert_output = self.insert_step_outputs[insert_step_key]
                else:
                    # first run: extract disease list from FINAL ANSWER
                    if 'FINAL ANSWER' in step_output or 'final answer' in step_output :
                        insert_step_diseases = self.extract_disease_names_from_prediction(step_output)
                        insert_step_diseases = list(dict.fromkeys(insert_step_diseases))
                        # format disease list
                        formatted_insert_output = self._format_step_diseases_output(insert_step_diseases, target_phenotypes, step_num+1)
                        # cache formatted output
                        self.insert_step_outputs[insert_step_key] = formatted_insert_output
                    else:
                        # extract outermost {} as dict
                        match = re.search(r"\{.*\}", step_output, flags=re.DOTALL)
                        if match:
                            out_block = match.group(0)
                            try:
                                out_dict = json.loads(out_block)
                            except Exception as e:
                                # if JSON fails, try lenient fix for trailing comma, quotes
                                try:
                                    out_block_fixed = out_block.replace("'", '"').replace('\n', '').replace(',}', '}').replace(',]', ']')
                                    out_dict = json.loads(out_block_fixed)
                                except Exception as e2:
                                    out_dict = {}
                        else:
                            out_dict = {}
                        # convert dict to string for replace
                        formatted_insert_output = json.dumps(out_dict, ensure_ascii=False, indent=2)
                        self.insert_step_outputs[insert_step_key] = formatted_insert_output
                
                # find and replace <INSERTSTEP1_OUTPUT> in next prompt
                # replace in all following prompts
                for j in range(i + 1, len(all_original_prompts)):
                    if insert_step_key in all_original_prompts[j]:
                        all_original_prompts[j] = all_original_prompts[j].replace(insert_step_key,formatted_insert_output)
                    
                # replace <INSERTSTEP1_OUTPUT> in current (always)
                if insert_step_key in current_prompt:
                    current_prompt = current_prompt.replace(insert_step_key, formatted_insert_output)
                
                    if is_independent_insert_step:
                        return current_prompt

                # skip insert step in enhanced_prompt
                continue
            
            
            original_prompt = self.remove_arrows_block(original_prompt)
            
            # find and replace Step N output format
            pattern = f"**Step {step_num} output format in json (must use json format, not markdown):**"
            if pattern in original_prompt:
                # find pattern
                start_pos = original_prompt.find(pattern)
                if start_pos != -1:
                    # find pattern end
                    pattern_end = start_pos + len(pattern)
                    
                    # find template start `{`
                    template_start = original_prompt.find("{", pattern_end)
                    if template_start != -1:
                        # find matching `}`
                        template_end = original_prompt.find("}", template_start)
                        if template_end != -1:
                            template_end += 1  # include `}`
                            
                            # replace template with actual, shorten format mark
                            before = original_prompt[:start_pos] + f"**Step {step_num} output:**"
                            after = original_prompt[template_end:]
                            enhanced_prompt += before + step_output + after
                        else:
                            # if no `}`, use raw prompt
                            enhanced_prompt += original_prompt
                    else:
                        # if no `{`, use raw prompt
                        enhanced_prompt += original_prompt
                else:
                    enhanced_prompt += original_prompt
            else:
                enhanced_prompt += original_prompt
        
        # add current raw prompt
        enhanced_prompt += current_prompt
        
        return enhanced_prompt
    
    def _format_history_as_string(self) -> str:
        """
        Format conversation history to string
        
        Returns:
            formatted history string
        """
        if not hasattr(self, 'history') or not self.history:
            return ""
        
        history_str = ""
        for message in self.history:
            role = message.get("role", "")
            content = message.get("content", "")
            if role == "user":
                history_str += f"User: {content}\n"
            elif role == "assistant":
                history_str += f"Assistant: {content}\n"
        
        return history_str.strip()
    
    
    def _execute_single_step_with_memory(self, step_number: int, prompt: str, target_phenotypes: List[str] = None, 
                                       temperature: float = None, is_final_step: bool = False, step_label: str = None, 
                                       use_history: bool = False, addTo_history: bool = False, enable_thinking: Optional[bool] = None,
                                       replace_last_step: bool = False) -> Dict:
        """
        Execute single step with memory
        
        Args:
            step_number: step number
            prompt: current step prompt
            target_phenotypes: target phenotype list
            temperature: temperature
            is_final_step: whether last step
            step_label: step label
            use_history: use history
            addTo_history: add current step to history
        
        Returns:
            step result
        """
        print(f"  Dialogue {step_number}: {step_label}...")
        
        # generate prediction
        try:
            phe_id_dict = {phenotype.split('(')[0].strip().lower(): phenotype.split('(')[1].split(')')[0].strip() for phenotype in target_phenotypes} if target_phenotypes else {}
            step_result = self.generate_prediction(prompt, temperature=temperature, patient_phenotypes=phe_id_dict, final_step=is_final_step, use_history=use_history, addTo_history=addTo_history, enable_thinking=enable_thinking, replace_last_step=replace_last_step)
            step_prediction = step_result["prediction"]
            step_thinking = step_result["thinking_content"]
            step_time = step_result["generation_time"]
        except Exception as e:
            print(f"  Error in step {step_number} prediction: {e}")
            step_prediction = f"Error in step {step_number}: {str(e)}"
            step_thinking = ""
            step_time = 0.0
        
        # if FINAL ANSWER in step_prediction, extract diseases
        # last step: extract disease names
        if is_final_step or 'FINAL ANSWER' in step_prediction or 'final answer' in step_prediction:
            step_diseases = self.extract_disease_names_from_prediction(step_prediction)
            step_diseases = list(dict.fromkeys(step_diseases))  # dedup, keep order
            print(f"  Dialogue {step_number}: extracted {len(step_diseases)} disease candidates")
            
            # format output
            step_diseases_str = ""
            if target_phenotypes and step_diseases:
                step_diseases_str = self._format_step_diseases_output(step_diseases, target_phenotypes, step_number)
        else:
            # non-final: do not extract diseases
            step_diseases = []
            step_diseases_str = ""
            print(f"  Dialogue {step_number} completed")
        
        # print(self._format_history_as_string())
        enhanced_prompt = ""
        if use_history and addTo_history:
            enhanced_prompt = self._format_history_as_string()
        elif use_history and not addTo_history:
            enhanced_prompt = self._format_history_as_string() + prompt
        else:
            enhanced_prompt = prompt

        return {
            "enhanced_prompt": enhanced_prompt,
            "prediction": step_prediction,
            "thinking": step_thinking,
            "diseases": step_diseases,
            "diseases_str": step_diseases_str,
            "time": step_time,
        }
    
    def _remove_numbering(self, text: str) -> str:
        """
        Remove leading numerals (1., 2., a., b., etc.)
        Args:
            text: input text
        Returns:
            cleaned_text: text after remove
        """
        if not text:
            return text
        
        lines = text.split('\n')
        cleaned_lines = []
        
        for line in lines:
            # remove leading numeral pattern
            # match: 1. 2. a. b. i. ii. etc.
            cleaned_line = re.sub(r'^[\s]*([0-9]+\.|[a-zA-Z]\.|[ivxlcdmIVXLCDM]+\.)[\s]*', '', line.strip())
            
            # keep if non-empty after clean
            if cleaned_line:
                cleaned_lines.append(cleaned_line)
            else:
                # if empty after clean but was non-empty, keep (blank or numeral only)
                if line.strip():
                    cleaned_lines.append(line.strip())
        
        cleaned_text = '\n'.join(cleaned_lines)
        cleaned_text = cleaned_text + "\n"

        return cleaned_text
    
    
    # example:
    # 
    # two-step
    # result = predictor.generate_two_step_prediction(step1_prompt, step2_prompt, target_phenotypes)
    # 
    # three-step
    # result = predictor.generate_three_step_prediction(step1_prompt, step2_prompt, step3_prompt, target_phenotypes)
    # 
    # unified three-step
    # result = predictor.generate_three_step_prediction_v1(step1_prompt, step2_prompt, step3_prompt, target_phenotypes)
    # 
    # extend by adding functions
    # def generate_four_step_prediction(self, step1_prompt, step2_prompt, step3_prompt, step4_prompt, target_phenotypes=None, temperature=None):
    # use _execute_single_step
    #     pass
    
    def extract_disease_names_from_answer(self, answer: str) -> List[str]:
        """Extract disease names from the answer text"""
        # Split by newlines and look for numbered entries
        lines = answer.split('\n')
        disease_names = []
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
            
            # Look for patterns like 
            # "1. DISEASE_NAME" or "**1. DISEASE_NAME**" or "1. DISEASE_NAME - description" or "1. DISEASE_NAME: description"
            # or \"1. DISEASE_NAME\" 
            # Remove numbering and any description after dash
            # print(f"  Extracting disease part from line: {line}")
            if re.match(r'^\d+\.', line) or re.match(r'^\*?\*?\d+\.', line) or re.match(r'^\"\d+\.', line):
                # Remove the number and dot, including any markdown formatting and escaped quotes
                disease_part = re.sub(r'^\"?\*?\*?\d+\.\s*', '', line)
                disease_part = re.sub(r'^Patient \d+\.\s*', '', disease_part)
                disease_part = disease_part.replace('\"', '')
                
                # Remove description after dash if present
                if ' - ' in disease_part:
                    disease_part = disease_part.split(' - ')[0].strip()
                elif ' – ' in disease_part:
                    disease_part = disease_part.split(' – ')[0].strip()
                elif ':' in disease_part:
                    disease_part = disease_part.split(':')[0].strip()
                
                # Remove parentheses and their content (explanations)
                disease_part = re.sub(r'\s*\([^)]*\)', '', disease_part)
                
                # Clean up any remaining formatting
                disease_part = disease_part.replace('**', '').replace('*', '').replace("'s", '').replace('\\"', '').strip()
                
                # Only keep if it looks like a valid disease name
                if disease_part and len(disease_part) > 2:
                    disease_names.append(disease_part)
        
        return disease_names
    
    def validate_prediction_steps(self, prediction: str, required_steps: List[str] = None) -> Dict[str, any]:
        """
        Validate if prediction contains required steps
        
        Args:
            prediction: The prediction text to validate
            required_steps: List of required step keywords (default: ["Step 1:", "Step 2:", "Step 3:", "Step 4:"])
            
        Returns:
            Dict containing validation results:
            - is_valid: bool
            - missing_steps: List[str]
            - found_steps: List[str]
        """
        if required_steps is None:
            # required_steps = ["Step 1:", "Step 2:", "Step 3:", "Step 4:"]
            required_steps = ["Step 1", "Step 2", "Step 3", "Step 4", "FINAL ANSWER"]
        
        missing_steps = []
        found_steps = []
        
        for step in required_steps:
            if step in prediction:
                found_steps.append(step)
            else:
                missing_steps.append(step)
        
        return {
            "is_valid": len(missing_steps) == 0,
            "missing_steps": missing_steps,
            "found_steps": found_steps
        }
    
    def extract_candidate_diseases_json(self, content: str) -> List[str]:
        """
        Extract disease names from JSON format '20 Candidate Rare Diseases' in content
        Supports both array format and object format (disease names as keys)
        
        Args:
            content: String containing JSON format with "20 Candidate Rare Diseases"
            
        Returns:
            List[str]: List of disease names extracted from the JSON
        """
        try:
            # First try to match object format (disease names as keys with symptoms as values)
            object_patterns = [
                r'"20\s+Candidate\s+Rare\s+Diseases":\s*\{([^}]+)\}',
                r'"20\s+Candidate\s+Diseases":\s*\{([^}]+)\}',
                r'"Candidate\s+Rare\s+Diseases":\s*\{([^}]+)\}',
                r'"Candidate\s+Diseases":\s*\{([^}]+)\}',
                r'20\s+Candidate\s+Rare\s+Diseases.*?:\s*\{([^}]+)\}',
                r'20\s+Candidate\s+Diseases.*?:\s*\{([^}]+)\}'
            ]
            
            for pattern in object_patterns:
                try:
                    matches = re.findall(pattern, content, re.IGNORECASE | re.DOTALL)
                    if matches:
                        object_content = matches[0].strip()
                        # Extract disease names (keys) from the object
                        # Pattern to match quoted disease names followed by colon
                        disease_pattern = r'"([^"]+)"\s*:'
                        disease_matches = re.findall(disease_pattern, object_content)
                        if disease_matches:
                            cleaned_diseases = []
                            for disease in disease_matches:
                                cleaned_disease = disease.strip()
                                if cleaned_disease:
                                    # Remove leading numbers and dots (e.g., "1. Disease Name" -> "Disease Name")
                                    cleaned_disease = re.sub(r'^\d+\.\s*', '', cleaned_disease)
                                    cleaned_diseases.append(cleaned_disease)
                            # print(f"Successfully extracted {len(cleaned_diseases)} diseases from JSON object format")
                            return cleaned_diseases
                except Exception as e:
                    print(f"Warning: Error extracting with object pattern {pattern}: {e}")
                    continue
            
            # Fallback: try to match array format for "20 Candidate Rare Diseases"
            array_patterns = [
                r'"20\s+Candidate\s+Rare\s+Diseases":\s*\[(.*?)\]',
                r'"20\s+Candidate\s+Diseases":\s*\[(.*?)\]',
                r'"Candidate\s+Rare\s+Diseases":\s*\[(.*?)\]',
                r'"Candidate\s+Diseases":\s*\[(.*?)\]',
                r'20\s+Candidate\s+Rare\s+Diseases.*?:\s*\[(.*?)\]',
                r'20\s+Candidate\s+Diseases.*?:\s*\[(.*?)\]'
            ]
            
            for pattern in array_patterns:
                try:
                    matches = re.findall(pattern, content, re.IGNORECASE | re.DOTALL)
                    if matches:
                        array_content = matches[0].strip()
                        # print(f"Found JSON array content: {array_content[:200]}...")
                        
                        # Try to parse as JSON array
                        try:
                            # Wrap the array content in brackets to make it a valid JSON array
                            json_str = f"[{array_content}]"
                            diseases = json.loads(json_str)
                            if isinstance(diseases, list):
                                # Clean up disease names (remove quotes, strip whitespace, remove numbers)
                                cleaned_diseases = []
                                for disease in diseases:
                                    if isinstance(disease, str):
                                        cleaned_disease = disease.strip().strip('"').strip("'")
                                        if cleaned_disease:
                                            # Remove leading numbers and dots (e.g., "1. Disease Name" -> "Disease Name")
                                            cleaned_disease = re.sub(r'^\d+\.\s*', '', cleaned_disease)
                                            cleaned_diseases.append(cleaned_disease)
                                
                                # print(f"Successfully extracted {len(cleaned_diseases)} diseases from JSON array: {cleaned_diseases}")
                                return cleaned_diseases
                        except json.JSONDecodeError as json_err:
                            print(f"JSON decode error: {json_err}")
                            # Fallback: try to extract individual quoted strings
                            quoted_pattern = r'"([^"]+)"'
                            quoted_matches = re.findall(quoted_pattern, array_content)
                            if quoted_matches:
                                cleaned_diseases = []
                                for match in quoted_matches:
                                    cleaned_match = match.strip()
                                    if cleaned_match:
                                        # Remove leading numbers and dots (e.g., "1. Disease Name" -> "Disease Name")
                                        cleaned_match = re.sub(r'^\d+\.\s*', '', cleaned_match)
                                        cleaned_diseases.append(cleaned_match)
                                # print(f"Fallback: Extracted {len(cleaned_diseases)} diseases using regex")
                                return cleaned_diseases
                            
                except Exception as e:
                    print(f"Warning: Error extracting with array pattern {pattern}: {e}")
                    continue
            
            print("    Warning: No JSON format '20 Candidate Rare Diseases' found")
            return []
            
        except Exception as e:
            print(f"Error in extract_candidate_diseases_json: {e}")
            return []

    def extract_disease_names_from_prediction(self, prediction: str) -> List[str]:
        """Extract disease names from model prediction"""
        try:
            # First try to extract from numbered list format in FINAL ANSWER section
            if "FINAL ANSWER" in prediction or "Final Answer" in prediction:
                # Extract content after the last "FINAL ANSWER:"
                try:
                    if "FINAL ANSWER" in prediction:
                        # Split by "FINAL ANSWER" and take the last part
                        parts = prediction.split("FINAL ANSWER")
                        final_answer_part = parts[-1].strip()
                    else:
                        # Split by "Final Answer" and take the last part
                        parts = prediction.split("Final Answer")
                        final_answer_part = parts[-1].strip()
                    disease_names = self.extract_disease_names_from_answer(final_answer_part)
                    if disease_names:
                        return disease_names
                except (IndexError, Exception) as e:
                    print(f"Warning: Error extracting from FINAL ANSWER section: {e}")
            
            # Fallback: try to extract from numbered list format anywhere
            # try:
            #     disease_names = self.extract_disease_names_from_answer(prediction)
            #     if disease_names:
            #         return disease_names
            # except Exception as e:
            #     print(f"Warning: Error extracting from numbered list: {e}")
            
            # # If no numbered list found, try to extract disease names from text
            # # Look for patterns like "Disease Name" or disease-like names
            # disease_patterns = [
            #     r'\b([A-Z][a-zA-Z\s\-]+(?:syndrome|disease|disorder|condition))\b',  # Extract disease-like names
            #     r'\b([A-Z][a-zA-Z\s\-]+)\b',  # Extract capitalized names
            # ]
            
            # for pattern in disease_patterns:
            #     try:
            #         matches = re.findall(pattern, prediction)
            #         if matches:
            #             # Filter out non-disease-like matches
            #             filtered_matches = []
            #             for match in matches:
            #                 # Skip if it's too short or doesn't look like a disease name
            #                 if len(match) >= 3 and not match.isdigit():
            #                     filtered_matches.append(match.strip())
                        
            #             if filtered_matches:
            #                 return filtered_matches[:10]  # Limit to first 10 matches
            #     except Exception as e:
            #         print(f"Warning: Error applying pattern {pattern}: {e}")
            #         continue
            
            return []
            
        except Exception as e:
            print(f"Error in extract_disease_names_from_prediction: {e}")
            return []
    
    def _format_step_diseases_output(self, diseases: List[str], target_phenotypes: List[str], step_number: int) -> str:
        """
        Generic disease output formatter, arbitrary step count
        Args:
            diseases: extracted disease list
            target_phenotypes: target phenotype list
            step_number: step number
        Returns:
            formatted_output: formatted disease list string
        """
        print(f"    INFO:_format_step{step_number}_diseases_output: Processing {len(diseases)} diseases")
        # print(diseases)
        # print(f"  Target phenotypes: {target_phenotypes}")
        
        if not diseases:
            return f"No diseases identified in Step {step_number}."
        
        formatted_lines = []
        
        start_idx = 51
        list_phenotypes = False

        for i, disease_name in enumerate(diseases, start_idx):
            # print(f"  Processing disease {i}: {disease_name}")
            
            try:
                # two-phase: fast then semantic
                matched_disease_info = self._find_best_matching_disease_two_stage(disease_name)

                if matched_disease_info["disease_id"] == "UNKNOWN":
                    continue
                
                # get phenotype match for disease
                target_phenotypes_id = [phenotype.split('(')[1].split(')')[0].strip() for phenotype in target_phenotypes]
                phenotype_matches = self._get_disease_phenotype_matches(matched_disease_info, target_phenotypes_id)
                # print(f"    Phenotype matches: {phenotype_matches}")
                
                # build formatted output line
                # if not phenotype_matches:
                #     continue
                
                formatted_line = self._build_disease_line(i, disease_name, matched_disease_info, phenotype_matches, list_phenotypes)
                formatted_lines.append(formatted_line)
                # print(f"    Formatted line: {formatted_line}")
                
            except Exception as e:
                print(f"    Error processing disease {disease_name}: {e}")
                # add simple fallback
                formatted_line = f"{i}. [{disease_name}] (Error processing)"
                formatted_lines.append(formatted_line)
        
        result = "\n".join(formatted_lines)
        result = result + "\n"
        return result
    
    # deprecated, kept as backup
    def _find_best_matching_disease_two_stage_deprecated(self, disease_name: str) -> Dict:
        """
        Two-phase disease match from HPOA only
        Args:
            disease_name: disease name
        Returns:
            disease_info: matched disease info
        """
        # print(f"      _find_best_matching_disease_two_stage: Searching for '{disease_name}'")
        
        # phase 1: fast match (HPOA only)
        candidates = self._fast_filter_key_parts_from_hpoa(disease_name)
        # print(f"      Stage 1: Found {len(candidates)} candidates from HPOA database")
        
        if not candidates:
            return {
                "disease_id": "UNKNOWN",
                "standard_name": disease_name,
                "similarity": 0.0
            }
        
        # phase 2: semantic match among candidates
        best_match = None
        best_similarity = 0.0
        
        for candidate in candidates:
            try:
                # split candidate names from DB only
                # candidates may be aliases, differ from standard
                candidate_names = self._split_disease_names(candidate['matched_names'])
                
                # compute similarity, take max
                max_similarity = 0.0
                best_candidate_name = candidate['standard_name']
                
                for candidate_name in candidate_names:
                    # candidate lowercased, so lower disease_name
                    disease_name = disease_name.lower()
                    similarity = self.calculate_semantic_similarity(disease_name, candidate_name)
                    if similarity > max_similarity:
                        max_similarity = similarity
                        best_candidate_name = candidate_name
                
                if max_similarity > best_similarity:
                    best_similarity = max_similarity
                    best_match = candidate.copy()
                    best_match['similarity'] = max_similarity
                    best_match['matched_candidate_name'] = best_candidate_name

            except Exception as e:
                print(f"      Error calculating similarity for {candidate['disease_id']}: {e}")
                continue
        
        # print(f"      Stage 2: best match: {best_match['disease_id']} ({best_similarity:.3f}) - matched to: '{best_match['matched_candidate_name']}'")
        
        # if similarity too low, return original
        similarity_threshold = self.config.get('evaluation_config', {}).get('similarity_threshold', 0.8)
        if not best_match or best_match["similarity"] < similarity_threshold:
            return {
                "disease_id": "UNKNOWN",
                "standard_name": disease_name,
                "similarity": best_match["similarity"] if best_match else 0.0
            }
        
        return best_match
    
    def _precompute_disease_embeddings(self):
        """
        Pre-encode embeddings for disease IDs, standard names and synonyms
        Key=disease ID, value=embedding list (multiple names/synonyms per disease)
        """
        if self._disease_names_embeddings_cache is not None:
            return  # already encoded
        
        print("Precomputing disease name embeddings...")
        
        # init cache
        self._disease_names_embeddings_cache = {}
        self._disease_names_cache = {}
        
        # collect names to encode
        all_disease_names = []  # processed for embedding
        all_original_disease_names = []  # raw for cache
        disease_id_to_indices = defaultdict(list)  # id -> indices
        
        # get unique disease IDs from disease_name_to_ids
        # align with phenotype_disease_case_database (20268 unique)
        all_disease_ids = set()
        for disease_ids_set in self.disease_name_to_ids.values():
            all_disease_ids.update(disease_ids_set)
        
        disease_ids = list(all_disease_ids)
        
        # disease_ids_from_name_to_ids = len(set().union(*self.disease_name_to_ids.values())) if self.disease_name_to_ids else 0
        
        print(f"Precomputing diseases numbers: {len(disease_ids)}")
        # print(f"  - From disease_name_to_ids: {disease_ids_from_name_to_ids}")
        # print(f"  - Expected from database: 20268")

        for disease_id in disease_ids:
            # get standard name
            standard_name = self.disease_mapping.get(disease_id, disease_id)
            if standard_name and standard_name != disease_id:
                # process: remove syndrome/disease suffix
                processed_name = self._preprocess_disease_name_for_embedding(standard_name)
                if processed_name:
                    idx = len(all_disease_names)
                    all_disease_names.append(processed_name)
                    all_original_disease_names.append(standard_name)  # save original
                    disease_id_to_indices[disease_id].append(idx)
            
            # get synonyms
            synonyms = self.disease_mapping_with_synonyms.get(disease_id, [])
            for synonym in synonyms:
                if synonym:
                    processed_synonym = self._preprocess_disease_name_for_embedding(synonym)
                    if processed_synonym:
                        idx = len(all_disease_names)
                        all_disease_names.append(processed_synonym)
                        all_original_disease_names.append(synonym)  # save original
                        disease_id_to_indices[disease_id].append(idx)
        
        if not all_disease_names:
            print("Warning: No disease names to encode")
            return
        
        # batch encode disease names
        print(f"Encoding {len(all_disease_names)} disease names in batches...")
        batch_size = 128  # batch size, adjust by GPU
        all_embeddings = []
        
        for i in range(0, len(all_disease_names), batch_size):
            batch = all_disease_names[i:i + batch_size]
            try:
                batch_embeddings = self.sentence_model.encode(batch, show_progress_bar=False, convert_to_numpy=True)
                all_embeddings.extend(batch_embeddings)
            except Exception as e:
                print(f"Error encoding batch {i//batch_size + 1}: {e}")
                # if batch fails, encode one by one
                for name in batch:
                    try:
                        embedding = self.sentence_model.encode([name], show_progress_bar=False, convert_to_numpy=True)[0]
                        all_embeddings.append(embedding)
                    except Exception as e2:
                        print(f"Error encoding '{name}': {e2}")
                        all_embeddings.append(None)
        
        # organize embedding by disease ID, normalize
        for disease_id, indices in disease_id_to_indices.items():
            embeddings_list = []
            names_list = []
            for idx in indices:
                if idx < len(all_embeddings) and all_embeddings[idx] is not None:
                    # normalize for cosine via dot product
                    embedding = all_embeddings[idx]
                    norm = np.linalg.norm(embedding)
                    if norm > 0:
                        embedding = embedding / norm
                    embeddings_list.append(embedding)
                    names_list.append(all_original_disease_names[idx])  # use original name
            
            if embeddings_list:
                self._disease_names_embeddings_cache[disease_id] = np.array(embeddings_list)
                self._disease_names_cache[disease_id] = names_list
        
        print(f"Precomputed embeddings for {len(self._disease_names_embeddings_cache)} diseases")
    
    def _preprocess_disease_name_for_embedding(self, name: str) -> str:
        """
        Preprocess disease name for embedding (remove suffix, lower)
        """
        if not name:
            return ""
        
        name = name.lower().strip()
        
        # remove common suffixes (low help for match)
        if name.endswith('syndrome'):
            name = name[:-9].strip()
        elif name.endswith('disease'):
            name = name[:-7].strip()
        elif name.endswith('disorder'):
            name = name[:-8].strip()
        
        return name
    
    def _find_best_matching_disease_two_stage(self, disease_name: str) -> Dict:
        """
        Faster: use pre-encoded embeddings
        Two-phase disease match:
        1. pre-encode all disease names and synonyms
        2. encode query, batch similarity to pre-encoded
        
        Args:
            disease_name: disease name
        Returns:
            disease_info: matched disease info
        """
        # ensure pre-encoded
        if self._disease_names_embeddings_cache is None:
            self._precompute_disease_embeddings()
        
        if not self._disease_names_embeddings_cache:
            # if pre-encode fails, fallback to legacy
            return self._find_best_matching_disease_two_stage_deprecated(disease_name)
        
        # preprocess query name
        query_name_processed = self._preprocess_disease_name_for_embedding(disease_name)
        if not query_name_processed:
            return {
                "disease_id": "UNKNOWN",
                "standard_name": disease_name,
                "similarity": 0.0
            }
        
        # embed query name
        try:
            query_embedding = self.sentence_model.encode([query_name_processed], show_progress_bar=False, convert_to_numpy=True)[0]
            query_embedding = query_embedding / np.linalg.norm(query_embedding)  # normalize
        except Exception as e:
            print(f"Error encoding query disease name '{disease_name}': {e}")
            return {
                "disease_id": "UNKNOWN",
                "standard_name": disease_name,
                "similarity": 0.0
            }
        
        # batch similarity to all diseases
        best_match = None
        best_similarity = 0.0
        best_matched_name = ""
        
        for disease_id, embeddings in self._disease_names_embeddings_cache.items():
            # max similarity over names/synonyms for disease
            # embeddings: (n_names, dim)
            similarities = np.dot(embeddings, query_embedding)  # dot=cosine if normalized
            max_sim_idx = np.argmax(similarities)
            max_similarity = float(similarities[max_sim_idx])
            
            if max_similarity > best_similarity:
                best_similarity = max_similarity
                best_match = disease_id
                best_matched_name = self._disease_names_cache[disease_id][max_sim_idx]
        
        # if similarity too low, return original
        similarity_threshold = self.config.get('evaluation_config', {}).get('similarity_threshold', 0.8)
        if not best_match or best_similarity < similarity_threshold:
            return {
                "disease_id": "UNKNOWN",
                "standard_name": disease_name,
                "similarity": best_similarity if best_match else 0.0
            }
        
        # return match
        standard_name = self.disease_mapping.get(best_match, best_match)
        return {
            "disease_id": best_match,
            "standard_name": standard_name,
            "similarity": best_similarity,
            "matched_candidate_name": best_matched_name
        }
    
    def _split_disease_names(self, disease_names: List[str]) -> List[str]:
        """
        Split disease names by ';' into list
        Args:
            disease_names: list, may contain ';'
        Returns:
            disease_names_list: split list
        """
        disease_names_list = []
        for disease_name in disease_names:

            if ';' not in disease_name: 
                disease_names_list.append(disease_name.strip())
                continue
            
            # split by ';' and clean
            parts = disease_name.split(';')
            
            for part in parts:
                part = part.strip()
                if part and part not in disease_names_list:  # non-empty only
                    disease_names_list.append(part)
        
        return disease_names_list
    
    def _split_single_disease_name(self, disease_name: str) -> List[str]:
        """
        Split single disease name by ';' into list
        Args:
            disease_name: may contain ';'
        Returns:
            disease_names: split list
        """
        if ';' not in disease_name:
            return [disease_name.strip()]
        
        # split by ';'
        semicolon_split = [d.strip() for d in disease_name.split(';') if d.strip()]
        return semicolon_split
    
    def _fast_filter_key_parts_from_hpoa(self, disease_name: str) -> List[Dict]:
        """
        Fast filter candidates by key parts (HPOA only)
        Args:
            disease_name: disease name
        Returns:
            candidates: candidate list
        """
        candidates = []
        disease_name_lower = disease_name.lower()
        
        # extract key parts (remove common), rough, may optimize
        key_parts = self._extract_key_parts(disease_name_lower)
        
        # set max candidates
        max_candidates = 500
        count = 0
        
        # disease_to_phenotypes has final merged assoc
        if self.disease_to_phenotypes:
            hpoa_diseases = set(self.disease_to_phenotypes.keys())
            
            for disease_id in hpoa_diseases:
                if count >= max_candidates:
                    break
                
                # get standard name
                standard_name = self.disease_mapping.get(disease_id, disease_id)
                standard_name_lower = standard_name.lower()
                
                synonyms = self.disease_mapping_with_synonyms.get(disease_id, [])
                synonyms_lower = [synonym.lower() for synonym in synonyms]
                
                # merge standard and synonyms
                all_names_lower = [standard_name_lower] + synonyms_lower
                
                # check key part match
                # prediction may be alias, added alias match
                matched_names = self._check_key_parts_match(key_parts, all_names_lower)
                
                if matched_names:
                    candidates.append({
                        "disease_id": disease_id,
                        "standard_name": standard_name,
                        "matched_names": matched_names
                    })
                    count += 1
        else:
            print(f"        Warning: HPOA database not available")
        
        # print(candidates)
        return candidates
    
    def _extract_key_parts(self, disease_name: str) -> List[str]:
        """
        Extract key parts from disease name, remove common
        Args:
            disease_name: lowercased
        Returns:
            key_parts: key parts list
        """
        # common words to skip in match
        common_words = {
            'syndrome', 'disease', 'disorder', 'condition', 'anomaly', 'defect',
            'malformation', 'abnormality', 'deficiency', 'insufficiency',
            'type', 'form', 'variant', 'subtype', 'class', 'category',
            'autosomal', 'dominant', 'recessive', 'x-linked', 'y-linked',
            'inherited', 'genetic', 'congenital', 'hereditary', 'familial',
            'rare', 'common', 'frequent', 'occasional', 'very', 'extremely', 
            'with', 'of', 'and', 'or',
        }
        
        # split name
        words = disease_name.split()
        
        # filter key parts
        key_parts = []
        for word in words:
            # remove punctuation
            clean_word = word.strip('.,;:!?()[]{}"\'-')
            # keep non-common, len>2
            if clean_word and len(clean_word) > 2 and clean_word not in common_words:
                key_parts.append(clean_word)
        
        return key_parts
    
    def _check_key_parts_match(self, key_parts: List[str], all_names_lower: List[str]) -> bool:
        """
        Check if key parts match
        Args:
            key_parts: key parts list
            standard_name: standard or alias (lower)
        Returns:
            is_match: whether match
        """
        if not key_parts:
            return False
        
        # match half or one key part len>3
        min_matches = max(1, len(key_parts) // 2)
        # if few key parts, stricter
        if len(key_parts) <= 2:
            min_matches = len(key_parts)

        # count matched key parts
        matched_names = []
        for all_name_lower in all_names_lower:
            matched_parts = 0
            for key_part in key_parts:
                if key_part in all_name_lower:
                    matched_parts += 1

            if matched_parts >= min_matches:
                matched_names.append(all_name_lower)
        
        return matched_names
            
    def rank_diseases_by_phenotype_associations(self, phenotypes: List[str], use_frequency_weights: bool = False, use_IC_weights: bool = False, disease_examples: List[str] = None) -> List[Dict]:
        """
        Rank diseases based on their association count with the given phenotypes.
        
        Args:
            phenotypes: List of phenotype IDs (HPO terms)
            use_frequency_weights: Whether to use frequency weights to rank diseases
            use_IC_weights: Whether to use IC weights to rank diseases
            disease_examples: List of disease IDs that are associated with the given phenotypes
            
        Returns:
            List of dictionaries containing disease info and ranking scores, sorted by score
        """
        
        disease_scores = defaultdict(int)
        disease_info = {}
        
        # get all parent phenotypes (including grandparents, great-grandparents, etc.) for each phenotype
        parents_phenotypes = {}
        for phenotype in phenotypes:
            all_parents = self.get_all_parent_phenotypes(phenotype)
            parents_phenotypes[phenotype] = list(all_parents)

        # print("disease_examples", disease_examples, "phenotypes", phenotypes)

        # Calculate scores for each disease based on phenotype associations
        for phenotype_id in phenotypes:
            # Get diseases associated with this phenotype
            associated_diseases = disease_examples
            
            for disease_id in associated_diseases:
                # Store disease info if not already stored
                if disease_id not in disease_info:
                    
                    # Calculate matching phenotypes for this disease
                    matching_phenotypes = list(self.disease_to_phenotypes[disease_id].intersection(set(phenotypes)))
                    # print("matching_phenotypes", matching_phenotypes)
                    if not matching_phenotypes:
                        continue

                    # Check for phenotype matches through parent relationships
                    matching_phenotypes_by_parents = []
                    matching_phenotypes_by_parents_map = {}

                    parents_phenotypes_in_disease = set()  # Use set for efficient lookup
                    
                    # Collect all parent phenotypes for this disease
                    disease_phenotypes = self.disease_to_phenotypes.get(disease_id, [])
                    for phenotype in disease_phenotypes:
                        phenotype_parents = self.hpo_is_a.get(phenotype, [])
                        parents_phenotypes_in_disease.update(phenotype_parents)
                    
                    # Check if any patient phenotype's parents match disease phenotype parents
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
                    # dedup by parent notes in matching_phenotypes_by_parents
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
                                
                                # no freq filter for predicted
                                # if frequency_display < 0.5:
                                #     # skip low-freq phenotype
                                #     continue

                                phenotype_with_freq = f"{phenotype_id_original} ({frequency_display})"
                                matching_phenotypes_with_freq.append(phenotype_with_freq)
                                matching_phenotypes_freq_info.append(frequency_display)
                                filtered_matching_phenotypes.append(phenotype_id_original)
                            else:
                                # skip no-freq phenotype
                                continue
                            
                                phenotype_with_freq = phenotype_id_original
                                matching_phenotypes_with_freq.append(phenotype_with_freq)
                                matching_phenotypes_freq_info.append(0.0)
                                filtered_matching_phenotypes.append(phenotype_id_original)

                        else:
                            print(f"Error: No frequency information for {phenotype_id_original}")
                            # skip no-freq phenotype
                            continue
                        
                            phenotype_with_freq = phenotype_id_original
                            matching_phenotypes_with_freq.append(phenotype_with_freq)
                            matching_phenotypes_freq_info.append(0.0)
                            filtered_matching_phenotypes.append(phenotype_id_original)
                    
                    # update matching_phenotypes after filter
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
                    
                    # Calculate total phenotype associations
                    total_phenotype_associations = self.disease_phenotype_counts[disease_id]
                    
                    disease_id_with_same_name = set()
                    for disease_name, disease_ids in self.disease_name_to_ids.items():
                        if disease_id in disease_ids:
                            disease_id_with_same_name.update(disease_ids)

                    disease_info[disease_id] = {
                        'disease_id': disease_id,
                        'disease_name': self.get_disease_name(disease_id),
                        'disease_id_with_same_name': list(disease_id_with_same_name),
                        'disease_synonyms': list(self.disease_mapping_with_synonyms[disease_id]),   
                        'total_phenotype_associations': total_phenotype_associations,
                        'matching_phenotypes': matching_phenotypes_with_freq,
                        'matching_phenotype_count': matching_count,
                        'matching_phenotype_count_weighted': matching_count_weighted,
                    }
                    
                    # Use matching count as score for ranking
                    disease_scores[disease_id] = matching_count
        
        # Convert to list and sort by matching count (descending)
        ranked_diseases = []
        for disease_id, matching_count in disease_scores.items():
            disease_data = disease_info[disease_id].copy()
            ranked_diseases.append(disease_data)
        
        # # dedup ranked_diseases by name
        # disease_name_to_id = {}  # name->id
        unique_diseases = []  # unique disease IDs

        # dedup ranked_diseases by disease_id_with_same_name
        processed_disease_ids = set()  # processed IDs
        
        for disease_info in ranked_diseases:
            disease_id_with_same_name = set(disease_info['disease_id_with_same_name'])
            
            # check duplicate (incl same-name)
            has_duplicate = False
            for processed_id in processed_disease_ids:
                if processed_id in disease_id_with_same_name:
                    # if duplicate, keep first
                    has_duplicate = True
                    break
            
            if not has_duplicate:
                # if no duplicate, add
                unique_diseases.append(disease_info)
                # mark same-name IDs processed
                processed_disease_ids.update(disease_id_with_same_name)
        
        ranked_diseases = unique_diseases

        # Sort by matching phenotype count (descending), then by total phenotype associations (ascending - fewer is better), then by disease_id for stability
        # Note: total_phenotype_associations is sorted in ascending order because fewer associations means more specific disease
        # ranked_diseases.sort(key=lambda x: (x['matching_phenotype_count'], -x['total_phenotype_associations'], x['disease_id']), reverse=True)
        
        # TEST
        # if use_IC_weights and use_frequency_weights:
        #     use_frequency_weights = False

        # if use_IC_weights:
        #     ranked_diseases.sort(key=lambda x: (x['matching_phenotype_IC'], x['matching_phenotype_count'], -x['total_phenotype_associations']), reverse=True)
        # elif use_frequency_weights:
        #     ranked_diseases.sort(key=lambda x: (x['matching_phenotype_count_weighted'], x['matching_phenotype_count'], -x['total_phenotype_associations']), reverse=True)
        # else:
        #     ranked_diseases.sort(key=lambda x: (x['matching_phenotype_count'], x['matching_phenotype_IC'], -x['total_phenotype_associations']), reverse=True)


        return ranked_diseases

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

    def _get_disease_phenotype_matches(self, disease_info: Dict, target_phenotypes: List[str]) -> List[Dict]:
        """
        Get disease-phenotype match for target
        Uses rank_diseases_by_phenotype_associations
        Args:
            disease_info: disease info
            target_phenotypes: target phenotype list
        Returns:
            matches: list of phenotype matches
        """
        matches = []
        
        disease_id = disease_info.get('disease_id')
        if not disease_id or disease_id == 'UNKNOWN':
            return matches
        
        # use rank_diseases... for assoc
        try:
            # call with disease_examples=target id
            ranked_diseases = self.rank_diseases_by_phenotype_associations(
                phenotypes=target_phenotypes,
                disease_examples=[disease_id]
            )
            
            # extract matches from result
            if ranked_diseases and len(ranked_diseases) > 0:
                disease_data = ranked_diseases[0]  # first (only) disease
                matching_phenotypes = disease_data.get('matching_phenotypes', [])
                
                # convert to expected format
                for phenotype_info in matching_phenotypes:
                    # parse phenotype, may have freq
                    if '(' in phenotype_info and ')' in phenotype_info:
                        # format: HP:xxx (0.8) or (by parent)
                        phenotype_id = phenotype_info.split('(')[0].strip()
                        frequency_info = phenotype_info.split('(')[1].split(')')[0].strip()
                        
                        # determine match type
                        if 'by parent' in frequency_info:
                            qualifier = 'by_parent'
                            frequency_type = 'parent_match'
                            frequency = "by parent"
                        elif 'by child' in frequency_info:
                            qualifier = 'by_child'
                            frequency_type = 'child_match'
                            frequency = "by child"
                        else:
                            # numeric freq
                            qualifier = ''
                            frequency_type = 'frequency'
                            frequency = frequency_info
                    else:
                        # no-freq phenotype
                        phenotype_id = phenotype_info.strip()
                        qualifier = ''
                        frequency_type = 'direct_match'
                        frequency = ''
                    
                    matches.append({
                        'phenotype': phenotype_id,
                        'hp_code': phenotype_id,
                        'frequency': frequency,
                        'qualifier': qualifier,
                        'frequency_type': frequency_type
                    })
        
        except Exception as e:
            print(f"Error in _get_disease_phenotype_matches: {e}")
            # on error return []
            return []
        
        return matches
    
    def _analyze_step_disease_mappings(self, stepN_diseases: List[str], n: int, true_diseases: List[str], target_phenotypes: List[str]) -> List[Dict]:
        """
        Analyze mapping: step N predicted vs true diseases
        Args:
            stepN_diseases: step N predicted list
            true_diseases: true disease list
            target_phenotypes: target patient phenotypes
        Returns:
            mappings: mapping list
        """
        mappings = []
        
        for i, predicted_disease in enumerate(stepN_diseases, 1):
            # print(f"    Analyzing step {n} disease {i}: {predicted_disease}")
            
            # find best matching true disease
            best_match_info = self._find_best_matching_disease_two_stage(predicted_disease)
            
            # get phenotype match for disease
            # target_phenotypes =  ['Compulsive behaviors (HP:0000722)', 'Delayed speech and language development (HP:0000750)']
            target_phenotypes_id = [phenotype.split('(')[1].split(')')[0].strip() for phenotype in target_phenotypes]
            phenotype_matches = self._get_disease_phenotype_matches(best_match_info, target_phenotypes_id)
            
            # compute total phenotype assoc
            total_phenotype_associations = 0
            if best_match_info.get('disease_id') in self.disease_to_phenotypes:
                total_phenotype_associations = len(self.disease_to_phenotypes[best_match_info['disease_id']])
            
            # build mapping info
            mapping_info = {
                "predicted_disease_name": predicted_disease,  # add predicted name
                "disease_id": best_match_info.get('disease_id', 'UNKNOWN'),
                "disease_name_standard": best_match_info.get('standard_name', 'Unknown'),
                "disease_name_matched": best_match_info.get('matched_candidate_name', []),
                "similarity": best_match_info.get('similarity', 0.0),  # add similarity
                "total_phenotype_associations": total_phenotype_associations,
                "matching_phenotypes": [],
                "matching_phenotype_count": len(phenotype_matches),
                "method": "predict"  # use predict
            }
            
            # add matching phenotype info
            for match in phenotype_matches:
                hp_code = match['hp_code']
                frequency = match['frequency']
                
                # if freq is HP code, get its description
                if 'HP:' in frequency:
                    frequency_display = self._convert_hpo_frequency_to_description(frequency)
                else:
                    frequency_display = frequency
                
                # add parens only when freq non-empty
                if frequency_display and frequency_display.strip():
                    mapping_info["matching_phenotypes"].append(f"{hp_code} ({frequency_display})")
                else:
                    mapping_info["matching_phenotypes"].append(hp_code)
            
            mappings.append(mapping_info)
            # print(f"      Mapped to: {mapping_info['disease_name']} (ID: {mapping_info['disease_id']})")
            # print(f"      Total phenotypes: {total_phenotype_associations}, Matching: {len(phenotype_matches)}")
        
        return mappings
    
    def _evaluate_step_accuracy(self, step_diseases: List[str], true_diseases: List[str]) -> Dict:
        """
        Evaluate single-step accuracy
        Args:
            step_diseases: step predicted disease list
            true_diseases: true disease list
        Returns:
            evaluation: evaluation result dict
        """
        if not step_diseases:
            return {
                "correct": False,
                "accuracy": 0.0,
                "best_match": None,
                "best_rank": -1,
                "best_similarity": 0.0,
                "matched_true_disease": None,
                "top1_correct": False,
                "top5_correct": False
            }
        
        # use evaluate_prediction_accuracy (has top1/top5 logic)
        evaluation = self.evaluate_prediction_accuracy(true_diseases, step_diseases)
        
        return evaluation
    
    def _calculate_step_statistics(self, results: List[Dict], step_evaluation_key: str) -> Dict:
        """
        Calculate single-step statistics
        Args:
            results: all results list
            step_evaluation_key: step evaluation key
        Returns:
            stats: statistics dict
        """
        # filter results with this step evaluation
        step_results = [r for r in results if r.get("multi_step_info") and step_evaluation_key in r["multi_step_info"]]
        
        if not step_results:
            return {
                "total": 0,
                "top1_correct": 0,
                "top5_correct": 0,
                "top10_correct": 0,
                "total_correct": 0,
                "top1_accuracy": 0.0,
                "top5_accuracy": 0.0,
                "top10_accuracy": 0.0,
                "total_accuracy": 0.0
            }
        
        total_count = len(step_results)
        top1_correct_count = sum(1 for r in step_results if r["multi_step_info"][step_evaluation_key].get("top1_correct", False))
        top5_correct_count = sum(1 for r in step_results if r["multi_step_info"][step_evaluation_key].get("top5_correct", False))
        top10_correct_count = sum(1 for r in step_results if r["multi_step_info"][step_evaluation_key].get("top10_correct", False))
        total_correct_count = sum(1 for r in step_results if r["multi_step_info"][step_evaluation_key].get("correct", False))
        
        return {
            "total": total_count,
            "top1_correct": top1_correct_count,
            "top5_correct": top5_correct_count,
            "top10_correct": top10_correct_count,
            "total_correct": total_correct_count,
            "top1_accuracy": top1_correct_count / total_count if total_count > 0 else 0.0,
            "top5_accuracy": top5_correct_count / total_count if total_count > 0 else 0.0,
            "top10_accuracy": top10_correct_count / total_count if total_count > 0 else 0.0,
            "total_accuracy": total_correct_count / total_count if total_count > 0 else 0.0
        }
    
    def _identify_steps_for_statistics(self, results: List[Dict]) -> List[int]:
        """
        Auto-detect steps to include in statistics
        Based on whether stepXX_disease_mappings in multi_step_info is non-empty
        
        Args:
            results: all results list
            
        Returns:
            steps: step numbers to include, e.g. [1,2] or [1,2,3]
        """
        steps_with_data = set()
        
        for result in results:
            multi_step_info = result.get("multi_step_info")
            if not multi_step_info:
                continue
                
            # check each possible step
            for step_num in range(1, 20):  # support up to 19 steps
                step_mappings_key = f"step{step_num}_disease_mappings"
                
                # if step has non-empty disease_mappings
                if step_mappings_key in multi_step_info:
                    step_mappings = multi_step_info[step_mappings_key]
                    if step_mappings and len(step_mappings) > 0:
                        steps_with_data.add(step_num)
        
        # return sorted step list
        return sorted(list(steps_with_data))
    
    def word2freq(self, word):
        """Convert word to frequency weight"""
        if self.prompt_generator.word2freq_dict is None:
            self.prompt_generator.word2freq_dict = {
                'very rare': 0.025, 'rare': 0.05, 'occasional': 0.17,
                'frequent': 0.545, 'very frequent': 0.895, 'obligate': 1.0
            }
        word2freq_dict = self.prompt_generator.word2freq_dict
        word = word.lower()
        if word in word2freq_dict:
            return word2freq_dict[word]

        match_obj = re.match(r'^(\d+)/(\d+)$', word)
        if match_obj:
            if int(match_obj.group(2)) < 3:
                return 0.0
            return int(match_obj.group(1)) / int(match_obj.group(2))

        match_obj = re.match(r'^([\d\.]+?)\s*%$', word)
        if match_obj:
            return float(match_obj.group(1)) / 100

        match_obj = re.match(r'^(\d+)\s*of\s*(\d+)$', word)
        if match_obj:
            if int(match_obj.group(2)) < 3:
                return 0.0
            return int(match_obj.group(1)) / int(match_obj.group(2))

        match_obj = re.match(r'^(\d+)%?-(\d+)%$', word)
        if match_obj:
            return (int(match_obj.group(1)) + int(match_obj.group(2))) / 200

        print('Error:', word)
        return 0.0

    def get_max_frequency_from_frequency_string(self, frequency_string: str) -> float:
        """
        Get max frequency from semicolon-sep string, return as float (0-1)
        
        Args:
            frequency_string: semicolon-sep freq string, e.g. 80%; 60%; 90%
            
        Returns:
            float: max frequency as decimal (0.0-1.0)
        """
        if not frequency_string or not frequency_string.strip():
            return 0.0
        
        # split by semicolon
        frequency_parts = [part.strip() for part in frequency_string.split(';') if part.strip()]
        
        if not frequency_parts:
            return 0.0
        
        max_frequency = 0.0
        
        for part in frequency_parts:
            try:
                # use word2freq to convert each freq
                freq_value = self.word2freq(part)
                max_frequency = max(max_frequency, freq_value)
            except:
                # on conversion failure, log and skip
                print(f'Error converting frequency: {part}')
                continue
        
        return max_frequency

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
        print(f"    Warning: Description for {disease_id} not found in JSON file, attempting to scrape from HPO website...")
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

    def _build_disease_line(self, index: int, disease_name: str, disease_info: Dict, phenotype_matches: List[Dict], list_phenotypes: bool = False) -> str:
        """
        Build formatted line for a single disease
        Args:
            index: index/ordinal
            disease_name: disease name
            disease_info: disease info
            phenotype_matches: phenotype match info
        Returns:
            formatted_line: formatted line
        """
        # use matched candidate disease name if available
        display_name = disease_info.get('standard_name', disease_name)
        
        # base format: index. disease name
        # base_line = f"{index}. **{display_name}**"
        base_line = f"Case {index} has **{display_name}**"
        
        # TODO: add associated phenotypes for reasoning?
        # if phenotype matches, add all
        disease_type = self.disease_types.get(disease_info.get('disease_id', ''), '')
        # if phenotype_matches:
        #     phenotype_info_parts = []
        #     for match in phenotype_matches:
        #         hp_code = match['hp_code']
        #         frequency = match['frequency']
                
        #         # if freq is HP code, get its description
        #         if 'HP:' in frequency:
        #             frequency_display = self._convert_hpo_frequency_to_description(frequency)
        #         else:
        #             frequency_display = frequency
                
        #         # case 1. with frequency
        #         # phenotype_info_parts.append(f"{hp_code} ({frequency_display})")

        #         # case 2. without frequency
        #         phenotype_info_parts.append(f"{hp_code}")
            
        #     # join phenotype info with semicolon
        #     # phenotype_info = ", ".join(phenotype_info_parts)
        #     # base_line += f" - {phenotype_info}"

        #     phenotype_description = ", ".join([self._get_phenotype_description(match['hp_code']) for match in phenotype_matches])
        #     if disease_type:
        #         base_line += f". Disease Category: **{disease_type}**. Description: The patient pending diagnosis exhibits **{phenotype_description}**, which belong to the typical symptoms of this disease."
        #     else:
        #         base_line += f". Description: The patient pending diagnosis exhibits **{phenotype_description}**, which belong to the typical symptoms of this disease."
        # else:
        disease_description = self.get_disease_description(disease_info.get('disease_id', ''))
        if disease_type:
            base_line += f". Disease Category: **{disease_type}**. Description: {disease_description}"
        else:
            base_line += f". Description: {disease_description}"

        # json format
        # \"Patient 20\": {
        #     \"Disease name\": \"Sweet syndrome\",
        #     \"Disease id\": [
        #     \"ORPHA:3243\"
        #     ],
        #     \"Disease category\": \"Rare skin disease\",
        #     \"Disease description\": \"A rare inflammatory disease characterized by abrupt appearance of painful, edematous and erythematous papules, plaques and nodules on the skin, and frequently accompanied by fever and neutrophilia with a dense infiltration of mature neutrophils that are typically located in the upper dermis. The disease is classically associated with inflammatory disease, pregnancy, infection (mostly of the upper respiratory tract), or vaccination but may be idiopathic, associated with a hematological or visceral malignancy, or drug-induced.\"
        # },
        # set base_line to above JSON format as string
        if disease_description == "":
                disease_description = "[Information is missing; please infer based on your memory.]"
        synonyms = self.disease_mapping_with_synonyms.get(disease_info.get('disease_id', ''), [])

        # get all phenotypes for disease including freq
        phenotypes = self.disease_to_phenotypes.get(disease_info.get('disease_id', ''), [])
        
        # first collect all phenotypes with freq
        phenotype_list = []
        for phenotype in phenotypes:
            freq_key = (phenotype, disease_info.get('disease_id', ''))
            freq_info = self.phenotype_disease_frequency[freq_key]
            frequency_string = freq_info.get('frequency', '')
            # use get_max_frequency_from_frequency_string to convert to number
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

        if list_phenotypes and phenotypes_text != "":
            base_line = json.dumps({
                f"Case {index}": {
                    "Disease name": disease_info.get('display_name', display_name),
                    # "Synonyms": synonyms,
                    # "Disease id": disease_info.get('disease_id', ''),
                    # "Disease category": disease_type,
                    "Disease phenotypes (with frequency)": phenotypes_text,
                    # "Disease description": disease_description
                }
            }, ensure_ascii=False, indent=2)
        else:
            base_line = json.dumps({
                f"Case {index}": {
                    "Disease name": disease_info.get('display_name', display_name),
                    # "Synonyms": synonyms,
                    # "Disease id": disease_info.get('disease_id', ''),
                    "Disease category": disease_type,
                    # "Disease phenotypes (with frequency)": phenotypes_text,
                    "Disease description": disease_description
                }
            }, ensure_ascii=False, indent=2)

        if base_line.startswith("{") and base_line.endswith("}"):
            # remove leading/trailing braces and newlines
            base_line = base_line[1:-1].strip()

        return base_line
    
    def calculate_semantic_similarity(self, text1: str, text2: str) -> float:
        """Calculate semantic similarity between two texts"""
        text1 = text1.lower()
        text2 = text2.lower()
        
        # If both texts end with "syndrome", remove it for better matching
        if text1.endswith('syndrome') and text2.endswith('syndrome'):
            text1 = text1[:-9].strip()  # Remove "syndrome" (9 characters)
            text2 = text2[:-9].strip()  # Remove "syndrome" (9 characters)
        elif text1.endswith('disease') and text2.endswith('disease'):
            text1 = text1[:-7].strip()  # Remove "disease" (7 characters)
            text2 = text2[:-7].strip()  # Remove "disease" (7 characters)
        
        try:
            embeddings = self.sentence_model.encode([text1, text2])
            similarity = np.dot(embeddings[0], embeddings[1]) / (np.linalg.norm(embeddings[0]) * np.linalg.norm(embeddings[1]))
            return float(similarity)
        except Exception as e:
            print(f"Error calculating semantic similarity: {e}")
            return 0.0
    
    def calculate_biomedical_similarity(self, text1: str, text2: str) -> float:
        """Calculate biomedical semantic similarity using BioLORD-2023 model
        
        Args:
            text1: First phenotype text
            text2: Second phenotype text
            
        Returns:
            float: Similarity score between 0 and 1
        """
        try:
            # Use BioLORD model for biomedical semantic similarity
            embeddings = self.biolord_model.encode([text1, text2])
            similarity = np.dot(embeddings[0], embeddings[1]) / (np.linalg.norm(embeddings[0]) * np.linalg.norm(embeddings[1]))
            return float(similarity)
        except Exception as e:
            print(f"Error calculating biomedical similarity with BioLORD: {e}")
            # Fallback to standard similarity
            return self.calculate_semantic_similarity(text1, text2)
    
    def evaluate_phenotype_similarity(self, disease_phenotype_data: dict, similarity_threshold: float = 0.7):
        """Evaluate phenotype similarity with BioLORD-2023, output mismatched as JSON
        
        Args:
            disease_phenotype_data: disease-phenotype data
            similarity_threshold: similarity threshold, default 0.7
            
        Returns:
            list: JSON with error match info
        """
        error_matches = []
        
        # init stats structure
        corrected_matched_counts = {}
        
        # first collect all phenotypes, init to 0
        all_phenotypes_set = set()
        for disease_name, data in disease_phenotype_data.items():
            matched_phenotypes = data.get('matched', [])
            unmatched_phenotypes = data.get('unmatched', [])
            all_phenotypes_set.update(matched_phenotypes)
            all_phenotypes_set.update(unmatched_phenotypes)
        
        # init all phenotype counts to 0
        for phenotype in all_phenotypes_set:
            corrected_matched_counts[phenotype] = 0
        
        # collect rebuilt candidate disease list
        rebuilt_diseases = {}
        
        for disease_name, data in disease_phenotype_data.items():
            matched_phenotypes = data.get('matched', [])
            unmatched_phenotypes = data.get('unmatched', [])
            all_phenotypes = data.get('all_phenotypes', {})
            
            if not all_phenotypes:
                continue
                
            database_phenotypes = list(all_phenotypes.values()) if isinstance(all_phenotypes, dict) else all_phenotypes
            
            # collect error-matched phenotypes for current disease
            error_phenotypes = []
            
            # check each matched phenotype
            for matched_phenotype in matched_phenotypes:
                best_similarity = 0.0
                best_match = None
                
                # find most similar DB phenotype
                for db_phenotype in database_phenotypes:
                    similarity = self.calculate_biomedical_similarity(matched_phenotype, db_phenotype)
                    if similarity > best_similarity:
                        best_similarity = similarity
                        best_match = db_phenotype
                
                # if below threshold, record error match
                if best_similarity < 0.4:
                    error_phenotypes.append(matched_phenotype)
            
            # collect missed: unmatched patient phenotypes similar to known disease phenotypes
            missed_phenotypes = []
            if unmatched_phenotypes:
                for unmatched_phenotype in unmatched_phenotypes:
                    best_similarity = 0.0
                    best_match = None
                    
                    # find most similar DB phenotype
                    for db_phenotype in database_phenotypes:
                        similarity = self.calculate_biomedical_similarity(unmatched_phenotype, db_phenotype)
                        if similarity > best_similarity:
                            best_similarity = similarity
                            best_match = db_phenotype
                    
                    # if above threshold, record as missed
                    if best_similarity > 0.7:
                        missed_phenotypes.append(unmatched_phenotype)
            
            # if errors or missed, add to result
            if error_phenotypes and missed_phenotypes:
                error_text = f"The matched phenotypes of disease **{disease_name}** should include **{', '.join(missed_phenotypes)}**, and should exclude **{', '.join(error_phenotypes)}**, please reconsider the ranking of this disease."
                error_matches.append(error_text)
            elif missed_phenotypes:
                error_text = f"The matched phenotypes of disease **{disease_name}** should include **{', '.join(missed_phenotypes)}**, please reconsider the ranking of this disease."
                error_matches.append(error_text)
            elif error_phenotypes:
                # build error match description
                error_text = f"The matched phenotypes of disease **{disease_name}** should exclude **{', '.join(error_phenotypes)}**, please reconsider the reliability of this diagnosis."                
                error_matches.append(error_text)
            
            
            # count phenotype in corrected matched (missed as corrected)
            for phenotype in missed_phenotypes:
                corrected_matched_counts[phenotype] = corrected_matched_counts.get(phenotype, 0) + 1
            for phenotype in matched_phenotypes:
                if phenotype not in error_phenotypes:
                    corrected_matched_counts[phenotype] = corrected_matched_counts.get(phenotype, 0) + 1

            if data.get('disease_id', 'UNKNOWN') == "UNKNOWN":
                error_text = f"The disease **{disease_name}** is not a standalone rare diseases or belongs to an overly broad disease category, please reconsider the reliability of this diagnosis."
                error_matches.append(error_text)
                continue  # skip UNKNOWN, do not add to rebuild
            
            # rebuild candidate list in loop
            # remove error_phenotypes from matched, add to unmatched
            corrected_matched = [p for p in matched_phenotypes if p not in error_phenotypes]
            corrected_unmatched = [p for p in unmatched_phenotypes if p not in missed_phenotypes]
            
            # add missed to matched, remove from unmatched
            corrected_matched.extend(missed_phenotypes)
            corrected_unmatched.extend(error_phenotypes)
                        
            # build rebuilt disease data
            rebuilt_diseases[disease_name] = {
                "Matched": corrected_matched,
                "Unmatched": corrected_unmatched,
            }

        # rebuild done, rebuilt_diseases has the list
        rebuilt_disease_json = json.dumps(rebuilt_diseases, ensure_ascii=False, indent=4)
        # error_matches += [f"There are {len(rebuilt_diseases)} diseases in the **FINAL ANSWER** that are present in the disease database.\nAnd the correct disease-phenotype matching relationships of these {len(rebuilt_diseases)} diseases are as follows:\n {rebuilt_disease_json}\nPlease reorder the **FINAL ANSWER** according to this information. If there are fewer than 10 diseases in the above list, please add other possible candidate diseases."]
        
        # each phenotype proportion in corrected assoc, sort low to high
        total_diseases = len(disease_phenotype_data)
        
        # compute each phenotype proportion in corrected matched
        matched_proportions = []
        for phenotype in corrected_matched_counts:
            matched_count = corrected_matched_counts.get(phenotype, 0)
            proportion = matched_count / total_diseases if total_diseases > 0 else 0
            
            matched_proportions.append({
                'phenotype': phenotype,
                'matched_count': matched_count,
                'proportion': proportion
            })
        
        # sort by proportion low to high
        matched_proportions.sort(key=lambda x: x['proportion'])
        
        phenotype_with_low_proportion = [item['phenotype'] for item in matched_proportions if item['proportion'] == 0]
        if phenotype_with_low_proportion:
            error_matches += [f"Phenotypes **{', '.join(phenotype_with_low_proportion)}** are missing interpretable diseases in the diagnostic list. Please add at least one rare disease that can explain these phenotypes as well as some other phenotypes into the top 10 diagnoses."]
        return error_matches
    
    def find_best_match_rank(self, true_disease: str, predicted_diseases: List[str]) -> Dict:
        """Find the rank of the best matching disease using semantic similarity"""
        if not predicted_diseases:
            return {
                "best_match": None,
                "best_rank": -1,
                "best_similarity": 0.0,
                "all_similarities": []
            }
        
        similarities = []
        true_disease = true_disease.lower()
        for i, pred_disease in enumerate(predicted_diseases):
            pred_disease = pred_disease.lower()
            # handle pred disease with / and ;, split into multiple
            if isinstance(pred_disease, str):
                # split by ; then /
                if ';' in pred_disease:
                    # split pred disease by ;
                    semicolon_split = [d.strip() for d in pred_disease.split(';') if d.strip()]
                    all_split_diseases = []
                    
                    for sub_disease in semicolon_split:
                        if '/' in sub_disease:
                            # then split by /
                            slash_split = [d.strip() for d in sub_disease.split('/') if d.strip()]
                            all_split_diseases.extend(slash_split)
                        else:
                            all_split_diseases.append(sub_disease)
                    
                    # max similarity over split disease descriptions
                    max_similarity = 0.0
                    best_split_disease = pred_disease  # default to original
                    
                    for split_disease in all_split_diseases:
                        similarity = self.calculate_semantic_similarity(true_disease, split_disease)
                        if similarity > max_similarity:
                            max_similarity = similarity
                            best_split_disease = split_disease
                    
                    similarities.append({
                        "rank": i + 1,
                        "disease": pred_disease,
                        "best_split_disease": best_split_disease,
                        "similarity": max_similarity
                    })
                elif '/' in pred_disease:
                    # split pred disease by / only
                    split_pred_diseases = [d.strip() for d in pred_disease.split('/') if d.strip()]
                    
                    # max similarity over split disease descriptions
                    max_similarity = 0.0
                    best_split_disease = pred_disease  # default to original
                    
                    for split_disease in split_pred_diseases:
                        similarity = self.calculate_semantic_similarity(true_disease, split_disease)
                        if similarity > max_similarity:
                            max_similarity = similarity
                            best_split_disease = split_disease
                    
                    similarities.append({
                        "rank": i + 1,
                        "disease": pred_disease,
                        "best_split_disease": best_split_disease,
                        "similarity": max_similarity
                    })
                else:
                    # if pred has no sep, compute similarity directly
                    similarity = self.calculate_semantic_similarity(true_disease, pred_disease)
                    similarities.append({
                        "rank": i + 1,
                        "disease": pred_disease,
                        "best_split_disease": pred_disease,
                        "similarity": similarity
                    })
            else:
                # if pred not str, compute similarity directly
                similarity = self.calculate_semantic_similarity(true_disease, pred_disease)
                similarities.append({
                    "rank": i + 1,
                    "disease": pred_disease,
                    "best_split_disease": pred_disease,
                    "similarity": similarity
                })
        
        # Sort by similarity (descending)
        similarities.sort(key=lambda x: x["similarity"], reverse=True)
        
        # Find the best match
        best_match = similarities[0]
        
        return {
            "best_match": best_match["disease"],
            "best_rank": best_match["rank"],
            "best_similarity": best_match["similarity"],
            "all_similarities": similarities
        }
    
    def evaluate_prediction_accuracy(self, true_diseases: List[str], predicted_diseases: List[str]) -> Dict:
        """Evaluate prediction accuracy using semantic similarity for multiple true diseases"""
        if not predicted_diseases:
            return {
                "correct": False,
                "accuracy": 0.0,
                "best_match": None,
                "best_rank": -1,
                "best_similarity": 0.0,
                "matched_true_disease": None
            }

        # predicted_diseases mapping
        # find disease ID for pred, get synonyms by ID, compute similarity
        for i, predicted_disease in enumerate(predicted_diseases):
            # use disease name matching for best match and ID
            match_result = self._find_best_matching_disease_two_stage(predicted_disease)
            predicted_disease_id = match_result.get("disease_id")
            if predicted_disease_id and predicted_disease_id != "UNKNOWN":
                predicted_disease_synonyms = self.disease_mapping_with_synonyms.get(predicted_disease_id, [])
                if predicted_disease_synonyms:
                    predicted_diseases[i] += f"; {'; '.join(predicted_disease_synonyms)}"
        
        def split_predicted_synonyms(name):
            """Split predicted disease entries by ';' to handle synonym lists."""
            if not isinstance(name, str):
                return [str(name).strip()] if name is not None else [""]
            parts = [part.strip() for part in name.split(';') if part and part.strip()]
            return parts or [name.strip()]
        
        def best_similarity_with_predicted(true_name, predicted_entry):
            """Find best similarity between a true disease name and a predicted entry (with possible synonyms)."""
            best_similarity = 0.0
            best_pred_name = None
            for candidate in split_predicted_synonyms(predicted_entry):
                similarity = self.calculate_semantic_similarity(true_name, candidate)
                if similarity > best_similarity:
                    best_similarity = similarity
                    best_pred_name = candidate
            if best_pred_name is None:
                best_pred_name = split_predicted_synonyms(predicted_entry)[0]
            return best_similarity, best_pred_name
        
        # if true_diseases is str, convert to list
        if isinstance(true_diseases, str):
            true_diseases = [true_diseases]
        
        # handle disease with / and ;, split into multiple
        processed_true_diseases = []
        for disease in true_diseases:
            if isinstance(disease, str):
                # split by ; then /
                if ';' in disease:
                    # split by ';'
                    semicolon_split = [d.strip() for d in disease.split(';') if d.strip()]
                    for sub_disease in semicolon_split:
                        if '/' in sub_disease:
                            # then split by /
                            slash_split = [d.strip() for d in sub_disease.split('/') if d.strip()]
                            processed_true_diseases.extend(slash_split)
                        else:
                            processed_true_diseases.append(sub_disease)
                elif '/' in disease:
                    # split disease name by / only
                    split_diseases = [d.strip() for d in disease.split('/') if d.strip()]
                    processed_true_diseases.extend(split_diseases)
                else:
                    processed_true_diseases.append(disease)
            else:
                processed_true_diseases.append(disease)
        
        # Remove duplicates while preserving order
        true_diseases = list(dict.fromkeys(processed_true_diseases))
        
        # for each true disease, best match
        best_overall_match = None
        best_overall_similarity = 0.0
        best_overall_rank = -1
        matched_true_disease = None

        # compute top1/top3/top5/top10 correctness
        top1_correct = False
        top3_correct = False
        top5_correct = False
        top10_correct = False
        
        # store topX best match info
        top1_best_match = None
        top3_best_match = None
        top5_best_match = None
        top10_best_match = None
        
        for true_disease in true_diseases:
            match_result = self.find_best_match_rank(true_disease, predicted_diseases)
            
            if match_result["best_similarity"] > best_overall_similarity:
                best_overall_similarity = match_result["best_similarity"]
                best_overall_match = match_result["best_match"]
                best_overall_rank = match_result["best_rank"]
                matched_true_disease = true_disease
            
            # check top1/top3/top5 correctness
            similarity_threshold = self.config['evaluation_config']['similarity_threshold']
            
            # TODO: modify
            # check if rank-1 matches true
            if len(predicted_diseases) > 0:
                top1_similarity, matched_pred_name = best_similarity_with_predicted(true_disease, predicted_diseases[0])
                if top1_similarity >= similarity_threshold:
                    top1_correct = True
                    if top1_best_match is None or top1_similarity > top1_best_match[2]:
                        top1_best_match = [matched_pred_name, true_disease, top1_similarity]
            
            # check if top-3 contains match
            for i in range(min(3, len(predicted_diseases))):
                top3_similarity, matched_pred_name = best_similarity_with_predicted(true_disease, predicted_diseases[i])
                if top3_similarity >= similarity_threshold:
                    top3_correct = True
                    if top3_best_match is None or top3_similarity > top3_best_match[2]:
                        top3_best_match = [matched_pred_name, true_disease, top3_similarity]
                    break
            
            # check if top-5 contains match
            for i in range(min(5, len(predicted_diseases))):
                top5_similarity, matched_pred_name = best_similarity_with_predicted(true_disease, predicted_diseases[i])
                if top5_similarity >= similarity_threshold:
                    top5_correct = True
                    if top5_best_match is None or top5_similarity > top5_best_match[2]:
                        top5_best_match = [matched_pred_name, true_disease, top5_similarity]
                    break
            
            # check if top-10 contains match
            for i in range(min(10, len(predicted_diseases))):
                top10_similarity, matched_pred_name = best_similarity_with_predicted(true_disease, predicted_diseases[i])
                if top10_similarity >= similarity_threshold:
                    top10_correct = True
                    if top10_best_match is None or top10_similarity > top10_best_match[2]:
                        top10_best_match = [matched_pred_name, true_disease, top10_similarity]
                    break
        
        # Consider prediction correct if similarity is above threshold
        similarity_threshold = self.config['evaluation_config']['similarity_threshold']
        correct = best_overall_similarity >= similarity_threshold
        
        return {
            "correct": correct,
            "accuracy": best_overall_similarity,
            "best_match": best_overall_match,
            "best_rank": best_overall_rank,
            "best_similarity": best_overall_similarity,
            "matched_true_disease": matched_true_disease,
            "top1_correct": top1_correct,
            "top3_correct": top3_correct,
            "top5_correct": top5_correct,
            "top10_correct": top10_correct,
            "top1_best_match": top1_best_match,
            "top3_best_match": top3_best_match,
            "top5_best_match": top5_best_match,
            "top10_best_match": top10_best_match,
            "all_predictions": predicted_diseases,
            "all_true_diseases": true_diseases,
        }
    
    def test_predictions(self, prompts: List[PromptData], num_samples: int = None, sample_indices: List[int] = None,
                        case_library: str = None, use_few_shot: bool = False, k_shot: int = None,
                        output_file: str = None, max_retries: int = None) -> Dict:
        # Use config values if not provided
        if k_shot is None:
            k_shot = self.config['evaluation_config']['default_k_shot']
        if max_retries is None:
            max_retries = self.config['evaluation_config']['max_retries']
        """Test predictions on a list of prompts with optional few-shot learning and immediate file writing"""
        if num_samples:
            prompts = prompts[:num_samples]
        if sample_indices is not None and len(sample_indices) > 0:
            # Validate indices
            max_index = len(prompts) - 1
            invalid_indices = [idx for idx in sample_indices if idx < 0 or idx > max_index]
            if invalid_indices:
                raise ValueError(f"Invalid sample indices: {invalid_indices}. Valid range: 0-{max_index}")
            
            # Select specified samples
            selected_prompts = []
            for idx in sample_indices:
                selected_prompts.append(prompts[idx])
            prompts = selected_prompts
        
        print(f"Testing {len(prompts)} prompts...")
        
        # Load case library if few-shot is enabled
        case_samples = None
        if use_few_shot and case_library:
            try:
                case_samples = self.load_case_library(case_library)
                print(f"Loaded {len(case_samples)} samples from case library for few-shot learning")
            except Exception as e:
                print(f"Warning: Failed to load case library: {e}")
                use_few_shot = False
        
        results = []
        correct_count = 0
        total_time = 0
        
        # Count multi-step prompts
        multi_step_count = sum(1 for p in prompts if p.is_multi_step)
        three_step_count = sum(1 for p in prompts if p.is_multi_step and p.step3_prompt)
        two_step_count = multi_step_count - three_step_count
        single_step_count = len(prompts) - multi_step_count
        
        # Initialize output file with metadata
        if output_file:
            # Get model information
            model_info = {
                "model_name": self.model_name
            }
            
            # For OpenRouter API, show the actual model being used
            if self.is_openrouter:
                model_info["actual_model"] = self.actual_model_name
            
            initial_data = {
                "metadata": {
                    "model_info": model_info,
                    "total_prompts": len(prompts),
                    "two_step_prompts": two_step_count,
                    "three_step_prompts": three_step_count,
                    "single_step_prompts": single_step_count,
                    "case_library": case_library,
                    "status": "in_progress",
                    "completed_count": 0,
                    "valid_count": 0,
                    "correct_count": 0,
                    "top1_accuracy": 0.0,
                    "top5_accuracy": 0.0,
                    "top10_accuracy": 0.0,
                    "step_statistics": {},
                    "total_time": 0.0,
                    "average_time": 0.0,
                    "max_retries": max_retries,
                },
                "results": []
            }
            out_dir = os.path.dirname(output_file)
            if out_dir:
                os.makedirs(out_dir, exist_ok=True)
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(initial_data, f, ensure_ascii=False, indent=2)
            print(f"Initialized output file: {output_file}")
        
        for i, prompt_data in enumerate(prompts):
                
            # show all diseases
            diseases_display = "; ".join(prompt_data.true_diseases) if prompt_data.true_diseases else "Unknown Disease"
            print(f"\nProcessing prompt {i+1}/{len(prompts)}: {diseases_display} (Sample ID: {prompt_data.sample_id})")
            
            # clear insert-step cache per sample (sample-specific)
            self.insert_step_outputs = {}
            self.history = []
            
            # Retry mechanism for handling errors
            retry_count = 0
            success = False
            
            while retry_count < max_retries and not success:
                try:
                    if retry_count > 0:
                        print(f"  Retry attempt {retry_count}/{max_retries} for prompt {i+1}")
                        # Clear GPU memory before retry
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
                        # Add a small delay before retry
                        time.sleep(2)
                    
                    # Check if this is a multi-step prompt (include: dynamic multi-step, three-step, two-step)
                    if prompt_data.is_multi_step:
                        
                        # Check if memory reasoning is enabled
                        memory_mode = None
                        if self.config.get('memory_reasoning_config', {}).get('enabled', False):
                            memory_mode = "full"  # fixed full memory mode
                        
                        # Check if this is a dynamic multi-step prompt
                        if memory_mode:
                            print(f"  Using memory reasoning (mode: {memory_mode}) for sample {prompt_data.sample_id}")
                            
                            # Prepare step prompts list
                            if hasattr(prompt_data, 'step_prompts') and prompt_data.step_prompts:
                                # Use step_prompts list if available
                                step_prompts = prompt_data.step_prompts
                            else:
                                # Fallback to individual step prompts
                                step_prompts = [prompt_data.step1_prompt, prompt_data.step2_prompt]
                                if prompt_data.step3_prompt:
                                    step_prompts.append(prompt_data.step3_prompt)
                            
                            # Initialize similar_samples_info
                            similar_samples_info = None
                            
                            # Apply few-shot learning if enabled
                            if use_few_shot and case_samples:
                                similar_samples, similar_values = self.find_similar_samples(prompt_data.phenotypes, case_samples, k_shot, use_embeddings=True)
                                if similar_samples:
                                    step_prompts = [self.create_few_shot_prompt(prompt, similar_samples) for prompt in step_prompts]
                                    # enrich sample with full description
                                    enriched_samples = [self._enrich_sample_info(s) for s in similar_samples]
                                    similar_samples_info = {
                                        'count': len(similar_samples),
                                        'samples': enriched_samples,
                                        'similar_values': similar_values
                                    }
                                    print(f"  Added {len(similar_samples)} few-shot examples to all steps")
                            
                            # Generate memory reasoning prediction
                            memory_result = self.generate_memory_reasoning_prediction_v2(
                                step_prompts=step_prompts,
                                target_phenotypes=prompt_data.phenotypes
                            )
                            
                            prediction = memory_result[f"step{memory_result["total_steps"]}_prediction"]
                            generation_time = memory_result["total_time"]
                            predicted_diseases = memory_result["final_diseases"]
                            
                            # Build multi-step info for memory reasoning
                            multi_step_info = {
                                "memory_mode": memory_mode,
                                "total_steps": memory_result["total_steps"]
                            }
                            
                            # Add step-specific information
                            for step_num in range(memory_result["total_steps"]):
                                step_key = f"step{step_num+1}"
                                step_diseases = memory_result[f"{step_key}_diseases"]
                                step_evaluation = self._evaluate_step_accuracy(step_diseases, prompt_data.true_diseases)
                                step_disease_mappings = self._analyze_step_disease_mappings(step_diseases, step_num+1, prompt_data.true_diseases, prompt_data.phenotypes)
                                
                                multi_step_info.update({
                                    f"{step_key}_prediction": memory_result[f"{step_key}_prediction"],
                                    f"{step_key}_disease_mappings": step_disease_mappings,
                                    f"{step_key}_evaluation": step_evaluation,
                                    f"{step_key}_time": memory_result[f"{step_key}_time"]
                                })
                        
                        # Check if this is a three-step prompt
                        elif prompt_data.step3_prompt:
                            print(f"  Using three-step reasoning for sample {prompt_data.sample_id}")
                            
                            # Prepare step prompts (with or without few-shot examples)
                            step1_prompt = prompt_data.step1_prompt
                            step2_prompt = prompt_data.step2_prompt
                            step3_prompt = prompt_data.step3_prompt
                            similar_samples_info = None
                            
                            if use_few_shot and case_samples:
                                # Find similar samples and create enhanced prompts
                                similar_samples, similar_values = self.find_similar_samples(prompt_data.phenotypes, case_samples, k_shot, use_embeddings=True)
                                if similar_samples:
                                    step1_prompt = self.create_few_shot_prompt(prompt_data.step1_prompt, similar_samples)
                                    step2_prompt = self.create_few_shot_prompt(prompt_data.step2_prompt, similar_samples)
                                    step3_prompt = self.create_few_shot_prompt(prompt_data.step3_prompt, similar_samples)
                                    # enrich sample with full description
                                    enriched_samples = [self._enrich_sample_info(s) for s in similar_samples]
                                    similar_samples_info = {
                                        'count': len(similar_samples),
                                        'samples': enriched_samples,
                                        'similar_values': similar_values
                                    }
                                    print(f"  Added {len(similar_samples)} few-shot examples to all three steps")
                            
                            # Generate three-step prediction
                            # optional: choose among three-step reasoning variants
                            multi_step_result = self.generate_three_step_prediction_v1(step1_prompt, step2_prompt, step3_prompt, prompt_data.phenotypes)
                            prediction = multi_step_result["step3_prediction"]
                            generation_time = multi_step_result["total_time"]
                            predicted_diseases = multi_step_result["final_diseases"]
                            
                            # Analyze step1, step2, step3 diseases mapping to true diseases
                            step1_diseases = multi_step_result["step1_diseases"]
                            step1_disease_mappings = self._analyze_step_disease_mappings(step1_diseases, 1, prompt_data.true_diseases, prompt_data.phenotypes)
                            step2_diseases = multi_step_result["step2_diseases"]
                            step2_disease_mappings = self._analyze_step_disease_mappings(step2_diseases, 2, prompt_data.true_diseases, prompt_data.phenotypes)
                            step3_disease_mappings = self._analyze_step_disease_mappings(predicted_diseases, 3, prompt_data.true_diseases, prompt_data.phenotypes)
                            
                            # Evaluate each step's accuracy
                            step1_evaluation = self._evaluate_step_accuracy(step1_diseases, prompt_data.true_diseases)
                            step2_evaluation = self._evaluate_step_accuracy(step2_diseases, prompt_data.true_diseases)
                            step3_evaluation = self._evaluate_step_accuracy(predicted_diseases, prompt_data.true_diseases)
                            
                            # Store three-step specific information (without thinking and prompt fields)
                            multi_step_info = {
                                "step1_prediction": multi_step_result["step1_prediction"],
                                "step1_disease_mappings": step1_disease_mappings,
                                "step1_evaluation": step1_evaluation,
                                "step1_time": multi_step_result["step1_time"],
                                "step2_prediction": multi_step_result["step2_prediction"],
                                "step2_disease_mappings": step2_disease_mappings,
                                "step2_evaluation": step2_evaluation,
                                "step2_time": multi_step_result["step2_time"],
                                "step3_prediction": multi_step_result["step3_prediction"],
                                "step3_disease_mappings": step3_disease_mappings,
                                "step3_evaluation": step3_evaluation,
                                "step3_time": multi_step_result["step3_time"]
                            }
                        
                        # Check if this is a two-step prompt
                        else:
                            print(f"  Using two-step reasoning for sample {prompt_data.sample_id}")
                            
                            # Prepare step prompts (with or without few-shot examples)
                            step1_prompt = prompt_data.step1_prompt
                            step2_prompt = prompt_data.step2_prompt
                            similar_samples_info = None
                            
                            if use_few_shot and case_samples:
                                # Find similar samples and create enhanced prompts
                                similar_samples, similar_values = self.find_similar_samples(prompt_data.phenotypes, case_samples, k_shot, use_embeddings=True)
                                if similar_samples:
                                    step1_prompt = self.create_few_shot_prompt(prompt_data.step1_prompt, similar_samples)
                                    step2_prompt = self.create_few_shot_prompt(prompt_data.step2_prompt, similar_samples)
                                    # enrich sample with full description
                                    enriched_samples = [self._enrich_sample_info(s) for s in similar_samples]
                                    similar_samples_info = {
                                        'count': len(similar_samples),
                                        'samples': enriched_samples,
                                        'similar_values': similar_values
                                    }
                                    print(f"  Added {len(similar_samples)} few-shot examples to both steps")
                            
                            # Generate multi-step prediction
                            multi_step_result = self.generate_two_step_prediction(step1_prompt, step2_prompt, prompt_data.phenotypes)
                            prediction = multi_step_result["step2_prediction"]
                            generation_time = multi_step_result["total_time"]
                            predicted_diseases = multi_step_result["final_diseases"]
                            
                            # Analyze step1 and step2 diseases mapping to true diseases
                            step1_diseases = multi_step_result["step1_diseases"]
                            step1_disease_mappings = self._analyze_step_disease_mappings(step1_diseases, 1, prompt_data.true_diseases, prompt_data.phenotypes)
                            step2_disease_mappings = self._analyze_step_disease_mappings(predicted_diseases, 2, prompt_data.true_diseases, prompt_data.phenotypes)
                            
                            # Evaluate each step's accuracy
                            step1_evaluation = self._evaluate_step_accuracy(step1_diseases, prompt_data.true_diseases)
                            step2_evaluation = self._evaluate_step_accuracy(predicted_diseases, prompt_data.true_diseases)
                            
                            # Store multi-step specific information (without thinking and prompt fields)
                            multi_step_info = {
                                "step1_prediction": multi_step_result["step1_prediction"],
                                "step1_disease_mappings": step1_disease_mappings,
                                "step1_evaluation": step1_evaluation,
                                "step1_time": multi_step_result["step1_time"],
                                "step2_prediction": multi_step_result["step2_prediction"],
                                "step2_disease_mappings": step2_disease_mappings,
                                "step2_evaluation": step2_evaluation,
                                "step2_time": multi_step_result["step2_time"]
                            }
                    
                    # Check if this is a one-step prompt
                    else:
                        # Single-step reasoning (original logic)
                        print(f"  Using single-step reasoning for sample {prompt_data.sample_id}")
                        
                        # Prepare prompt (with or without few-shot examples)
                        current_prompt = prompt_data.step1_prompt
                        similar_samples_info = None
                        
                        if use_few_shot and case_samples:
                            # Find similar samples and create enhanced prompt
                            similar_samples, similar_values = self.find_similar_samples(prompt_data.phenotypes, case_samples, k_shot, use_embeddings=True)
                            if similar_samples:
                                current_prompt = self.create_few_shot_prompt(prompt_data.step1_prompt, similar_samples)
                                # enrich sample with full description
                                enriched_samples = [self._enrich_sample_info(s) for s in similar_samples]
                                similar_samples_info = {
                                    'count': len(similar_samples),
                                    'samples': enriched_samples,
                                    'similar_values': similar_values
                                }
                                print(f"  Added {len(similar_samples)} few-shot examples")
                        
                        # Generate prediction
                        prediction_result = self.generate_prediction(current_prompt)
                        prediction = prediction_result["prediction"]
                        generation_time = prediction_result["generation_time"]
                        
                        # Check if prediction contains required steps
                        # Just for CoT prompts
                        # validation_result = self.validate_prediction_steps(prediction)
                        # if not validation_result["is_valid"]:
                        #     print(f"  Warning: Prediction missing required steps: {validation_result['missing_steps']}")
                        #     print(f"  Prediction does not meet format requirements")
                        #     # Set success to False to trigger retry in outer loop
                        #     success = False
                        #     continue
                        # else:
                        #     print(f"  Success: Prediction contains all required steps: {validation_result['found_steps']}")
                        
                        # Extract disease names from prediction
                        # prediction = prediction.replace('\"', '')
                        predicted_diseases = self.extract_disease_names_from_prediction(prediction)
                        predicted_diseases = list(dict.fromkeys(predicted_diseases))
                        
                        # No multi-step info for single-step reasoning
                        multi_step_info = None
                    
                    # Evaluate accuracy
                    evaluation = self.evaluate_prediction_accuracy(prompt_data.true_diseases, predicted_diseases)
                    
                    # Update statistics
                    if evaluation["correct"]:
                        correct_count += 1
                    total_time += generation_time
                    
                    # Store result
                    result = {
                        "sample_id": prompt_data.sample_id,  # use actual sample id
                        "true_diseases": prompt_data.true_diseases,
                        "true_disease_ids": prompt_data.true_disease_ids,
                        "phenotypes": prompt_data.phenotypes,
                        "prediction": prediction,
                        "evaluation": evaluation,
                        "few_shot_info": similar_samples_info,
                        "is_multi_step": prompt_data.is_multi_step,
                        "generation_time": generation_time,
                        "retry_count": retry_count
                    }
                    
                    # Add step-specific information
                    if prompt_data.is_multi_step:
                        
                        result["multi_step_info"] = multi_step_info
                        
                        # Check if this is memory reasoning mode
                        if multi_step_info and "memory_mode" in multi_step_info:
                            # Memory reasoning mode - extract enhanced prompts from memory_result
                            enhanced_step_prompts = []
                            for step_num in range(multi_step_info["total_steps"]):
                                step_key = f"step{step_num+1}"
                                enhanced_prompt = memory_result.get(f"{step_key}_enhanced_prompt", "")
                                enhanced_step_prompts.append(enhanced_prompt)
                            
                            result["prompt_info"] = {
                                "memory_mode": multi_step_info["memory_mode"],
                                "total_steps": multi_step_info["total_steps"],
                                "step_prompts": enhanced_step_prompts,  # These are the actual enhanced prompts with memory context
                                "step_prompts_length": [len(prompt) for prompt in enhanced_step_prompts]
                            }
                            # Memory reasoning doesn't have traditional thinking content structure
                            # thinking: in memory mode, collect each step thinking
                            # build step thinking dict with each step reasoning
                            step_thinkings = {}
                            for step_num in range(multi_step_info["total_steps"]):
                                step_key = f"step{step_num+1}"
                                # extract each step thinking from memory result
                                step_thinking = memory_result.get(f"{step_key}_thinking", "")
                                step_thinkings[step_key] = step_thinking
                            
                            result["thinking_content"] = {
                                "total_steps": multi_step_info["total_steps"],
                                "step_thinkings": step_thinkings,
                                "step_thinkings_length": [len(thinking) for thinking in step_thinkings.values()]
                            }
                        else:
                            # Traditional multi-step mode
                            # Check if this is a three-step prompt
                            if prompt_data.step3_prompt:
                                result["prompt_info"] = {
                                    "step1_prompt": step1_prompt,
                                    "step2_prompt": multi_step_result["enhanced_step2_prompt"],
                                    "step3_prompt": multi_step_result["enhanced_step3_prompt"],
                                    "step1_length": len(step1_prompt),
                                    "step2_length": len(multi_step_result["enhanced_step2_prompt"]),
                                    "step3_length": len(multi_step_result["enhanced_step3_prompt"]),
                                }
                                result["thinking_content"] = {
                                    "step1_thinking": multi_step_result["step1_thinking"],
                                    "step2_thinking": multi_step_result["step2_thinking"],
                                    "step3_thinking": multi_step_result["step3_thinking"]
                                }
                            else:
                                result["prompt_info"] = {
                                    "step1_prompt": step1_prompt,
                                    "step2_prompt": multi_step_result["enhanced_step2_prompt"],
                                    "step1_length": len(step1_prompt),
                                    "step2_length": len(multi_step_result["enhanced_step2_prompt"]),
                                }
                                result["thinking_content"] = {
                                    "step1_thinking": multi_step_result["step1_thinking"],
                                    "step2_thinking": multi_step_result["step2_thinking"]
                                }
                    else:
                        result["prompt_info"] = {
                            "enhanced_prompt": current_prompt,
                            "prompt_length": len(current_prompt),
                        }
                        result["thinking_content"] = {
                            "thinking_content": prediction_result["thinking_content"]
                        }
                    results.append(result)
                    
                    # Immediately write result to file
                    if output_file:
                        self._write_result_to_file(output_file, result, i+1, len(prompts), correct_count, total_time)
                    
                    print(f"  True diseases: {prompt_data.true_diseases}")
                    print(f"  Predicted diseases: {predicted_diseases}")
                    print(f"  Best match: {evaluation['best_match']} (rank {evaluation['best_rank']}, similarity {evaluation['best_similarity']:.3f})")
                    print(f"  Matched true disease: {evaluation['matched_true_disease']}")
                    print(f"  Correct: {evaluation['correct']}")
                    
                    if prompt_data.is_multi_step:
                        if prompt_data.step3_prompt:
                            print(f"  Step 1 time: {multi_step_info['step1_time']:.2f}s")
                            print(f"  Step 2 time: {multi_step_info['step2_time']:.2f}s")
                            print(f"  Step 3 time: {multi_step_info['step3_time']:.2f}s")
                            print(f"  Total generation time: {generation_time:.2f}s")
                            # print(f"  Step 1 diseases: {multi_step_info['step1_prediction']}")
                            # print(f"  Step 2 diseases: {multi_step_info['step2_prediction']}")
                        else:
                            # print(f"  Step 1 time: {multi_step_info['step1_time']:.2f}s")
                            # print(f"  Step 2 time: {multi_step_info['step2_time']:.2f}s")
                            print(f"  Total generation time: {generation_time:.2f}s")
                            # print(f"  Step 1 diseases: {multi_step_info['step1_prediction']}")
                    else:
                        print(f"  Generation time: {generation_time:.2f}s")
                    
                    if retry_count > 0:
                        print(f"  Successfully completed after {retry_count} retries")
                    print(f"  Result saved to: {output_file}")
                    
                    success = True
                    
                except Exception as e:
                    retry_count += 1
                    error_msg = str(e)
                    print(f"  Error processing prompt {i+1} (attempt {retry_count}/{max_retries}): {error_msg}")
                    
                    # Check if it's a list index out of range error or other recoverable error
                    if "list index out of range" in error_msg.lower() or "index out of range" in error_msg.lower():
                        print(f"  Detected index out of range error, will retry...")
                    elif "cuda out of memory" in error_msg.lower() or "out of memory" in error_msg.lower():
                        print(f"  Detected memory error, clearing cache and retrying...")
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
                    else:
                        print(f"  Non-recoverable error: {error_msg}")
                        break
                    
                    # If we've exhausted all retries, write error information to file
                    if retry_count >= max_retries:
                        print(f"  Failed to process prompt {i+1} after {max_retries} attempts")
                        if output_file:
                            error_result = {
                                "sample_id": prompt_data.sample_id,  # use actual sample id
                                "true_diseases": prompt_data.true_diseases,
                                "true_disease_ids": prompt_data.true_disease_ids,
                                "error": error_msg,
                                "retry_count": retry_count,
                                "status": "error"
                            }
                            self._write_result_to_file(output_file, error_result, i+1, len(prompts), correct_count, total_time)
                        break
        
        # Calculate rank statistics
        ranks = [r["evaluation"]["best_rank"] for r in results if r["evaluation"]["best_rank"] > 0]
        avg_rank = sum(ranks) / len(ranks) if ranks else 0.0
        
        # Calculate top1, top5, top10 and total accuracy
        valid_results = [r for r in results if r["evaluation"]["best_rank"] > 0]
        if valid_results:
            # Top1: samples with rank-1 match / total valid
            top1_correct_count = sum(1 for r in valid_results if r["evaluation"].get("top1_correct", False))
            top1_accuracy = top1_correct_count / len(valid_results)
            
            # Top5: samples with match in top-5 / total valid
            top5_correct_count = sum(1 for r in valid_results if r["evaluation"].get("top5_correct", False))
            top5_accuracy = top5_correct_count / len(valid_results)
            
            # Top10: samples with match in top-10 / total valid
            top10_correct_count = sum(1 for r in valid_results if r["evaluation"].get("top10_correct", False))
            top10_accuracy = top10_correct_count / len(valid_results)
            
        else:
            top1_accuracy = 0.0
            top5_accuracy = 0.0
            top10_accuracy = 0.0

        # Calculate overall statistics
        accuracy = correct_count / len(valid_results) if valid_results else 0.0
        avg_time = total_time / len(valid_results) if valid_results else 0.0
        
        # Calculate similarity statistics
        similarities = [r["evaluation"]["best_similarity"] for r in results]
        avg_similarity = sum(similarities) / len(similarities) if similarities else 0.0
        
        # Calculate step-wise statistics (auto-detect steps)
        steps_to_calculate = self._identify_steps_for_statistics(results)
        step_statistics = {}
        
        if steps_to_calculate:
            print(f"  Auto-detected steps for overall statistics: {steps_to_calculate}")
        
        for step_num in steps_to_calculate:
            step_evaluation_key = f"step{step_num}_evaluation"
            step_stats = self._calculate_step_statistics(results, step_evaluation_key)
            step_statistics[f'step{step_num}'] = step_stats
        
        overall_stats = {
            "total_prompts": len(prompts),
            "multi_step_prompts": multi_step_count,
            "two_step_prompts": two_step_count,
            "three_step_prompts": three_step_count,
            "single_step_prompts": single_step_count,
            "valid_predictions": len(valid_results),
            "correct_predictions": correct_count,
            "accuracy": accuracy,
            "top1_accuracy": top1_accuracy,
            "top5_accuracy": top5_accuracy,
            "top10_accuracy": top10_accuracy,
            "step_statistics": step_statistics,
            "average_generation_time": avg_time,
            "average_rank": avg_rank,
            "average_similarity": avg_similarity,
            "results": results
        }
        
        return overall_stats
    
    def save_results(self, results: Dict, output_file: str):
        """
        Save results to JSON. If test_predictions already wrote metadata+results to the file, do not overwrite;
        otherwise write in legacy format (backward compatibility).
        """
        try:
            with open(output_file, 'r', encoding='utf-8') as f:
                existing = json.load(f)
                if 'metadata' in existing and 'results' in existing:
                    return
        except (FileNotFoundError, json.JSONDecodeError):
            pass

        print(f"Saving results to: {output_file}")
        out_dir = os.path.dirname(output_file)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        print("Results saved successfully!")


def _build_parser(config: Dict) -> argparse.ArgumentParser:
    """Build CLI argument parser."""
    parser = argparse.ArgumentParser(description="Test Phenotype to Disease Prediction")
    openrouter_cfg = config.get('openrouter_config', {})

    # I/O
    parser.add_argument("--prompts_file", type=str, required=True, help="Path to prompts JSON")
    parser.add_argument("--output_file", type=str, default=config['output_config']['default_output_file'], help="Output path for results")

    # Sample selection
    parser.add_argument("--num_samples", type=int, default=None, help="Number of samples to test (default: all)")
    parser.add_argument("--sample_indices", type=str, default=None, help="Sample indices, e.g. '0,5,10' or '0-5,10-15'")

    # Model
    parser.add_argument("--model_name", type=str, default=config['model_config']['default_model_name'],
                        help="Platform/model: Qwen/Qwen3-8B or openrouter")
    parser.add_argument("--api_model", type=str, default=None, help="OpenRouter model name (overrides config)")
    parser.add_argument("--cache_dir", type=str, default=None, help="Cache dir for Qwen models")
    parser.add_argument("--gpu_id", type=int, default=0, help="GPU id for Qwen")

    # OpenRouter
    parser.add_argument("--openrouter_api_key", type=str, default=None, help="OpenRouter API Key")
    parser.add_argument("--openrouter_proxy_url", type=str, default=openrouter_cfg.get('proxy_url', "https://openrouter.ai"), help="OpenRouter proxy URL")
    parser.add_argument("--openrouter_provider", type=str, default=openrouter_cfg.get('provider', ""), help="OpenRouter provider, e.g. fireworks")

    # Few-shot
    parser.add_argument("--use_few_shot", action="store_true", help="Enable few-shot (use case library)")
    parser.add_argument("--case_library", type=str, default=config.get('file_paths', {}).get('case_library', ''), help="Path to case library")
    parser.add_argument("--k_shot", type=int, default=config['evaluation_config']['default_k_shot'], help="Number of similar samples for few-shot")

    # Inference and evaluation
    parser.add_argument("--max_retries", type=int, default=config['evaluation_config']['max_retries'], help="Max retries on failure")
    parser.add_argument("--enable_thinking", action="store_true", default=config.get('model_config', {}).get('enable_thinking', False), help="Enable thinking mode")

    # Utility
    return parser


def _setup_gpu(args, is_openrouter: bool) -> bool:
    """Set up GPU for local Qwen. Returns False if main should exit."""
    if is_openrouter:
        return True
    if args.gpu_id is None:
        return True
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu_id)
    print(f"Set CUDA_VISIBLE_DEVICES={args.gpu_id}")
    if not torch.cuda.is_available():
        print("Warning: CUDA not available, GPU selection will be ignored")
        args.gpu_id = None
        return True
    if torch.cuda.device_count() == 0:
        print(f"Warning: No GPUs available after setting CUDA_VISIBLE_DEVICES={args.gpu_id}")
        return False
    print(f"Selected GPU {args.gpu_id} (visible as GPU 0): {torch.cuda.get_device_name(0)}")
    return True


def _resolve_api_credentials(args, config: Dict, is_openrouter: bool):
    """Resolve OpenRouter api_key and proxy_url. Returns (None, None) for non-OpenRouter."""
    if not is_openrouter:
        return None, None
    config.setdefault('openrouter_config', {})['provider'] = args.openrouter_provider
    return args.openrouter_api_key, args.openrouter_proxy_url


def _print_test_summary(results: Dict, args, is_openrouter: bool, actual_model_name: str = None):
    """Print test summary. actual_model_name is the model used (e.g. OpenRouter's google/gemini-2.0-flash-001); falls back to args.model_name."""
    model_display = actual_model_name if actual_model_name else args.model_name
    print("\n" + "=" * 60 + "\nTEST SUMMARY\n" + "=" * 60)
    print(f"Model used: {model_display}\nModel type: {'OpenRouter' if is_openrouter else 'Qwen'}")
    print(f"Total prompts: {results['total_prompts']}")
    print(f"  - Multi-step: {results['multi_step_prompts']} (two: {results['two_step_prompts']}, three: {results['three_step_prompts']})")
    print(f"  - Single-step: {results['single_step_prompts']}")
    print(f"Correct: {results['correct_predictions']}\nAccuracy: {results['accuracy']:.3f} ({results['accuracy']*100:.1f}%)")
    print(f"Top1: {results['top1_accuracy']:.3f}  Top5: {results['top5_accuracy']:.3f}")
    print(f"Average time: {results['average_generation_time']:.2f}s  rank: {results['average_rank']:.2f}  similarity: {results['average_similarity']:.3f}")


def _print_step_statistics(step_stats: Dict):
    """Print step-wise statistics."""
    print("\n" + "=" * 60 + "\nSTEP-WISE STATISTICS\n" + "=" * 60)
    for step_key in sorted(step_stats.keys()):
        s = step_stats[step_key]
        if s.get('total', 0) <= 0:
            continue
        n = step_key.replace('step', '')
        print(f"Step {n}: total={s['total']}  top1={s['top1_correct']}({s['top1_accuracy']*100:.1f}%)  top5={s['top5_correct']}({s['top5_accuracy']*100:.1f}%)  top10={s['top10_correct']}({s['top10_accuracy']*100:.1f}%)  total_correct={s['total_correct']}({s['total_accuracy']*100:.1f}%)")


def main():
    config = load_config()
    parser = _build_parser(config)
    args = parser.parse_args()

    is_openrouter = (args.model_name or "").lower().count("openrouter") > 0
    if is_openrouter:
        print(f"Using OpenRouter API: {args.model_name}")
    if not _setup_gpu(args, is_openrouter):
        return

    api_key, proxy_url = _resolve_api_credentials(args, config, is_openrouter)

    predictor = PhenotypeToDiseasePredictor(
        model_name=args.model_name, cache_dir=args.cache_dir, config=config, gpu_id=args.gpu_id,
        api_key=api_key, proxy_url=proxy_url, api_model=args.api_model, enable_thinking=args.enable_thinking
    )

    prompts = predictor.load_prompts(args.prompts_file)
    sample_indices = None
    if args.sample_indices:
        try:
            sample_indices = parse_sample_indices(args.sample_indices)
            print(f"Selected sample indices: {sample_indices}")
        except ValueError as e:
            print(f"Error parsing sample indices: {e}")
            return

    results = predictor.test_predictions(
        prompts, args.num_samples, sample_indices,
        case_library=args.case_library if args.use_few_shot else None,
        use_few_shot=args.use_few_shot, k_shot=args.k_shot, output_file=args.output_file, max_retries=args.max_retries
    )

    _print_test_summary(results, args, is_openrouter, actual_model_name=getattr(predictor, 'actual_model_name', None))
    if results.get('step_statistics'):
        _print_step_statistics(results['step_statistics'])

    predictor.save_results(results, args.output_file)


if __name__ == "__main__":
    main() 