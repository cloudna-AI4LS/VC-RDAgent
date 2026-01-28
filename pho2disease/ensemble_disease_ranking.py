#!/usr/bin/env python3
"""
Ensemble disease ranking: merge multiple case-extraction strategies and re-rank by Z-statistics.

Input: phenotype-disease JSON (same format as generate_prompts_bysteps), i.e. [[phenotypes],[diseases]] per sample.

Flow:
  1. For each sample, run several extraction methods (e.g. overlap+use_score, overlap+use_IC_weights,
     embedding+resort). Each method returns top_k (default 100) candidate diseases.
  2. Normalize per-method rankings to [0,1] ratios for cross-method comparison.
  3. For each disease, compute a Z-statistic from its normalized rank ratios across methods;
     lower Z means more consistently highly ranked across methods.
  4. Final ranking: sort by Z (ascending) and take the top final_top_k (default 50).

Output: ranking JSON (and optionally prompts in the same format as generate_prompts_bysteps), with
  hit-rate evaluation against the true diseases. Uses prompt_config.json and the generator from
  generate_prompts_bysteps.

Usage:

  # 1. Config-only (input_file from prompt_config.json), with prompts, 2-step
  python ensemble_disease_ranking.py --config prompt_config.json --prompt_steps 2

  # 2. Custom input, top_k 100 per method, final_top_k 50, with prompts
  python ensemble_disease_ranking.py --input_file /path/to/phenotype_disease.json \\
    --prompt_steps 2 --top_k 100 --final_top_k 50 --output_file ensemble_out.json

  # 3. Ranking only, no prompt generation
  python ensemble_disease_ranking.py --config prompt_config.json --prompt_steps 2 --no_prompt

  # 4. Process specific samples (e.g. indices 0,5,10 and 19–24)
  python ensemble_disease_ranking.py --config prompt_config.json --prompt_steps 2 \\
    --sample_indices 0,5,10,19-24 --output_file ensemble_selected.json

  # 5. Limit to first N samples
  python ensemble_disease_ranking.py --config prompt_config.json --prompt_steps 2 --num_samples 10

Key options:
  --input_file      phenotype-disease JSON (or set input_file in config)
  --prompt_steps    2 or 3 (required)
  --top_k           candidates per method (default 100)
  --final_top_k     size of final ranking (default 50)
  --no_prompt       skip prompt generation, only output ranking
  --sample_indices  e.g. 0,5,10 or 0-5,19-24
  --num_samples     process only the first N samples
"""

import json
import sys
import os
import numpy as np
import pandas as pd
from typing import List, Dict, Set, Optional, Tuple
from collections import defaultdict
from scipy import stats
import argparse
from datetime import datetime
import re
from copy import deepcopy

# Import existing generator classes
from generate_prompts_bysteps import PhenotypeToDiseasePromptGenerator, load_config, parse_sample_indices

def _ensemble_get_disease_key(disease_info: Dict) -> Optional[str]:
    if not disease_info:
        return None

    disease_name = disease_info.get("disease_name")
    if disease_name:
        return str(disease_name)

    disease_id = disease_info.get("disease_id")
    if isinstance(disease_id, list):
        return ";".join(str(item) for item in disease_id)
    if disease_id:
        return str(disease_id)

    return None

def _ensemble_sort_tuple(method_name: str, disease: Dict) -> Tuple:
    base_method = method_name.split("+")[0]
    use_ic_weights = "+use_IC_weights" in method_name
    use_frequency_weights = "+use_frequency_weights" in method_name
    use_score = "+use_score" in method_name
    embedding_resort = "+resort" in method_name

    total_assoc = disease.get("total_phenotype_associations")
    if total_assoc is None:
        total_assoc = float("inf")

    if base_method == "embedding":
        matching_count = disease.get("matching_phenotype_count", 0)
        case_similarity = disease.get("case_similarity", 0.0)

        if embedding_resort:
            return (
                -matching_count,
                -case_similarity,
                total_assoc,
            )
        return (
            -case_similarity,
            -matching_count,
            total_assoc,
        )

    matching_count = disease.get("matching_phenotype_count", 0)
    matching_ic = disease.get("matching_phenotype_IC", 0.0)
    matching_weighted = disease.get("matching_phenotype_count_weighted", 0.0)
    score = disease.get("score", 0.0)

    if use_ic_weights:
        return (
            -matching_ic,
            -matching_count,
            total_assoc,
        )
    if use_frequency_weights:
        return (
            -matching_weighted,
            -matching_count,
            total_assoc,
        )
    if use_score:
        return (
            -score,
            -matching_count,
            total_assoc,
        )

    return (
        -matching_count,
        -matching_ic,
        total_assoc,
    )


def _ensemble_sort_diseases(method_name: str, diseases: List[Dict]) -> None:
    # print(diseases)
    diseases.sort(key=lambda disease: _ensemble_sort_tuple(method_name, disease))
    # print(diseases)


class EnsembleDiseaseRanker:
    """Ensemble Disease Ranker"""
    
    def __init__(self, config_file: str = "prompt_config.json"):
        """Initialize ensemble ranker"""
        print("Initializing ensemble disease ranker...")
        
        # Load configuration
        self.config = load_config(config_file)
        
        # Initialize generator
        self.generator = PhenotypeToDiseasePromptGenerator(
            disease_mapping_file=self.config.get("disease_mapping",""),
            phenotype_hpoa_file=self.config.get("phenotype_hpoa",""),
            phenotype_to_genes_file=self.config.get("phenotype_to_genes",""),
            genes_to_phenotype_file=self.config.get("genes_to_phenotype",""),
            embedding_file=self.config.get("embedding_file",""),
            ic_file=self.config.get("ic_file",""),
            case_library=self.config.get("case_library",""),
            disease_descriptions_file=self.config.get("disease_descriptions_file",""),
            config=self.config
        )
        
        # Load phenotype names (from OBO file)
        print("Loading phenotype names...")
        self.generator.load_phenotype_names_from_obo(self.config.get("obo_file",""))
        
        # Load phenotype-disease mappings
        print("Loading phenotype-disease mappings...")
        self.generator.load_phenotype_disease_mappings()
        
        # Enhance phenotype-disease mappings (if gene files are provided)
        # if self.config.get("phenotype_to_genes") or self.config.get("genes_to_phenotype"):
        #     print("Enhancing phenotype-disease mappings with gene information...")
        #     self.generator.enhance_phenotype_disease_mappings_with_genes()
        
        # Merge phenotypes with same disease names
        print("Merging phenotypes with same disease names...")
        self.generator.merge_disease_phenotypes_by_name()
        # Merge disease synonyms by disease name to keep alias lists consistent
        self.generator.merge_disease_synonyms_by_name()
        
        # Load disease descriptions
        print("Loading disease descriptions...")
        self.generator.load_disease_descriptions()

        # unified numeric for disease-phenotype frequency; if multiple freqs, take max
        self.generator.integrate_frequency_numeric()
        
        # Case extraction strategies
        self.extraction_methods = ["overlap+use_IC_weights", "overlap+use_frequency_weights", "embedding+resort"]
        # self.extraction_methods = ["overlap+use_score"]
        
        # Store ranking results for each method
        self.method_rankings = {}

        # Aggregate per-method statistics across samples
        self.method_statistics: Optional[Dict[str, Dict[str, int]]] = None
        
        print("Ensemble disease ranker initialization completed")
    
    def extract_top_diseases_by_method(self, phenotypes: List[str], diseases: List[str], 
                                     method: str, top_k: int = 100, sample_id: int = None) -> List[Dict]:
        """Extract top diseases using specified method"""
        print(f"Using {method} method to extract top{top_k} diseases...")
        
        try:
            # Parse method name to determine whether to use IC weights and embedding_resort
            use_frequency_weights = method.endswith("+use_frequency_weights")
            use_IC_weights = method.endswith("+use_IC_weights")
            use_score = method.endswith("+use_score")
            embedding_resort = method.endswith("+resort")
            
            # Extract base method name
            case_extraction_method = method.replace("+use_IC_weights", "").replace("+use_frequency_weights", "").replace("+resort", "").replace("+use_score", "")
            
            # Call generator method
            prompt_data = self.generator.generate_phenotype_to_disease_prompt(
                phenotypes=phenotypes,
                diseases=diseases,
                top_k=top_k,
                case_extraction_method=case_extraction_method,
                embedding_resort=embedding_resort,
                use_frequency_weights=use_frequency_weights,
                use_IC_weights=use_IC_weights,
                use_score=use_score,
                prompt_steps=self.config.get("prompt_steps", 3),
                # use_samples=True,
                # exclude_sample_id=sample_id,
            )
            
            if prompt_data and "ranked_diseases" in prompt_data:
                ranked_diseases = prompt_data["ranked_diseases"]
                # print("ranked_diseases: ", ranked_diseases)
                print(f"  {method} method successfully extracted {len(ranked_diseases)} diseases")
                # print(ranked_diseases)
                return ranked_diseases
            else:
                print(f"Warning: {method} method did not return ranking results")
                if prompt_data:
                    print(f"  Returned data keys: {list(prompt_data.keys())}")
                return []
                
        except Exception as e:
            print(f"Error: Exception occurred when using {method} method: {e}")
            import traceback
            traceback.print_exc()
            return []
    
    def normalize_rankings(self, rankings_by_method: Dict[str, List[Dict]]) -> Dict[str, Dict[str, float]]:
        """Merge per-method results, re-sort with method rules, and normalize ranks."""
        print("Step 1: Normalizing rankings...")

        if not rankings_by_method:
            print("  Warning: rankings_by_method is empty, skip normalization")
            return {}

        merged_rankings_by_method: Dict[str, List[Dict]] = {}

        for target_method, diseases in rankings_by_method.items():
            merged_list: List[Dict] = []
            existing_keys: Set[str] = set()

            if diseases:
                for disease_info in diseases:
                    disease_key = _ensemble_get_disease_key(disease_info)
                    if not disease_key:
                        continue
                    merged_list.append(dict(disease_info))
                    existing_keys.add(disease_key)

            for other_method, other_diseases in rankings_by_method.items():
                if other_method == target_method:
                    continue
                for disease_info in other_diseases:
                    disease_key = _ensemble_get_disease_key(disease_info)
                    if not disease_key or disease_key in existing_keys:
                        continue
                    merged_list.append(dict(disease_info))
                    existing_keys.add(disease_key)

            _ensemble_sort_diseases(target_method, merged_list)
            merged_rankings_by_method[target_method] = merged_list
            rankings_by_method[target_method] = merged_list

        max_diseases = max(len(diseases) for diseases in merged_rankings_by_method.values())
        print(f"  Using normalization baseline: {max_diseases} diseases (merged maximum)")

        normalized_rankings: Dict[str, Dict[str, float]] = {}
        for method, diseases in merged_rankings_by_method.items():
            if not diseases:
                continue

            method_rankings: Dict[str, float] = {}
            for rank, disease_info in enumerate(diseases, 1):
                disease_key = _ensemble_get_disease_key(disease_info)
                if not disease_key:
                    continue
                method_rankings[disease_key] = rank / max_diseases

            normalized_rankings[method] = method_rankings

        return normalized_rankings
    
    def calculate_z_statistics(self, normalized_rankings: Dict[str, Dict[str, float]]) -> Dict[str, float]:
        """
        Step 2: Calculate Z-statistics to quantify "surprise degree"
        For each disease, calculate its Z-statistic from ranking ratios across all methods
        """
        print("Step 2: Calculating Z-statistics...")
        
        # Collect all disease names
        all_diseases = set()
        for method_rankings in normalized_rankings.values():
            all_diseases.update(method_rankings.keys())
        
        print(f"  Calculating Z-statistics for {len(all_diseases)} unique diseases")
        
        z_statistics = {}
        method_names = list(normalized_rankings.keys())
        n_methods = len(method_names)
        
        for disease_name in all_diseases:
            # Collect ranking ratios for this disease across all methods
            ranking_ratios = []
            
            for method in method_names:
                if disease_name in normalized_rankings[method]:
                    # Disease exists in this method, use actual ranking ratio
                    ranking_ratios.append(normalized_rankings[method][disease_name])
                else:
                    # Disease does not appear in this method, assume it ranks at the end of the method's ranking list
                    # Use the worst possible ranking ratio for this method (1.0, meaning last place)
                    ranking_ratios.append(1.0)
            
            # Calculate Z-statistic (using recursive formula)
            # Formula: V_i = V_{i-1} * r_i / i, where V_0 = 1
            # Final: Z = N! * V_N
            
            # Sort ranking ratios (from small to large)
            sorted_ratios = sorted(ranking_ratios)
            n = len(sorted_ratios)
            
            # Recursively calculate intermediate values V_i
            v = [1.0]  # V_0 = 1
            for i in range(1, n + 1):
                v_i = v[i-1] * sorted_ratios[i-1] / i
                v.append(v_i)
            
            # Calculate factorial of N
            n_factorial = 1
            for i in range(1, n + 1):
                n_factorial *= i
            
            # Calculate final Z-statistic
            z_stat = n_factorial * v[n]

            # Calculate average ranking ratio
            # z_stat = sum(ranking_ratios) / len(ranking_ratios)

            # # score
            # mu = np.mean(ranking_ratios)
            # rank_quality = 1.0 - mu

            # covered = sum(r < 1.0 for r in ranking_ratios)
            # C = covered / len(ranking_ratios)

            # std = np.std(ranking_ratios)
            # S = 1.0 / (1.0 + std)
            
            z_statistics[disease_name] = z_stat
        
        print(f"  Z-statistics calculation completed, range: {min(z_statistics.values()):.6f} - {max(z_statistics.values()):.6f}")
        # print(z_statistics)
        return z_statistics
    
    def rank_diseases_by_z_statistics(self, z_statistics: Dict[str, float], 
                                    disease_info: Dict[str, Dict]) -> List[Dict]:
        """
        Step 3: Final ranking based on Z-statistics
        Disease with smallest Z-statistic becomes rank 1
        """
        print("Step 3: Final ranking based on Z-statistics...")
        
        # Sort by Z-statistic (from small to large)
        sorted_diseases = sorted(z_statistics.items(), key=lambda x: x[1])
        
        final_rankings = []
        for rank, (disease_name, z_stat) in enumerate(sorted_diseases, 1):
            # Get disease ID from disease_info
            disease_id = disease_name  # Default to using name
            if disease_name in disease_info:
                disease_id = disease_info[disease_name].get("disease_id", disease_name)
            
            # Collect associated phenotypes for this disease
            associated_phenotypes = []
            seen_phenotypes = set()
            
            # Search through all method rankings to find matching phenotypes
            for method, original_rankings in self.method_rankings.items():
                for method_disease_info in original_rankings:
                    method_disease_name = method_disease_info.get("disease_name", "")
                    if method_disease_name == disease_name:
                        if "matching_phenotypes" in method_disease_info:
                            for phenotype_with_freq in method_disease_info["matching_phenotypes"]:
                                # Extract HPO ID as deduplication key
                                if '(' in phenotype_with_freq and ')' in phenotype_with_freq:
                                    hpo_id = phenotype_with_freq.split(' (')[0]
                                else:
                                    hpo_id = phenotype_with_freq
                                
                                if hpo_id not in seen_phenotypes:
                                    seen_phenotypes.add(hpo_id)
                                    associated_phenotypes.append(phenotype_with_freq)
                        break
            
            disease_data = {
                "final_rank": rank,
                "z_statistic": z_stat,
                "disease_id": disease_id,
                "disease_name": disease_name,
                "associated_phenotypes": associated_phenotypes
            }
            
            # Add ranking information for each method
            for method, original_rankings in self.method_rankings.items():
                # Search for this disease in original ranking data
                actual_rank = None
                for rank, method_disease_info in enumerate(original_rankings, 1):
                    # Match by disease name
                    method_disease_name = method_disease_info.get("disease_name", "")
                    
                    if method_disease_name == disease_name:
                        actual_rank = rank
                        break
                
                if actual_rank is not None:
                    # Disease exists in this method, record actual rank and ratio
                    # Calculate ranking ratio using the same logic as normalize_rankings
                    method_diseases_count = len(self.method_rankings[method])
                    ranking_ratio = actual_rank / method_diseases_count
                    
                    disease_data[f"{method}_rank"] = actual_rank
                    disease_data[f"{method}_ratio"] = ranking_ratio
                    disease_data[f"{method}_status"] = "found"
                else:
                    # Disease does not exist in this method, mark as not found
                    disease_data[f"{method}_rank"] = None
                    disease_data[f"{method}_ratio"] = None
                    disease_data[f"{method}_status"] = "not_found"
            
            final_rankings.append(disease_data)
        
        print(f"  Final ranking completed, total {len(final_rankings)} diseases")
        return final_rankings
    
    def calculate_hit_rate(self, final_rankings: List[Dict], target_diseases: List[str], 
                          top_k: int = 50) -> Dict:
        """
        Calculate top-k hit rate for true diseases, strictly following the phenotype-to-disease prompt generator logic
        
        Args:
            final_rankings: Final ranking results
            target_diseases: True disease list
            top_k: Calculate top-k hit rate
            
        Returns:
            Dictionary containing hit statistics
        """
        print(f"Calculating top{top_k} hit rate...")
        
        # Get true disease names (for semantic matching) - include all aliases
        all_true_disease_names = []
        for disease_id in target_diseases:
            disease_all_names = self.generator.get_disease_all_names(disease_id)
            all_true_disease_names.extend(disease_all_names)
        
        # Remove duplicates while preserving order
        all_true_disease_names = list(dict.fromkeys(all_true_disease_names))
        
        # Get top-k disease names (for semantic matching)
        ranked_disease_names = [disease["disease_name"] for disease in final_rankings[:top_k]]
        
        # Use method from the prompt generator to calculate best match
        match_result = self.generator.find_best_match_rank_optimized(all_true_disease_names, ranked_disease_names)
        
        # Get similarity threshold (consistent with the prompt generator)
        similarity_threshold = self.config.get('evaluation_config', {}).get('similarity_threshold', 0.8)
        
        # Initialize results
        target_disease_rank = None
        best_matched_disease = None
        best_similarity = 0
        
        if match_result["best_similarity"] > 0:
            best_similarity = match_result["best_similarity"]
            best_matched_disease = match_result["best_match"]
            target_disease_rank = match_result["best_rank"]
        
        # Only count as hit if similarity exceeds threshold
        if best_similarity < similarity_threshold:
            target_disease_rank = None
            best_matched_disease = None
        
        # print(match_result)

        # Get top-k correctness from the optimized matching method
        top1_correct = match_result.get("top1_correct", False)
        top3_correct = match_result.get("top3_correct", False)
        top5_correct = match_result.get("top5_correct", False)
        top10_correct = match_result.get("top10_correct", False)
        top20_correct = match_result.get("top20_correct", False)
        top30_correct = match_result.get("top30_correct", False)

        # print(top1_correct, top5_correct, top10_correct)
        
        # print(all_true_disease_names, ranked_disease_names)
        
        # Build hit statistics (consistent with the prompt generator)
        hit_statistics = {
            "top_k": top_k,
            "target_disease_rank": target_disease_rank,
            "target_disease_matching_count": best_similarity if best_similarity else 0.0,
            "target_disease_similarity": best_similarity if best_similarity else 0.0,
            "best_matched_disease": best_matched_disease,
            "similarity_threshold": similarity_threshold,
            "is_hit": target_disease_rank is not None,
            "true_diseases": target_diseases,
            "true_disease_names": all_true_disease_names,
            # Add top-k hit rates
            "top1_correct": top1_correct,
            "top3_correct": top3_correct,
            "top5_correct": top5_correct,
            "top10_correct": top10_correct,
            "top20_correct": top20_correct,
            "top30_correct": top30_correct
        }
        
        if target_disease_rank is not None:
            print(f"  Hit! Rank: {target_disease_rank}, Similarity: {best_similarity:.3f}")
            print(f"  Best match: {best_matched_disease}")
        else:
            print(f"  Miss (highest similarity: {best_similarity:.3f}, threshold: {similarity_threshold})")
        
        return hit_statistics
    
    def generate_prompt_from_ranked_diseases(self, phenotypes: List[str], diseases: List[str], 
                                           final_rankings: List[Dict], prompt_steps: int = 3) -> Dict:
        """
        Generate prompt based on ensemble ranking results, following the phenotype-to-disease prompt generator logic
        
        Args:
            phenotypes: Phenotype list
            diseases: Target disease list
            final_rankings: Ensemble ranking results
            prompt_steps: Number of prompt steps
            
        Returns:
            Dictionary containing prompt data
        """
        print("Generating prompt...")
        
        # Filter valid diseases (only keep OMIM and ORPHA)
        valid_diseases = [d for d in diseases if d.startswith(('OMIM:', 'ORPHA:'))]
        if not valid_diseases:
            return None
        
        # Get disease names - include all aliases
        all_true_disease_names = []
        for d in valid_diseases:
            disease_all_names = self.generator.get_disease_all_names(d)
            all_true_disease_names.extend(disease_all_names)
        
        # Remove duplicates while preserving order
        all_true_disease_names = list(dict.fromkeys(all_true_disease_names))
        all_true_diseases = valid_diseases
        
        # Get phenotype names and build detailed phenotype information
        phenotype_names = []
        phenotype_disease_mappings = {}
        any_true_disease_in_phenotype_lists = False
        
        for hpo_id in phenotypes:
            phenotype_name = self.generator.get_phenotype_name(hpo_id)
            if phenotype_name:
                full_phenotype_name = f"{phenotype_name} ({hpo_id})"
                phenotype_names.append(full_phenotype_name)
                
                # Get disease associations
                phenotype_diseases = self.generator.get_phenotype_diseases(hpo_id)
                if phenotype_diseases:
                    phenotype_disease_mappings[full_phenotype_name] = phenotype_diseases
                    # Check if any true diseases are in the phenotype list
                    for true_disease_name in all_true_disease_names:
                        if true_disease_name in phenotype_diseases:
                            any_true_disease_in_phenotype_lists = True
                            break
            else:
                phenotype_names.append(f"Unknown Phenotype ({hpo_id})")
        
        if not phenotype_names:
            return None
        
        # Filter samples with only one phenotype
        # if len(phenotype_names) <= 1:
        #     return None
        
        # Sort by number of diseases associated with phenotypes (ascending: fewer diseases = more specific phenotypes)
        phenotype_disease_counts = []
        for hpo_id in phenotypes:
            disease_count = len(self.generator.phenotype_to_diseases.get(hpo_id, set()))
            phenotype_disease_counts.append((hpo_id, disease_count))
        
        phenotype_disease_counts.sort(key=lambda x: x[1])
        sorted_phenotypes = [hpo_id for hpo_id, _ in phenotype_disease_counts]
        phenotypes = sorted_phenotypes
        
        # Build detailed phenotype information
        phenotype_details_list = []
        pheno_num = 0
        for hpo_id in phenotypes:
            pheno_num += 1
            phenotype_name = self.generator.get_phenotype_name(hpo_id)
            synonyms = self.generator.get_phenotype_synonyms(hpo_id)
            definition = self.generator.get_phenotype_definition(hpo_id)
            is_a_names = self.generator.get_phenotype_is_a_names(hpo_id)
            
            # Get disease count
            disease_count = len(self.generator.phenotype_to_diseases.get(hpo_id, set()))

            phenotype_abnormal_category_ids = self.generator.get_phenotype_abnormal_category(hpo_id)
            if phenotype_abnormal_category_ids:
                phenotype_abnormal_categories = []
                for category_id in phenotype_abnormal_category_ids:
                    category_name = self.generator.get_phenotype_name(category_id)
                    if category_name:
                        phenotype_abnormal_categories.append(f"{category_id} ({category_name})")
                    else:
                        phenotype_abnormal_categories.append(category_id)
                phenotype_abnormal_category = " | ".join(phenotype_abnormal_categories)
            else:
                phenotype_abnormal_category = "Unknown category"
            
            # Build natural language description
            # "Reported in {disease_count} diseases" Adding this information to the prompt would cause the model to focus on phenotypes with the largest disease_count, which is not the original purpose
            if is_a_names:
                is_a_text = f"belongs to {', '.join(is_a_names)}"
                if synonyms and definition:
                    detailed_info = f"{pheno_num}. **{phenotype_name} ({hpo_id})**:{phenotype_abnormal_category}, reported in {disease_count} diseases, {is_a_text}, also known as {', '.join(synonyms)}. {definition}"
                elif synonyms:
                    detailed_info = f"{pheno_num}. **{phenotype_name} ({hpo_id})**:{phenotype_abnormal_category}, reported in {disease_count} diseases, {is_a_text}, also known as {', '.join(synonyms)}."
                elif definition:
                    detailed_info = f"{pheno_num}. **{phenotype_name} ({hpo_id})**:{phenotype_abnormal_category}, reported in {disease_count} diseases, {is_a_text}. {definition}"
                else:
                    detailed_info = f"{pheno_num}. **{phenotype_name} ({hpo_id})**:{phenotype_abnormal_category}, reported in {disease_count} diseases, {is_a_text}."
            else:
                if synonyms and definition:
                    detailed_info = f"{pheno_num}. **{phenotype_name} ({hpo_id})**:{phenotype_abnormal_category}, reported in {disease_count} diseases, also known as {', '.join(synonyms)}. {definition}"
                elif synonyms:
                    detailed_info = f"{pheno_num}. **{phenotype_name} ({hpo_id})**:{phenotype_abnormal_category}, reported in {disease_count} diseases, also known as {', '.join(synonyms)}."
                elif definition:
                    detailed_info = f"{pheno_num}. **{phenotype_name} ({hpo_id})**:{phenotype_abnormal_category}, reported in {disease_count} diseases, {definition}"
                else:
                    detailed_info = f"{pheno_num}. **{phenotype_name} ({hpo_id})**:{phenotype_abnormal_category}, reported in {disease_count} diseases, no additional information available."
            
            phenotype_details_list.append(detailed_info)
        
        pheno_details_text = "\n".join(phenotype_details_list) if phenotype_details_list else "No detailed information available"

        # json format
        # phenotype_name:
        #     "hpo_id": hpo_id
        #     "phenotype abnormal category": phenotype_abnormal_category
        #     "synonyms": synonyms
        #     "parent categories": is_a_text
        #     "number of associated diseases": disease_count
        #     "detailed information": definition
        # }
        # reorganize each phenotype into the json format above, then assign to pheno_details_text
        phenotype_details_structured = {}
        for hpo_id in phenotypes:
            phenotype_name = self.generator.get_phenotype_name(hpo_id)
            synonyms = self.generator.get_phenotype_synonyms(hpo_id)
            definition = self.generator.get_phenotype_definition(hpo_id)
            is_a_names = self.generator.get_phenotype_is_a_names(hpo_id)
            disease_count = len(self.generator.phenotype_to_diseases.get(hpo_id, set()))
            if disease_count == 0:
                disease_count = 1
            phenotype_abnormal_category_ids = self.generator.get_phenotype_abnormal_category(hpo_id)
            if phenotype_abnormal_category_ids:
                phenotype_abnormal_categories = []
                for category_id in phenotype_abnormal_category_ids:
                    category_name = self.generator.get_phenotype_name(category_id)
                    if category_name:
                        phenotype_abnormal_categories.append(f"{category_name}")
                    else:
                        phenotype_abnormal_categories.append(category_id)
                phenotype_abnormal_category = " | ".join(phenotype_abnormal_categories)
            else:
                phenotype_abnormal_category = "Unknown category"
            phenotype_details_structured[phenotype_name] = {
                "hpo_id": hpo_id,
                "phenotype abnormal category": phenotype_abnormal_category,
                "synonyms": synonyms if synonyms else [],
                "parent categories": is_a_names if is_a_names else [],
                "number of associated diseases": disease_count,
                "detailed information": definition if definition else ""
            }
        pheno_details_text = json.dumps(phenotype_details_structured, ensure_ascii=False, indent=2)
        
        # pheno_details_text = ", ".join([self.generator.get_phenotype_name(hpo_id) for hpo_id in phenotypes])
       
        # Build candidate disease list text
        ranked_diseases_text = ""
        if final_rankings:
            # First deduplicate disease names, completely following the original script logic
            seen_disease_names = set()
            unique_diseases = []
            
            # Filter out duplicate disease names while preserving order
            # This prevents the same disease from appearing multiple times in the prompt
            for disease_info in final_rankings:
                disease_name = disease_info['disease_name']
                if disease_name not in seen_disease_names:
                    seen_disease_names.add(disease_name)
                    unique_diseases.append(disease_info)
                        
            # Generate candidate disease text using deduplicated disease list
            for i, disease_info in enumerate(unique_diseases, 1):
                disease_name = disease_info['disease_name']
                disease_id = disease_info['disease_id']
                
                # Get matching phenotype information, completely following the original script logic
                # Collect matching phenotype information from rankings of each method, but deduplicate
                matching_phenotypes = []
                seen_phenotypes = set()
                
                for method in self.extraction_methods:
                    method_rankings = self.method_rankings.get(method, [])
                    for method_disease_info in method_rankings:
                        method_disease_id = method_disease_info.get("disease_id")
                        # Check if disease_id matches: string comparison or list membership
                        if (isinstance(disease_id, str) and method_disease_id == disease_id) or \
                           (isinstance(disease_id, list) and method_disease_id in disease_id):
                            if "matching_phenotypes" in method_disease_info:
                                for phenotype_with_freq in method_disease_info["matching_phenotypes"]:
                                    # Extract HPO ID as deduplication key
                                    if '(' in phenotype_with_freq and ')' in phenotype_with_freq:
                                        hpo_id = phenotype_with_freq.split(' (')[0]
                                    else:
                                        hpo_id = phenotype_with_freq
                                    
                                    if hpo_id not in seen_phenotypes:
                                        seen_phenotypes.add(hpo_id)
                                        matching_phenotypes.append(phenotype_with_freq)
                            break
                
                # Process phenotype information, completely following the original script logic
                matching_phenotypes_without_frequency = []
                for phenotype_with_freq in matching_phenotypes:
                    if '(' in phenotype_with_freq and ')' in phenotype_with_freq:
                        hpo_id = phenotype_with_freq.split(' (')[0]
                        matching_phenotypes_without_frequency.append(hpo_id)
                    else:
                        matching_phenotypes_without_frequency.append(phenotype_with_freq)
                
                # Format the disease entry with associated HPO IDs and frequency information
                # renum the id to avoid conflict with the step1 output
                # renum_id = i  # for step3 prompt
                if prompt_steps == 2:
                    renum_id = 10 + i  # for step2 prompt to renumber the id of cases
                else:
                    renum_id = i  # for step3 prompt

                if isinstance(disease_id, list):
                    disease_type = self.generator.disease_types.get(disease_id[0], "")
                else:
                    disease_type = self.generator.disease_types.get(disease_id, "")

                disease_info["matching_phenotypes"] = matching_phenotypes_without_frequency

                # if matching_phenotypes:

                #     # case 1: matching_phenotypes already contains frequency information for embedding method
                #     # case 1.1: prompt without frequency information
                #     # phenotype_text = ", ".join(matching_phenotypes_without_frequency)
                #     phenotype_description = ", ".join([self.generator.get_phenotype_name(hpo_id) for hpo_id in matching_phenotypes_without_frequency])
                    
                #     # case 1.2: prompt with frequency information
                #     # phenotype_text = "; ".join(matching_phenotypes)

                #     # ranked_diseases_text += f"{renum_id}. **{disease_name}** - {phenotype_text}.\n"
                #     if disease_type:
                #       ranked_diseases_text += f"Patient {renum_id} has **{disease_name}**. Disease Category: **{disease_type}**. Description: The patient pending diagnosis exhibits **{phenotype_description}**, which belong to the typical symptoms of this disease.\n" 
                #     else:
                #       ranked_diseases_text += f"Patient {renum_id} has **{disease_name}**. The patient pending diagnosis exhibits **{phenotype_description}**, which belong to the typical symptoms of this disease.\n" 
                    
                #     # case 2: without phenotype information
                #     # ranked_diseases_text += f"{renum_id}. **{disease_name}**.\n"

                #     # case 3: prompt with disease_description
                #     # ranked_diseases_text += f"Patient {renum_id} has **{disease_name}**, {disease_description}\n" 
                # else:

                if isinstance(disease_id, list):
                    disease_description = self.generator.get_disease_description(disease_id[0])
                else:   
                    disease_description = self.generator.get_disease_description(disease_id)

                # ranked_diseases_text += f"{renum_id}. **{disease_name}**.\n"

                if disease_type:
                    ranked_diseases_text += f"Patient {renum_id} has **{disease_name}**. Disease Category: **{disease_type}**. Description: {disease_description}\n"
                else:
                    ranked_diseases_text += f"Patient {renum_id} has **{disease_name}**. Description: {disease_description}\n"

        # json format: organize known patient disease info into the format below, assign to ranked_diseases_text
        # format:
        # {
        #     "Patient {renum_id}": {
        #         "Disease name": disease_name,
        #         "Disease id": disease_id,
        #         "Disease category": disease_type,
        #         "Disease description": disease_description,
        #         "Matching phenotypes": matching_phenotypes,
        #     },
        #     ...
        # }
        
        ranked_diseases_json = {}
        start_id = 1
        for i, d in enumerate(unique_diseases):
            renum_id = i+start_id
            # disease_name = d.get('disease_name', '')
            disease_id = d.get('disease_id', '')
            disease_name = set(self.generator.get_disease_name(_id) for _id in disease_id if isinstance(disease_id, list))
            disease_name = "; ".join(disease_name)
            synonyms = self.generator.get_disease_synonyms(disease_id[0] if isinstance(disease_id, list) else disease_id)
            disease_type = self.generator.disease_types.get(disease_id[0] if isinstance(disease_id, list) else disease_id, "")
            disease_description = self.generator.get_disease_description(disease_id[0] if isinstance(disease_id, list) else disease_id)
            if disease_description == "":
                disease_description = "[Information is missing; please infer based on your memory.]"
            
            # get all phenotypes for disease including freq
            phenotypes = self.generator.disease_to_phenotypes.get(disease_id[0] if isinstance(disease_id, list) else disease_id, [])
            
            # first collect all phenotypes with freq
            phenotype_list = []
            for phenotype in phenotypes:
                freq_key = (phenotype, disease_id[0]) if isinstance(disease_id, list) else (phenotype, disease_id)
                freq_info = self.generator.phenotype_disease_frequency[freq_key]
                frequency_string = freq_info.get('frequency', '')
                # use get_max_frequency_from_frequency_string to convert to number
                if freq_info['frequency_type'] == 'hpo_id':
                    frequency_string = self.generator._convert_hpo_frequency_to_description(frequency_string)
                else:
                    frequency_string = frequency_string
                frequency_numeric = self.generator.get_max_frequency_from_frequency_string(frequency_string)
                
                if frequency_numeric <= 0.17 and frequency_numeric > 0:
                    continue

                frequency_numeric = round(frequency_numeric, 3)
                phenotype_name = self.generator.get_phenotype_name(phenotype)
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

        # build disease category aggregate JSON and list of diseases missing category
        disease_categories_grouped = {}
        missing_disease_categories = []
        try:
            for patient_key, info in ranked_diseases_json.items():
                disease_name = info.get("Disease name")
                disease_id = info.get("Disease id")
                disease_category = info.get("Disease category") or ""
                display_name = f"{disease_name} ({disease_id})" if disease_id else f"{disease_name}"
                if not disease_category:
                    missing_disease_categories.append(display_name)
                else:
                    disease_categories_grouped.setdefault(disease_category, []).append(display_name)
        except Exception:
            disease_categories_grouped = {}
            missing_disease_categories = []
        disease_categories_json_text = json.dumps({
            "Disease Categories": disease_categories_grouped
        }, ensure_ascii=False, indent=2)
        missing_disease_categories_text = ", ".join(missing_disease_categories) if missing_disease_categories else "None"
        
        # Calculate semantic similarity and hit statistics
        similarity_threshold = self.config.get('evaluation_config', {}).get('similarity_threshold', 0.8)
        # Use deduplicated disease list for matching, completely following the original script logic
        ranked_disease_names = [d['disease_name'] for d in unique_diseases]
        
        # Use optimized matching method
        match_result = self.generator.find_best_match_rank_optimized(all_true_disease_names, ranked_disease_names)
        
        target_disease_rank = None
        best_matched_disease = None
        best_similarity = 0
        
        if match_result["best_similarity"] > 0:
            best_similarity = match_result["best_similarity"]
            best_matched_disease = match_result["best_match"]
            target_disease_rank = match_result["best_rank"]
        
        # Only count as hit if similarity exceeds threshold
        if best_similarity < similarity_threshold:
            target_disease_rank = None
            best_matched_disease = None
        

        pheno_list_length = len(phenotypes)
        ranked_diseases_length = len(unique_diseases)

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
        

        # Compute phenotype category grouping JSON and list of phenotypes missing category
        phenotype_categories_grouped = {}
        missing_category_phenotypes = []
        try:
            for pheno_name, info in phenotype_details_structured.items():
                hpo_id = info.get("hpo_id")
                category_name = info.get("phenotype abnormal category") or "Unknown category"
                if not category_name or category_name == "Unknown category":
                    missing_category_phenotypes.append(f"{pheno_name}")
                else:
                    phenotype_categories_grouped.setdefault(category_name, []).append(f"{pheno_name}")
        except Exception:
            phenotype_categories_grouped = {}
            missing_category_phenotypes = []
        phenotype_categories_json_text = json.dumps({
            "Functional/System Categories": phenotype_categories_grouped
        }, ensure_ascii=False, indent=2)
        missing_category_phenotypes_text = ", ".join(missing_category_phenotypes) if missing_category_phenotypes else "None"
        
        # build memory-reasoning step_prompts: split CoT prompt into consecutive dialogue steps
        insertStep1_output = "<INSERTSTEP1_OUTPUT>"
        insertStep2_output = "<INSERTSTEP2_OUTPUT>"
        insertStep3_output = "<INSERTSTEP3_OUTPUT>"

        step_prompts = [
            
            f"""
**Rare Disease Diagnosis Chain-of-Thought (CoT) Prompt**

**Role**:
You are a top-tier rare disease expert and clinical geneticist, skilled in analyzing and inferring potential diagnoses from complex, multi-systemic combinations of phenotypes.

**Task**:
You have received a patient case with {pheno_list_length} detailed phenotypes:
{pheno_details_text}

**Step 1: Phenotype Deconstruction & Categorization**
First, do not treat all input phenotypes as a disordered list. Group them into meaningful clinical categories **(that is by the "phenotype abnormal category" of patient phenotypes)**. Create several core functional/system categories and assign all relevant phenotypes to them:

<-
- Abnormality of the genitourinary system: (if present) Collate all phenotypes related to the genitourinary system. 
- Abnormality of head or neck: (if present) Collate all phenotypes related to the head or neck.
- Abnormality of the eye: (if present) Collate all phenotypes related to the eye.
- Abnormality of the ear: (if present) Collate all phenotypes related to the ear.
- Abnormality of the nervous system: (if present) Collate all phenotypes related to the nervous system.
- Abnormality of the breast: (if present) Collate all phenotypes related to the breast.
- Abnormality of the endocrine system: (if present) Collate all phenotypes related to the endocrine system.
- Abnormality of prenatal development or birth: (if present) Collate all phenotypes related to the prenatal development or birth.
- Growth abnormality: (if present) Collate all phenotypes related to the growth abnormality.
- Abnormality of the integument: (if present) Collate all phenotypes related to the integument.
- Abnormality of the voice: (if present) Collate all phenotypes related to the voice.
- Abnormality of the cardiovascular system: (if present) Collate all phenotypes related to the cardiovascular system.
- Abnormality of blood and blood-forming tissues: (if present) Collate all phenotypes related to the blood and blood-forming tissues.
- Abnormality of metabolism/homeostasis: (if present) Collate all phenotypes related to the metabolism/homeostasis.
- Abnormality of the respiratory system: (if present) Collate all phenotypes related to the respiratory system.
- Neoplasm: (if present) Collate all phenotypes related to the neoplasm.
- Abnormality of the immune system: (if present) Collate all phenotypes related to the immune system.
- Abnormality of the digestive system: (if present) Collate all phenotypes related to the digestive system.
- Constitutional symptom: (if present) Collate all phenotypes related to the constitutional symptom.
- Abnormal cellular phenotype: (if present) Collate all phenotypes related to the abnormal cellular phenotype.
- Abnormality of the musculoskeletal system: (if present) Collate all phenotypes related to the musculoskeletal system.
- Abnormality of limbs: (if present) Collate all phenotypes related to the limbs.
- Abnormality of the thoracic cavity: (if present) Collate all phenotypes related to the thoracic cavity.

Phenotype categorization based on known phenotype classifications (JSON):
{phenotype_categories_json_text}

Phenotypes missing category (need to classification): **{missing_category_phenotypes_text}**. 
If this list is empty, skip this step; otherwise categorize these phenotypes lacking category information into the appropriate functional/system categories in the Step 1 output.
->

**Step 1 output format in json (must use json format, not markdown):**
<-
{{
    "Functional/System Categories":{{
        "[Categories]": [phenotypes1, phenotypes2, ...],
    }}
}}
->
""",

            f"""
**Step 2: Identifying Key Diagnostic Clues from Observable Symptoms**
Focus on **clinically observable symptoms and signs** to identify "anchor phenotypes" or "phenotypic combinations" with the highest diagnostic value. These are often rare, highly specific features that can significantly narrow the differential diagnosis.
The input phenotypes are arranged in descending order of disease specificity, so those toward the bottom of the list are less specific. Do not overemphasize a single low-specificity phenotypes during reasoning.

<-
- Key rule:
  - Focus first on the combinations of observable symptoms and signs that clearly point to a known syndrome.
  - If the patient exhibits **metabolic or laboratory abnormal phenotypes**, consider them as **explanatory evidence** to support or clarify the underlying pathophysiology of the key observable symptoms.

- Instructions for generating Step 2 output:
  - For "anchor clues": Identify the most central and clinically informative features in the patient’s phenotype list. These features are pivotal for narrowing disease categories and are highly specific or distinctive, guiding clinical reasoning.
    - Example: ["phenotype1", "phenotype2"]
  - For "key phenotypic clusters": Identify groups of phenotypes that form a recognizable clinical pattern or appear together in a highly specific way, which suggest a particular disease category.
    - Example: ["[Description of key phenotypic cluster 1]": "A classic feature of [Disease Category], such as [disease1, disease2, ...]", "[Description of key phenotypic cluster 2]": "Indicates [Disease Category], such as [disease1, disease2, ...]", ..., "[Description of key phenotypic cluster N]": "Suggests [Disease Category], such as [disease1, disease2, ...]"],  
  - Please use specific combinations of phenotypes to represent each key phenotypic cluster.
  - Please provide three different decisions/verdicts with distinct focuses and record them under Judgment 1, Judgment 2, and Judgment 3.
->

**Step 2 output format in json (must use json format, not markdown):**
<-
{{
    Judgment 1:{{
        "anchor clues": [anchor phenotypes list],
        "key phenotypic clusters": [
            "[Description of key phenotypic cluster 1]": "A classic feature of [Disease Category], such as [disease1, disease2, ...]",
            "[Description of key phenotypic cluster 2]": "Indicates [Disease Category], such as [disease1, disease2, ...]",
            ...
            "[Description of key phenotypic cluster N]": "Suggests [Disease Category], such as [disease1, disease2, ...]",
        ]
    }},
    Judgment 2:{{
        "anchor clues": [anchor phenotypes list],
        "key phenotypic clusters": [
            "[Description of key phenotypic cluster 1]": "A classic feature of [Disease Category], such as [disease1, disease2, ...]",
            "[Description of key phenotypic cluster 2]": "Indicates [Disease Category], such as [disease1, disease2, ...]",
            ...
            "[Description of key phenotypic cluster N]": "Suggests [Disease Category], such as [disease1, disease2, ...]",
        ]
    }},
    Judgment 3:{{
        "anchor clues": [anchor phenotypes list],
        "key phenotypic clusters": [
            "[Description of key phenotypic cluster 1]": "A classic feature of [Disease Category], such as [disease1, disease2, ...]",
            "[Description of key phenotypic cluster 2]": "Indicates [Disease Category], such as [disease1, disease2, ...]",
            ...
            "[Description of key phenotypic cluster N]": "Suggests [Disease Category], such as [disease1, disease2, ...]",
        ]
    }}
}}
->

""",

            f"""
**Step 3: Formulating a Core Clinical Hypothesis**
Based on the key phenotypic clusters from the **Step 2 output**, formulate a unifying hypothesis and a concurrent diseases hypothesis. This hypothesis must not focus on a single organ system. Instead, it must define the underlying pathophysiological process that you believe connects the most significant, yet seemingly disparate, clinical clusters.

<-
- Instructions for generating Step 3 output: Your hypothesis must be framed from a systemic perspective. Identify the fundamental disease process that can independently explain the different key phenotypic clusters. The diagnostic challenge is to identify an entity within this class of disease that can account for these core features as well as other significant involvements.

- **Hypothesis Template Example (in json format):**

  "Unifying Hypothesis":
    [
      - The most parsimonious explanation is a single systemic disease process, that is [Description of a Broad Disease Category].
      - This hypothesis is based on the pathophysiological mechanism of [X] causing both [Phenotypic Cluster A] and [Phenotypic Cluster B] simultaneously.
      - Supporting Evidence: Key findings that strongly support this single diagnosis include: [List the most compelling evidence that fits this one hypothesis].
      - Inconsistencies/Red Flags: [List clinical findings that are difficult to explain or contradict this hypothesis].
    ]

  "Concurrent Diseases Hypothesis":
    [
      - An alternative, and often more probable, explanation is that the patient has two or more distinct conditions occurring concurrently. This hypothesis is proposed to resolve the inconsistencies identified in the Unifying Hypothesis.
      - The most likely combination of conditions is:
        - Primary Condition: A [Description of a Broad Disease Category] to explain the most severe or defining features, such as [Phenotypic Cluster A].
        - Co-existing Condition(s): A [Description of a Broad Disease Category] to explain the remaining key finding(s) of [Phenotypic Cluster B].
      - Supporting Evidence: This combination is plausible because [Explain why this combination makes clinical sense].
      - Inconsistencies/Red Flags: [List aspects this combination fails to explain].
    ]
->

**Step 3 output format in json (must use json format, not markdown):**
<-
{{
    "Unifying Hypothesis": "[Description of Unifying Hypothesis according to the above template]",
    "Concurrent Diseases Hypothesis": "[Description of Concurrent Diseases Hypothesis according to the above template]",
}}
->
""",

            f"""
**Insert Step 1: Generating Candidate Rare Diseases**

Taking into account the above **hypothesis** and the overlap of **anchor phenotypes** and **key phenotypic clusters** between the known patients and the patient to be diagnosed, systematically identify and select 20 rare diseases that are most likely to be relevant as candidate diagnoses.
**Note: You must strictly select exactly 20 candidate rare diseases as potential diagnoses—no more, no less.**

**Insert Step 1 output format in json (must use json format, not markdown):**

{{
    "20 Candidate Rare Diseases": {{
        "[Rare Disease Name]": [List the key patient phenotypes this disease can explain],
    }}
}}

""",

            f"""
**Insert Step 2: Generating and Ranking the Differential Diagnosis**
Taking into account the above **hypothesis** and the **anchor phenotypes** and **key phenotypic clusters** of the patient to be diagnosed, systematically identify and select 10 rare diseases that are most likely to be relevant as candidate diagnoses from the **20 Candidate Rare Diseases** below:
{insertStep1_output}

Output Template:

1. [A specific rare disease name] (Primary Diagnosis)
  - Matched: [List key patient phenotypes this disease explains.]
  - Unmatched: [List key patient phenotypes this disease does not explain.]
  - Reasoning: [Explain why this is the best fit.]

2. [A specific rare disease name] (Secondary Diagnosis)
  - Matched: [List phenotypes explained.]
  - Unmatched: [List phenotypes this disease does not explain.]
  - Reasoning: [Explain why this is a plausible but secondary option.]

(Continue this format for the top 10 diagnoses.)
**Note: The diagnosis list needs to be sorted by matching degree from high to low.**

**Insert Step 2 output format in json (must use json format, not markdown):**
{{
    "FINAL ANSWER":{{
        "[INDEX. Rare Disease Name]": {{
            "Matched": [List key patient phenotypes this disease explains.],
            "Unmatched": [List key patient phenotypes this disease does not explain.],
            "Reasoning": "[Rationale for ranking the disease of diagnosis]",
        }},
    }}
}}

""",

            f"""
**Insert Step 3 <>**
You are provided with a set of patients, each exhibiting clinical manifestations characteristic of a specific disease. Detailed information for each patient is as follows:
{ranked_diseases_text}{insertStep2_output}

Summarize the key symptoms or combinations of phenotypes that are essential and specific for the diagnosis of each disease, and provide your output in the following JSON format (provide only valid JSON, not markdown format):
Please use standard HPO phenotype names to describe the key symptoms of each disease.

**Insert Step 3 output format in json (must use json format, not markdown):**
{{
    "[Patient INDEX. Disease Name]": "[List the key symptoms or combinations of phenotypes that are essential and specific for the diagnosis of this disease. Please use standard HPO phenotype names.]"
}}
""",

            f"""
**Step 4: Extracting Candidate Rare Diseases from Known Patients**
The following known patients (no ranking implied) have overlapping phenotypes with the patient to be diagnosed, among which:
{ranked_diseases_text}{insertStep2_output}

The key features of the diseases manifested by the known patients are as follows:
{insertStep3_output}

Taking into account the above **hypothesis** and the overlap of **anchor phenotypes** and **key phenotypic clusters** between the known patients and the patient to be diagnosed, systematically identify and select 20 rare diseases that are most likely to be relevant as candidate diagnoses.
The Candidate Rare Diseases must select from the disease of the **known patients**.
**Note: You must strictly select exactly 20 candidate rare diseases as potential diagnoses—no more, no less.**

**Step 4 output format in json (must use json format, not markdown):**
<-
{{
    "20 Candidate Rare Diseases": {{
        "[Disease Name (Patient INDEX)]": ["Reason for selecting this as a candidate diagnose"],
    }}
}}
->
""",       

            f"""
**Step 5: Generating and Ranking the Differential Diagnosis**
Please analyze the 20 candidate rare diseases listed above **in greater detail**, and **sort the 20 Candidate Rare Diseases** to get the 10 most likely diagnoses.

Output Template:

1. [Rare Disease Name] (Primary Diagnosis)
  - Matched: [List key patient phenotypes this disease explains.]
  - Unmatched: [List key patient phenotypes this disease does not explain.]
  - Reasoning: [Explain why this is the best fit.]

2. [Rare Disease Name] (Secondary Diagnosis)
  - Matched: [List phenotypes explained.]
  - Unmatched: [List phenotypes NOT explained.]
  - Reasoning: [Explain why this is a plausible but secondary option.]

(Continue this format for the top 10 diagnoses.)
**Note: The diagnosis list needs to be sorted by matching degree from high to low.**

**Step 5 output format in json (must use json format, not markdown):**
{{
    "FINAL ANSWER":{{
        "[INDEX. Rare Disease Name]": {{
            "Matched": [List key patient phenotypes this disease explains.],
            "Unmatched": [List key patient phenotypes this disease does not explain.],
            "Reasoning": "[Rationale for ranking the disease of diagnosis]",
        }},
    }}
}}
"""
        ]

        step_prompts = []
      
        # Generate answer format
        if best_matched_disease:
            answer = f"{best_matched_disease}"
        else:
            answer = f"{all_true_disease_names[0] if all_true_disease_names else 'Unknown Disease'}"
        
        # Calculate prompt length
        step1_prompt_length = len(step1_prompt) if step1_prompt else 0
        step2_prompt_length = len(step2_prompt) if step2_prompt else 0
        step3_prompt_length = len(step3_prompt) if step3_prompt else 0
        
        # Get phenotypes associated with true diseases
        all_true_disease_associated_phenotypes = []
        for hpo_id in phenotypes:
            phenotype_diseases = self.generator.phenotype_to_diseases.get(hpo_id, set())
            for true_disease_id in all_true_diseases:
                if true_disease_id in phenotype_diseases:
                    all_true_disease_associated_phenotypes.append(hpo_id)
                    break
        
        return {
            "task_type": "phenotype_to_disease",
            "prompt_steps": prompt_steps,
            "step1_prompt": step1_prompt or "",
            "step2_prompt": step2_prompt or "",
            "step3_prompt": step3_prompt or "",
            "step_prompts": step_prompts,
            "memory_mode": "full",
            "step1_prompt_length": step1_prompt_length,
            "step2_prompt_length": step2_prompt_length,
            "step3_prompt_length": step3_prompt_length,
            "answer": answer,
            "answer_length": len(answer),
            "phenotypes": phenotype_names,
            "pheno_details_text": pheno_details_text,
            "true_diseases": all_true_disease_names,
            "true_disease_ids": all_true_diseases,
            "target_disease_in_phenotype_lists": any_true_disease_in_phenotype_lists,
            "target_disease_associated_phenotypes": all_true_disease_associated_phenotypes,
            "ranked_diseases": unique_diseases,
            "ranked_disease_names": ranked_disease_names,
            "target_disease_rank": target_disease_rank,
            "target_disease_matching_count": best_similarity if best_similarity else 0.0,
            "target_disease_similarity": best_similarity if best_similarity else 0.0,
            "best_matched_disease": best_matched_disease,
            "top_k": len(final_rankings),
            "case_extraction_method": "ensemble",
            "matching_method": "optimized_advanced_with_splitting",
            "similarity_threshold": similarity_threshold,
            "total_diseases_processed": len(all_true_disease_names) + len(ranked_disease_names)
        }

    def process_sample(self, phenotypes: List[str], diseases: List[str], 
                      top_k: int = 100, final_top_k: int = 50, generate_prompt: bool = True, 
                      prompt_steps: int = 3, sample_id: int = None) -> Dict:
        """Process a single sample, return ensemble ranking results and optional prompt"""
        print(f"\nProcessing sample: {len(phenotypes)} phenotypes, {len(diseases)} target diseases")
        
        # Use different methods to extract top diseases
        rankings_by_method = {}
        all_disease_info = {}
        
        for method in self.extraction_methods:
            print(f"\nUsing {method} method...")
            method_rankings = self.extract_top_diseases_by_method(
                phenotypes, diseases, method, top_k, sample_id
            )
            # print(method_rankings)
            if method_rankings:
                rankings_by_method[method] = method_rankings
                # Collect disease information
                for disease_info in method_rankings:
                    disease_name = disease_info.get("disease_name", "")
                    if disease_name:
                        all_disease_info[disease_name] = disease_info
                
                print(f"  {method} method extracted {len(method_rankings)} diseases")
            else:
                print(f"  {method} method did not extract any diseases")
        
        if not rankings_by_method:
            print("Error: All methods failed to extract diseases")
            return {"error": "All methods failed to extract diseases"}
        
        # Store method rankings for final results
        self.method_rankings = rankings_by_method
        
        # Step 1: Normalize rankings
        normalized_rankings = self.normalize_rankings(rankings_by_method)
        
        # Step 2: Calculate Z-statistics
        z_statistics = self.calculate_z_statistics(normalized_rankings)
        
        # Step 3: Final ranking
        final_rankings = self.rank_diseases_by_z_statistics(z_statistics, all_disease_info)
        
        # Calculate hit rate
        hit_statistics = self.calculate_hit_rate(final_rankings, diseases, final_top_k)

        # Update per-method aggregate statistics
        if self.method_statistics is not None:
            for method in self.extraction_methods:
                method_stats = self.method_statistics.setdefault(method, {
                    "total_samples": 0,
                    "hit_count": 0
                })
                method_stats["total_samples"] += 1

                method_rankings = rankings_by_method.get(method, [])
                if method_rankings:
                    method_hit_statistics = self.calculate_hit_rate(method_rankings, diseases, final_top_k)
                else:
                    method_hit_statistics = {"is_hit": False}

                if method_hit_statistics.get("is_hit"):
                    method_stats["hit_count"] += 1
        
        # Basic results
        result = {
            "sample_info": {
                "phenotypes": phenotypes,
                "target_diseases": diseases
            },
            "final_rankings": final_rankings[:final_top_k],
            "is_hit": hit_statistics["is_hit"],
            "target_disease_rank": hit_statistics["target_disease_rank"],
            "target_disease_similarity": hit_statistics["target_disease_similarity"],
            "top1_correct": hit_statistics["top1_correct"],
            "top3_correct": hit_statistics["top3_correct"],
            "top5_correct": hit_statistics["top5_correct"],
            "top10_correct": hit_statistics["top10_correct"],
            "top20_correct": hit_statistics["top20_correct"],
            "top30_correct": hit_statistics["top30_correct"]
        }
        
        # If prompt generation is needed, add prompt-related data
        if generate_prompt:
            prompt_data = self.generate_prompt_from_ranked_diseases(
                phenotypes, diseases, final_rankings[:final_top_k], 
                prompt_steps
            )
            if prompt_data:
                # Merge prompt data into results
                result.update(prompt_data)
            else:
                print("Warning: Unable to generate prompt data")
        
        print(f"\nEnsemble ranking completed:")
        print(f"  Found {len(final_rankings)} diseases")
        print(f"  Top {final_top_k} candidate diseases:")
        for i, disease in enumerate(final_rankings[:final_top_k], 1):
            print(f"    {i}. {disease['disease_name']} (Z={disease['z_statistic']:.6f})")
        
        return result
    
    def process_dataset(self, input_file: str, output_file: str, num_samples: int = None, 
                       sample_indices: List[int] = None, top_k: int = 100, final_top_k: int = 50, 
                       generate_prompt: bool = True, prompt_steps: int = 3) -> List[Dict]:
        """Process entire dataset, save after processing each sample"""
        print(f"\nStarting to process dataset: {input_file}")
        
        # Load input data
        input_data = self.generator.load_input_data(input_file)
        
        # Apply sample filtering based on sample_indices or num_samples
        if sample_indices:
            # Filter by specific sample indices (0-based from parse_sample_indices function)
            filtered_data = []
            for idx in sample_indices:
                if 0 <= idx < len(input_data):
                    filtered_data.append(input_data[idx])
                else:
                    print(f"Warning: Sample index {idx} is out of range (0-{len(input_data)-1})")
            input_data = filtered_data
            print(f"Processing {len(input_data)} samples from indices: {sample_indices}")
        elif num_samples:
            input_data = input_data[:num_samples]
            print(f"Processing first {len(input_data)} samples")
        else:
            print(f"Processing all {len(input_data)} samples")
        
        # Initialize per-method statistics
        self.method_statistics = {method: {
            "total_samples": 0,
            "hit_count": 0
        } for method in self.extraction_methods}

        results = []
        for i, sample in enumerate(input_data, 1):
            print(f"\n{'='*50}")
            print(f"Processing sample {i}/{len(input_data)}")
            
            phenotypes = sample[0] if len(sample) > 0 else []
            diseases = sample[1] if len(sample) > 1 else []
            
            if not phenotypes or not diseases:
                print(f"Skipping sample {i}: missing phenotype or disease information")
                continue
            
            result = self.process_sample(phenotypes, diseases, top_k, final_top_k, generate_prompt, prompt_steps, sample_id=i)
            
            if "error" in result:
                print(f"Sample {i} processing failed: {result['error']}")
                continue
            
            # Add sample ID
            result["sample_id"] = i
            
            # If prompt is generated, reorganize data structure to match the prompt generator output format
            if generate_prompt and "task_type" in result:
                # Create format consistent with the prompt generator
                ordered_result = {
                    'sample_id': result['sample_id'],
                    'task_type': result['task_type'],
                    'step1_prompt': result['step1_prompt'],
                    'step2_prompt': result['step2_prompt'],
                    'step3_prompt': result['step3_prompt'],
                    'step_prompts': result['step_prompts'],
                    'memory_mode': result['memory_mode'],
                    'step1_prompt_length': result['step1_prompt_length'],
                    'step2_prompt_length': result['step2_prompt_length'],
                    'step3_prompt_length': result['step3_prompt_length'],
                    'answer': result['answer'],
                    'answer_length': result['answer_length'],
                    'phenotypes': result['phenotypes'],
                    'pheno_details_text': result['pheno_details_text'],
                    'true_diseases': result['true_diseases'],
                    'true_disease_ids': result['true_disease_ids'],
                    'target_disease_in_phenotype_lists': result['target_disease_in_phenotype_lists'],
                    'target_disease_associated_phenotypes': result['target_disease_associated_phenotypes'],
                    'ranked_diseases': result['ranked_diseases'],
                    'ranked_disease_names': result['ranked_disease_names'],
                    'target_disease_rank': result['target_disease_rank'],
                    'target_disease_matching_count': result['target_disease_matching_count'],
                    'target_disease_similarity': result['target_disease_similarity'],
                    'best_matched_disease': result['best_matched_disease'],
                    'top_k': result['top_k'],
                    'case_extraction_method': result['case_extraction_method'],
                    'matching_method': result['matching_method'],
                    'similarity_threshold': result['similarity_threshold'],
                    'total_diseases_processed': result['total_diseases_processed'],
                    # Preserve ensemble ranking specific information
                    'ensemble_info': {
                        'final_rankings': result['final_rankings'],
                        'is_hit': result['is_hit'],
                        'target_disease_rank_ensemble': result['target_disease_rank'],
                        'target_disease_similarity_ensemble': result['target_disease_similarity'],
                        'top1_correct': result.get('top1_correct', False),
                        'top3_correct': result.get('top3_correct', False),
                        'top5_correct': result.get('top5_correct', False),
                        'top10_correct': result.get('top10_correct', False),
                        'top20_correct': result.get('top20_correct', False),
                        'top30_correct': result.get('top30_correct', False)
                    }
                }
                results.append(ordered_result)
            else:
                # If no prompt is generated, maintain original format
                results.append(result)
            
            # Save after processing each sample
            self.save_results_incremental(output_file, results, i, len(input_data))
        
        return results
    
    def save_results_incremental(self, output_file: str, results: List[Dict], 
                                current_sample: int, total_samples: int):
        """Incrementally save results, save after processing each sample"""
        if not results:
            return
        
        # Calculate current overall statistics
        overall_statistics = self.calculate_overall_statistics(results)
        
        # Build disease matching summary, completely following original script format
        disease_matching_summary = []
        for result in results:
            if "ensemble_info" in result:
                ensemble_info = result["ensemble_info"]
                summary_item = {
                    "sample_id": result["sample_id"],
                    "true_diseases": result.get("true_diseases", []),
                    "best_matched_disease": result.get("best_matched_disease"),
                    "similarity_score": result.get("target_disease_similarity", 0.0),
                    "rank": result.get("target_disease_rank"),
                    "is_matched": ensemble_info.get("is_hit", False)
                }
                disease_matching_summary.append(summary_item)
        
        # Build output data, completely following original script format
        output_data = {
            "metadata": {
                "timestamp": datetime.now().isoformat(),
                "progress": {
                    "current_sample": current_sample,
                    "total_samples": total_samples,
                    "completed_samples": len(results)
                },
                "overall_statistics": overall_statistics
            },
            "disease_matching_summary": disease_matching_summary,
            "samples": results
        }
        
        # Save to file (ensure dir exists)
        out_dir = os.path.dirname(output_file)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, ensure_ascii=False, indent=2)
        
        print(f"  Results saved to: {output_file} (sample {current_sample}/{total_samples})")
    
    def calculate_overall_statistics(self, results: List[Dict]) -> Dict:
        """Calculate overall statistics"""
        if not results:
            return None
        
        total_samples = len(results)
        
        # Calculate hit statistics
        ranks = []
        hit_count = 0
        
        # Calculate top-k hit rates using the new fields
        top1_hits = 0
        top3_hits = 0
        top5_hits = 0
        top10_hits = 0
        top20_hits = 0
        top30_hits = 0
        
        for result in results:
            # Check new data structure (containing prompt data)
            if "ensemble_info" in result:
                # New format: get hit information from ensemble_info
                ensemble_info = result["ensemble_info"]
                if ensemble_info.get("is_hit", False):
                    hit_count += 1
                    target_rank = ensemble_info.get("target_disease_rank_ensemble")
                    if target_rank is not None:
                        ranks.append(target_rank)
                
                # Count top-k hits using the new fields
                if ensemble_info.get("top1_correct", False):
                    top1_hits += 1
                if ensemble_info.get("top3_correct", False):
                    top3_hits += 1
                if ensemble_info.get("top5_correct", False):
                    top5_hits += 1
                if ensemble_info.get("top10_correct", False):
                    top10_hits += 1
                if ensemble_info.get("top20_correct", False):
                    top20_hits += 1
                if ensemble_info.get("top30_correct", False):
                    top30_hits += 1
            else:
                # Old format: get directly from result
                if result.get("is_hit", False):
                    hit_count += 1
                    target_rank = result.get("target_disease_rank")
                    if target_rank is not None:
                        ranks.append(target_rank)
                
                # Count top-k hits using the new fields
                if result.get("top1_correct", False):
                    top1_hits += 1
                if result.get("top3_correct", False):
                    top3_hits += 1
                if result.get("top5_correct", False):
                    top5_hits += 1
                if result.get("top10_correct", False):
                    top10_hits += 1
                if result.get("top20_correct", False):
                    top20_hits += 1
                if result.get("top30_correct", False):
                    top30_hits += 1
        
        # Calculate statistics
        hit_rate = (hit_count / total_samples * 100) if total_samples > 0 else 0
        top1_hit_rate = (top1_hits / total_samples * 100) if total_samples > 0 else 0
        top3_hit_rate = (top3_hits / total_samples * 100) if total_samples > 0 else 0
        top5_hit_rate = (top5_hits / total_samples * 100) if total_samples > 0 else 0
        top10_hit_rate = (top10_hits / total_samples * 100) if total_samples > 0 else 0
        top20_hit_rate = (top20_hits / total_samples * 100) if total_samples > 0 else 0
        top30_hit_rate = (top30_hits / total_samples * 100) if total_samples > 0 else 0
        avg_rank = sum(ranks) / len(ranks) if ranks else 0
        median_rank = sorted(ranks)[len(ranks)//2] if ranks else 0
        
        # Build overall statistics
        overall_statistics = {
            "total_samples": total_samples,
            "hit_count": hit_count,
            "hit_rate": hit_rate,
            "top1_hits": top1_hits,
            "top1_hit_rate": top1_hit_rate,
            "top3_hits": top3_hits,
            "top3_hit_rate": top3_hit_rate,
            "top5_hits": top5_hits,
            "top5_hit_rate": top5_hit_rate,
            "top10_hits": top10_hits,
            "top10_hit_rate": top10_hit_rate,
            "top20_hits": top20_hits,
            "top20_hit_rate": top20_hit_rate,
            "top30_hits": top30_hits,
            "top30_hit_rate": top30_hit_rate,
            "avg_rank": avg_rank,
            "median_rank": median_rank,
            "total_ranked": len(ranks),
            "unranked_samples": total_samples - len(ranks)
        }
        
        if ranks:
            overall_statistics.update({
                "best_rank": min(ranks),
                "worst_rank": max(ranks)
            })

        if self.method_statistics:
            method_summary = {}
            for method, stats in self.method_statistics.items():
                total = stats.get("total_samples", 0)
                if total <= 0:
                    continue
                hit_count = stats.get("hit_count", 0)
                method_summary[method] = {
                    "total_samples": total,
                    "hit_count": hit_count,
                    "hit_rate": (hit_count / total * 100) if total else 0
                }
            if method_summary:
                overall_statistics["method_hit_rates"] = method_summary
        
        return overall_statistics


def main():
    """Main function"""
    parser = argparse.ArgumentParser(description="Ensemble disease ranking script")
    parser.add_argument("--config", type=str, default="prompt_config.json", 
                       help="Configuration file path")
    parser.add_argument("--input_file", type=str, 
                       help="Input JSON file path (phenotype-disease data)")
    parser.add_argument("--num_samples", type=int, default=None,
                       help="Number of samples to process (None means process all samples)")
    parser.add_argument("--sample_indices", type=str, default=None,
                       help="Comma-separated list of sample indices to process (0-based, e.g., '0,4,9,19-24')")
    parser.add_argument("--top_k", type=int, default=100,
                       help="Number of top diseases extracted by each method")
    parser.add_argument("--final_top_k", type=int, default=50,
                       help="Number of top diseases in final output")
    parser.add_argument("--output_file", type=str, default=None,
                       help="Output file path")
    parser.add_argument("--generate_prompt", action="store_true", default=True,
                       help="Whether to generate prompt (default: True)")
    parser.add_argument("--no_prompt", action="store_true", default=False,
                       help="Do not generate prompt, only perform ensemble ranking")
    parser.add_argument("--prompt_steps", type=int, required=True,
                        help="Number of prompt steps (must be specified)")

    
    args = parser.parse_args()
    
    # Parse sample indices if provided
    sample_indices = None
    if args.sample_indices:
        try:
            sample_indices = parse_sample_indices(args.sample_indices)
            print(f"Parsed sample indices: {sample_indices}")
        except ValueError as e:
            print(f"Error parsing sample indices '{args.sample_indices}': {e}")
            sys.exit(1)
    
    # Determine whether to generate prompt
    generate_prompt = args.generate_prompt and not args.no_prompt
    
    # Load configuration
    config = load_config(args.config)
    
    # Use command line arguments or values from configuration file
    input_file = args.input_file or config.get("input_file")
    if not input_file:
        print("Error: input file path not specified (use --input_file or config input_file)")
        sys.exit(1)
    
    # Generate output file name: 指定 > base_path/pho2disease/prompt > prompt/
    if not args.output_file:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        base = config.get("base_path") or ""
        fname = f"ensemble_disease_ranking_with_prompts_{args.prompt_steps}steps_with_top_{args.final_top_k}_{ts}.json" if generate_prompt else f"ensemble_disease_ranking_results_{args.prompt_steps}steps_with_top_{args.final_top_k}_{ts}.json"
        if base:
            args.output_file = os.path.join(base, "pho2disease", "prompt", fname)
        else:
            args.output_file = os.path.join("prompt", fname)
    
    # Initialize ensemble ranker
    ranker = EnsembleDiseaseRanker(args.config)
    
    # Process dataset (save after processing each sample)
    results = ranker.process_dataset(
        input_file=input_file,
        output_file=args.output_file,
        num_samples=args.num_samples,
        sample_indices=sample_indices,
        top_k=args.top_k,
        final_top_k=args.final_top_k,
        generate_prompt=generate_prompt,
        prompt_steps=args.prompt_steps
    )
    
    # Finally save complete metadata information
    if results:
        # Calculate final overall statistics
        overall_statistics = ranker.calculate_overall_statistics(results)
        
        # Read current file content
        with open(args.output_file, 'r', encoding='utf-8') as f:
            current_data = json.load(f)
        
        # Update metadata
        current_data["metadata"].update({
            "config_file": args.config,
            "input_file": input_file,
            "parameters": {
                "num_samples": args.num_samples,
                "sample_indices": args.sample_indices,
                "top_k": args.top_k,
                "final_top_k": args.final_top_k,
                "generate_prompt": generate_prompt,
                "prompt_steps": args.prompt_steps
            },
            "extraction_methods": ranker.extraction_methods,
            "overall_statistics": overall_statistics
        })
        
        # Update disease matching summary
        disease_matching_summary = []
        for result in results:
            if "ensemble_info" in result:
                ensemble_info = result["ensemble_info"]
                summary_item = {
                    "sample_id": result["sample_id"],
                    "true_diseases": result.get("true_diseases", []),
                    "best_matched_disease": result.get("best_matched_disease"),
                    "similarity_score": result.get("target_disease_similarity", 0.0),
                    "rank": result.get("target_disease_rank"),
                    "is_matched": ensemble_info.get("is_hit", False)
                }
                disease_matching_summary.append(summary_item)
        
        current_data["disease_matching_summary"] = disease_matching_summary
        
        # Save final results (ensure dir exists)
        out_dir = os.path.dirname(args.output_file)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
        with open(args.output_file, 'w', encoding='utf-8') as f:
            json.dump(current_data, f, ensure_ascii=False, indent=2)
        
        print(f"\n{'='*60}")
        print("Final overall hit rate statistics")
        print(f"{'='*60}")
        print(f"Total samples: {overall_statistics['total_samples']}")
        print(f"Hit samples: {overall_statistics['hit_count']}")
        print(f"Hit rate: {overall_statistics['hit_rate']:.2f}%")
        print(f"Top1 hits: {overall_statistics['top1_hits']} ({overall_statistics['top1_hit_rate']:.2f}%)")
        print(f"Top3 hits: {overall_statistics['top3_hits']} ({overall_statistics['top3_hit_rate']:.2f}%)")
        print(f"Top5 hits: {overall_statistics['top5_hits']} ({overall_statistics['top5_hit_rate']:.2f}%)")
        print(f"Top10 hits: {overall_statistics['top10_hits']} ({overall_statistics['top10_hit_rate']:.2f}%)")
        print(f"Top20 hits: {overall_statistics['top20_hits']} ({overall_statistics['top20_hit_rate']:.2f}%)")
        print(f"Top30 hits: {overall_statistics['top30_hits']} ({overall_statistics['top30_hit_rate']:.2f}%)")
        print(f"Average rank: {overall_statistics['avg_rank']:.1f}")
        print(f"Median rank: {overall_statistics['median_rank']}")
        if 'best_rank' in overall_statistics:
            print(f"Best rank: {overall_statistics['best_rank']}")
            print(f"Worst rank: {overall_statistics['worst_rank']}")
        
        if generate_prompt:
            print(f"\nGenerated prompt data for {len(results)} samples")
    
    print(f"\nFinal results saved to: {args.output_file}")
    print(f"Processed {len(results)} samples")


if __name__ == "__main__":
    main()
