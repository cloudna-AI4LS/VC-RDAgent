#!/usr/bin/env python3
"""
Phenotype-driven disease ranking utilities.

This module provides a Python implementation of the phenotype-based disease ranking
logic used by LIRICAL. Given a cohort of diseases with phenotype annotations and a
set of observed/excluded phenotypes for an individual, the ranking algorithm computes
likelihood ratios for each disease and returns them ordered by descending post-test
probability.
"""

from __future__ import annotations

import logging
import math
from collections import defaultdict
from dataclasses import dataclass, field
from enum import Enum
import json
from pathlib import Path
import re
from typing import Dict, Iterable, Iterator, List, Mapping, Optional, Sequence, Set, Tuple


FREQUENCY_TERM_TO_VALUE: Dict[str, float] = {
    "HP:0040280": 1.0,      # Obligate (100%)
    "HP:0040281": 0.895,    # Very frequent (80-99%)
    "HP:0040282": 0.545,    # Frequent (30-79%)
    "HP:0040283": 0.17,     # Occasional (5-29%)
    "HP:0040284": 0.025,    # Very rare (1-4%)
    "HP:0040285": 0.0,      # Excluded (0%)
}

CASE_DATABASE_FREQUENCY_TEXT_TO_VALUE: Dict[str, float] = {
    "OBLIGATE": FREQUENCY_TERM_TO_VALUE["HP:0040280"],
    "VERY FREQUENT": FREQUENCY_TERM_TO_VALUE["HP:0040281"],
    "FREQUENT": FREQUENCY_TERM_TO_VALUE["HP:0040282"],
    "OCCASIONAL": FREQUENCY_TERM_TO_VALUE["HP:0040283"],
    "VERY RARE": FREQUENCY_TERM_TO_VALUE["HP:0040284"],
    "EXCLUDED": FREQUENCY_TERM_TO_VALUE["HP:0040285"],
}

CASE_DATABASE_UNKNOWN_TOKENS = {"UNKNOWN", "NOT REPORTED", "N/A", "NA"}
CASE_DATABASE_DEFAULT_FREQUENCY = 1.0

logger = logging.getLogger(__name__)


def _load_config_file(config_path: Optional[str | Path] = None) -> Dict:
    """
    Load configuration from JSON file with parameter substitution.
    
    Args:
        config_path: Path to the configuration file. If None, looks for prompt_config.json
                     in the same directory as this script.
    
    Returns:
        Dictionary containing the configuration, with {base_path} parameters resolved.
    """
    if config_path is None:
        config_path = Path(__file__).resolve().parent / "prompt_config.json"
    else:
        config_path = Path(config_path)
    
    if not config_path.exists():
        logger.debug("Configuration file not found at %s; using default paths.", config_path)
        return {}
    
    try:
        with config_path.open(encoding="utf-8") as handle:
            config = json.load(handle)
        
        # Replace {base_path} only (use replace to avoid breaking other placeholders like ${timestamp})
        if "base_path" in config:
            base_path = config["base_path"]
            for key, value in config.items():
                if isinstance(value, str) and "{base_path}" in value:
                    config[key] = value.replace("{base_path}", base_path)
                elif isinstance(value, dict):
                    # Handle nested dictionaries (e.g., orphanet_files, output_config)
                    for nested_key, nested_value in value.items():
                        if isinstance(nested_value, str) and "{base_path}" in nested_value:
                            config[key][nested_key] = nested_value.replace("{base_path}", base_path)
        
        return config
    except (json.JSONDecodeError, IOError) as exc:
        logger.warning("Error loading configuration file %s: %s; using default paths.", config_path, exc)
        return {}


def _get_path_from_config(config_key: str, config_path: Optional[str | Path] = None, fallback_relative_path: Optional[str] = None) -> Path:
    """
    Get a file path from configuration file.
    
    Args:
        config_key: Key in the configuration file to look for.
        config_path: Path to the configuration file. If None, looks for prompt_config.json
                     in the same directory as this script.
        fallback_relative_path: Optional relative path to construct from base_path if config_key is not found.
                                Format: "subdir/filename.ext"
    
    Returns:
        Path to the file.
    
    Raises:
        ValueError: If configuration file is missing or path is not configured.
    """
    config = _load_config_file(config_path)
    
    if not config:
        if config_path is None:
            config_path = Path(__file__).resolve().parent / "prompt_config.json"
        raise ValueError(
            f"Configuration file not found at {config_path}. "
            f"Please ensure prompt_config.json exists and contains '{config_key}' configuration."
        )
    
    if config_key in config:
        file_path = Path(config[config_key])
        return file_path
    
    # If config_key is not in config, try to construct from base_path
    if fallback_relative_path and "base_path" in config:
        base_path = config["base_path"]
        file_path = Path(base_path) / fallback_relative_path
        logger.info(
            "%s not found in config, constructing from base_path: %s",
            config_key,
            file_path
        )
        return file_path
    
    raise ValueError(
        f"'{config_key}' not found in configuration file and base_path is not available. "
        f"Please add '{config_key}' to prompt_config.json"
    )


def _get_case_database_path(config_path: Optional[str | Path] = None) -> Path:
    """
    Get the case database path from configuration file.
    
    Args:
        config_path: Path to the configuration file. If None, looks for prompt_config.json
                     in the same directory as this script.
    
    Returns:
        Path to the case database file.
    
    Raises:
        ValueError: If configuration file is missing or case_database_file is not configured.
    """
    return _get_path_from_config(
        "case_database_file",
        config_path,
        fallback_relative_path="general_cases/phenotype_disease_case_database.json"
    )


def _get_obo_path(config_path: Optional[str | Path] = None) -> Path:
    """
    Get the hp.obo file path from configuration file.

    Args:
        config_path: Path to the configuration file. If None, looks for prompt_config.json
                     in the same directory as this script.

    Returns:
        Path to the hp.obo file.

    Raises:
        ValueError: If configuration file is missing or obo_file is not configured.
    """
    return _get_path_from_config(
        "obo_file",
        config_path,
        fallback_relative_path="hpo_annotations/hp.obo"
    )


def _get_hpoa_path(config_path: Optional[str | Path] = None) -> Path:
    """
    Get the phenotype.hpoa file path from configuration file.
    
    Args:
        config_path: Path to the configuration file. If None, looks for prompt_config.json
                     in the same directory as this script.
    
    Returns:
        Path to the phenotype.hpoa file.
    
    Raises:
        ValueError: If configuration file is missing or phenotype_hpoa is not configured.
    """
    return _get_path_from_config(
        "phenotype_hpoa",
        config_path,
        fallback_relative_path="hpo_annotations/phenotype.hpoa"
    )


# Get default paths from config
CASE_DATABASE_DEFAULT_PATH = _get_case_database_path()
OBO_DEFAULT_PATH = _get_obo_path()
HPOA_DEFAULT_PATH = _get_hpoa_path()

PHENOTYPIC_ABNORMALITY_TERM_ID = "HP:0000118"

HPO_FREQUENCY_PATTERN = re.compile(r"HP:\d{7}")
RATIO_PATTERN = re.compile(r"(?P<numerator>\d+)\s*/\s*(?P<denominator>\d+)")
PERCENTAGE_PATTERN = re.compile(r"(?P<value>\d+(?:\.\d+)?)%")

DEFAULT_DISEASE_DATABASES = {"OMIM", "ORPHA", "DECIPHER"} # {"OMIM", "ORPHA", "DECIPHER"}


@dataclass(frozen=True)
class Term:
    """Simple representation of an ontology term."""

    id: str
    name: str


class OntologyGraph:
    """Lightweight directed acyclic graph for representing HPO relationships."""

    def __init__(self, parents_map: Mapping[str, Iterable[str]]):
        self._parents: Dict[str, Set[str]] = {
            term: set(parents) for term, parents in parents_map.items()
        }
        self._nodes: Set[str] = set(self._parents.keys())
        for parents in self._parents.values():
            self._nodes.update(parents)
        self._children: Dict[str, Set[str]] = {term: set() for term in self._nodes}
        for term, parents in self._parents.items():
            for parent in parents:
                self._children.setdefault(parent, set()).add(term)
        # Ensure every node exists in both parent and child maps.
        for term in list(self._nodes):
            self._parents.setdefault(term, set())
            self._children.setdefault(term, set())

    @property
    def nodes(self) -> Set[str]:
        return set(self._nodes)

    def add_term(self, term: str, parents: Optional[Iterable[str]] = None) -> None:
        parents_set = set(parents or [])
        self._nodes.add(term)
        self._parents.setdefault(term, set()).update(parents_set)
        self._children.setdefault(term, set())
        for parent in parents_set:
            self._nodes.add(parent)
            self._parents.setdefault(parent, set())
            self._children.setdefault(parent, set()).add(term)

    def ensure_term(self, term: str) -> None:
        if term not in self._nodes:
            self.add_term(term, [])

    def get_parents(self, term: str) -> Set[str]:
        return set(self._parents.get(term, set()))

    def get_children(self, term: str) -> Set[str]:
        return set(self._children.get(term, set()))

    def get_ancestors(self, term: str, include_self: bool = False) -> Set[str]:
        visited: Set[str] = set()
        stack: List[str] = [term]
        while stack:
            current = stack.pop()
            if current in visited:
                continue
            visited.add(current)
            stack.extend(self._parents.get(current, ()))
        if not include_self:
            visited.discard(term)
        return visited

    def is_ancestor_of(self, ancestor: str, descendant: str) -> bool:
        if ancestor == descendant:
            return True
        return ancestor in self.get_ancestors(descendant, include_self=False)

    def extend_with_ancestors(self, term: str, include_self: bool, sink: Set[str]) -> None:
        if include_self:
            sink.add(term)
        stack: List[str] = [term]
        while stack:
            current = stack.pop()
            for parent in self._parents.get(current, ()):
                if parent in sink:
                    continue
                sink.add(parent)
                stack.append(parent)


class MinimalOntology:
    """Minimal ontology abstraction mirroring the Java implementation."""

    def __init__(
        self,
        parents_map: Mapping[str, Iterable[str]],
        term_labels: Optional[Mapping[str, str]] = None,
        alias_map: Optional[Mapping[str, str]] = None,
    ):
        self._graph = OntologyGraph(parents_map)
        self._term_labels: Dict[str, str] = dict(term_labels or {})
        self._alias_to_primary: Dict[str, str] = dict(alias_map or {})
        # Ensure the phenotypic abnormality root exists.
        self._graph.add_term(PHENOTYPIC_ABNORMALITY_TERM_ID, [])
        self._term_labels.setdefault(PHENOTYPIC_ABNORMALITY_TERM_ID, "Phenotypic abnormality")
        for term in self._graph.nodes:
            self._term_labels.setdefault(term, term)

    def ensure_term(self, term_id: str, parents: Optional[Iterable[str]] = None, label: Optional[str] = None) -> None:
        self._graph.add_term(term_id, parents)
        if label is not None:
            self._term_labels[term_id] = label

    def graph(self) -> OntologyGraph:
        return self._graph

    def term_for_term_id(self, term_id: str) -> Optional[Term]:
        name = self._term_labels.get(term_id)
        if name is None:
            return None
        return Term(term_id, name)

    def non_obsolete_term_ids(self) -> Set[str]:
        return self._graph.nodes

    def canonical_term_id(self, term_id: str) -> str:
        return self._alias_to_primary.get(term_id, term_id)

    def has_term(self, term_id: str) -> bool:
        canonical = self.canonical_term_id(term_id)
        return canonical in self._term_labels


@dataclass(frozen=True)
class PhenotypeAnnotation:
    term_id: str
    frequency: float


@dataclass
class DiseaseModel:
    """Container describing a disease and its phenotype annotations."""

    disease_id: str
    name: str
    annotations: Mapping[str, float]
    absent_annotations: Set[str] = field(default_factory=set)
    pretest_probability: Optional[float] = None

    def annotation_items(self) -> Iterator[Tuple[str, float]]:
        return iter(self.annotations.items())

    def annotations_iter(self) -> Iterator[PhenotypeAnnotation]:
        for term_id, frequency in self.annotations.items():
            yield PhenotypeAnnotation(term_id, frequency)

    def absent_annotation_ids(self) -> Set[str]:
        return set(self.absent_annotations)

    def is_directly_annotated_to(self, term_id: str) -> bool:
        return term_id in self.annotations

    def annotation_frequency(self, term_id: str) -> Optional[float]:
        return self.annotations.get(term_id)

    def is_annotated_to(self, term_id: str, ontology: MinimalOntology) -> bool:
        if self.is_directly_annotated_to(term_id):
            return True
        graph = ontology.graph()
        for annotated in self.annotations:
            if graph.is_ancestor_of(term_id, annotated):
                return True
        for annotated in self.annotations:
            if graph.is_ancestor_of(annotated, term_id):
                return True
        return False


@dataclass(frozen=True)
class Term2Freq:
    term_id: str
    frequency: float

    def non_root_common_ancestor(self) -> bool:
        return self.term_id != PHENOTYPIC_ABNORMALITY_TERM_ID


class InducedDiseaseGraph:
    """Pre-computed data structure for LIRICAL phenotype likelihood ratio evaluation."""

    def __init__(
        self,
        disease: DiseaseModel,
        term_to_frequency: Mapping[str, float],
        negative_induced_graph: Set[str],
    ):
        self._disease = disease
        self._term_to_frequency = dict(term_to_frequency)
        self._negative_induced_graph = set(negative_induced_graph)

    @staticmethod
    def create(disease: DiseaseModel, ontology: MinimalOntology) -> InducedDiseaseGraph:
        term_frequencies: Dict[str, float] = {}
        graph = ontology.graph()
        for annotation in disease.annotations_iter():
            frequency = annotation.frequency
            stack: List[Tuple[str, int]] = [(annotation.term_id, 0)]
            seen: Set[str] = set()
            while stack:
                term_id, distance = stack.pop()
                if term_id in seen:
                    continue
                seen.add(term_id)
                adjusted_frequency = frequency / math.pow(2.0, distance)  # was 10.0: gentler decay along ancestor chain
                term_frequencies[term_id] = max(adjusted_frequency, term_frequencies.get(term_id, 0.0))
                for parent in graph.get_parents(term_id):
                    if parent == PHENOTYPIC_ABNORMALITY_TERM_ID:
                        continue
                    stack.append((parent, distance + 1))

        negative_induced_graph: Set[str] = set()
        for absent_term in disease.absent_annotation_ids():
            graph.extend_with_ancestors(absent_term, True, negative_induced_graph)

        return InducedDiseaseGraph(disease, term_frequencies, negative_induced_graph)

    def is_exact_excluded_match(self, term_id: str) -> bool:
        return term_id in self._negative_induced_graph

    def term_frequency(self, term_id: str) -> Optional[float]:
        return self._term_to_frequency.get(term_id)

    @property
    def disease(self) -> DiseaseModel:
        return self._disease

    def get_closest_ancestor(self, term_id: str, ontology: MinimalOntology) -> Term2Freq:
        queue: List[str] = [term_id]
        visited: Set[str] = set()
        graph = ontology.graph()
        while queue:
            current = queue.pop(0)
            if current in visited:
                continue
            visited.add(current)
            frequency = self._term_to_frequency.get(current)
            if frequency is not None:
                return Term2Freq(current, frequency)
            queue.extend(graph.get_parents(current))
        return Term2Freq(PHENOTYPIC_ABNORMALITY_TERM_ID, 1.0)


class LrMatchType(Enum):
    EXACT_MATCH = "exact_match"
    QUERY_TERM_SUBCLASS_OF_DISEASE_TERM = "query_term_subclass_of_disease_term"
    QUERY_TERM_PRESENT_BUT_EXCLUDED_IN_DISEASE = "query_term_present_but_excluded_in_disease"
    DISEASE_TERM_SUBCLASS_OF_QUERY = "disease_term_subclass_of_query"
    NON_ROOT_COMMON_ANCESTOR = "non_root_common_ancestor"
    UNUSUAL_BACKGROUND_FREQUENCY = "unusual_background_frequency"
    EXCLUDED_QUERY_TERM_NOT_PRESENT_IN_DISEASE = "excluded_query_term_not_present_in_disease"
    EXCLUDED_QUERY_TERM_EXCLUDED_IN_DISEASE = "excluded_query_term_excluded_in_disease"
    EXCLUDED_QUERY_TERM_PRESENT_IN_DISEASE = "excluded_query_term_present_in_disease"
    NO_MATCH_BELOW_ROOT = "no_match_below_root"


@dataclass(frozen=True)
class LrWithExplanation:
    query_term: str
    matching_term: Optional[str]
    match_type: LrMatchType
    lr: float
    explanation: str

    def __lt__(self, other: LrWithExplanation) -> bool:
        return self.lr < other.lr


class LrWithExplanationFactory:
    def __init__(self, ontology: MinimalOntology):
        self._ontology = ontology

    def create(self, query_term: str, match_type: LrMatchType, lr: float) -> LrWithExplanation:
        return self._create(query_term, query_term, match_type, lr)

    def _create(self, query_term: str, matching_term: Optional[str], match_type: LrMatchType, lr: float) -> LrWithExplanation:
        explanation = self._build_explanation(query_term, matching_term, match_type, lr)
        return LrWithExplanation(query_term, matching_term, match_type, lr, explanation)

    def create_with_match(self, query_term: str, matching_term: str, match_type: LrMatchType, lr: float) -> LrWithExplanation:
        return self._create(query_term, matching_term, match_type, lr)

    def _build_label(self, term_id: str) -> str:
        term = self._ontology.term_for_term_id(term_id)
        label = term.name if term else "UNKNOWN"
        return f"{label}[{term_id}]"

    def _build_explanation(self, query_term: str, matching_term: Optional[str], match_type: LrMatchType, lr: float) -> str:
        query_label = self._build_label(query_term)
        match_label = self._build_label(matching_term) if matching_term else query_label
        safe_lr = lr if lr > 0 else 1e-300
        log10_lr = math.log10(safe_lr)
        if match_type == LrMatchType.EXACT_MATCH:
            return f"E:{query_label}[{log10_lr:.3f}]"
        if match_type == LrMatchType.QUERY_TERM_SUBCLASS_OF_DISEASE_TERM:
            return f"Q<D:{query_label}<{match_label}[{log10_lr:.3f}]"
        if match_type == LrMatchType.DISEASE_TERM_SUBCLASS_OF_QUERY:
            return f"D<Q:{match_label}<{query_label}[{log10_lr:.3f}]"
        if match_type == LrMatchType.NON_ROOT_COMMON_ANCESTOR:
            return f"Q~D:{query_label}~{match_label}[{log10_lr:.3f}]"
        if match_type == LrMatchType.UNUSUAL_BACKGROUND_FREQUENCY:
            return f"U:{query_label}[{log10_lr:.3f}]"
        if match_type == LrMatchType.EXCLUDED_QUERY_TERM_EXCLUDED_IN_DISEASE:
            return f"XX:{query_label}[{log10_lr:.3f}]"
        if match_type == LrMatchType.EXCLUDED_QUERY_TERM_NOT_PRESENT_IN_DISEASE:
            return f"XA:{query_label}[{log10_lr:.3f}]"
        if match_type == LrMatchType.EXCLUDED_QUERY_TERM_PRESENT_IN_DISEASE:
            return f"XP:{query_label}[{log10_lr:.3f}]"
        return f"NM:{query_label}[{log10_lr:.3f}]"


class PhenotypeLikelihoodRatio:
    DEFAULT_BACKGROUND_FREQUENCY = 1.0 / 10_000.0
    EXCLUDED_IN_DISEASE_BUT_PRESENT_IN_QUERY_PROBABILITY = 1.0 / 1000.0
    EXCLUDED_IN_DISEASE_AND_EXCLUDED_IN_QUERY_PROBABILITY = 1000.0
    FALSE_NEGATIVE_OBSERVATION_OF_PHENOTYPE_PROB = 0.01
    DEFAULT_TERM_FREQUENCY = 1.0
    DEFAULT_FALSE_POSITIVE_NO_COMMON_ORGAN_PROBABILITY = 0.01
    NO_MATCH_BELOW_ROOT_LR = 0.5  # was 0.01: avoid one NO_MATCH killing composite_lr

    def __init__(self, ontology: MinimalOntology, diseases: Sequence[DiseaseModel]):
        self._ontology = ontology
        self._explanation_factory = LrWithExplanationFactory(ontology)
        self._hpo_term_to_overall_frequency = self._initialize_frequency_map(ontology, diseases)

    def lr_for_observed_term(self, query_tid: str, idg: InducedDiseaseGraph) -> LrWithExplanation:
        disease = idg.disease
        graph = self._ontology.graph()
        query_ancestors = graph.get_ancestors(query_tid, include_self=True)
        if any(absent in query_ancestors for absent in disease.absent_annotation_ids()):
            return self._explanation_factory.create(
                query_tid,
                LrMatchType.QUERY_TERM_PRESENT_BUT_EXCLUDED_IN_DISEASE,
                self.EXCLUDED_IN_DISEASE_BUT_PRESENT_IN_QUERY_PROBABILITY,
            )

        if disease.is_directly_annotated_to(query_tid):
            frequency = disease.annotation_frequency(query_tid)
            if frequency is None:
                frequency = self.DEFAULT_TERM_FREQUENCY
            denominator = self._get_background_frequency(query_tid)
            lr = frequency / denominator
            return self._explanation_factory.create_with_match(
                query_tid,
                query_tid,
                LrMatchType.EXACT_MATCH,
                lr,
            )

        maximum_frequency_of_descendant_term = 0.0
        is_ancestor = False
        disease_matching_term: Optional[str] = None
        for annotation in disease.annotations_iter():
            if graph.is_ancestor_of(query_tid, annotation.term_id):
                maximum_frequency_of_descendant_term = max(maximum_frequency_of_descendant_term, annotation.frequency)
                disease_matching_term = annotation.term_id
                is_ancestor = True
        if is_ancestor and disease_matching_term is not None:
            denominator = self._get_background_frequency(query_tid)
            lr = maximum_frequency_of_descendant_term / denominator
            return self._explanation_factory.create_with_match(
                query_tid,
                disease_matching_term,
                LrMatchType.DISEASE_TERM_SUBCLASS_OF_QUERY,
                lr,
            )

        has_non_root_common_ancestor = False
        max_f = 0.0
        best_match_term_id: Optional[str] = None
        denominator_for_non_root_common_ancestor = self._get_background_frequency(query_tid)
        for annotation in disease.annotations_iter():
            if graph.is_ancestor_of(annotation.term_id, query_tid):
                proportional_frequency = self._get_proportion_in_children(query_tid, annotation.term_id)
                query_frequency = annotation.frequency
                f_value = proportional_frequency * query_frequency
                if f_value > max_f:
                    best_match_term_id = annotation.term_id
                    max_f = f_value
                    has_non_root_common_ancestor = True
        if has_non_root_common_ancestor and best_match_term_id is not None:
            lr = max(
                max_f,
                self._no_common_organ_probability(query_tid),
            ) / denominator_for_non_root_common_ancestor
            return self._explanation_factory.create_with_match(
                query_tid,
                best_match_term_id,
                LrMatchType.QUERY_TERM_SUBCLASS_OF_DISEASE_TERM,
                lr,
            )

        term_freq = idg.get_closest_ancestor(query_tid, self._ontology)
        if term_freq.non_root_common_ancestor():
            numerator = term_freq.frequency
            denominator = self._get_background_frequency(term_freq.term_id)
            lr = max(
                self.DEFAULT_FALSE_POSITIVE_NO_COMMON_ORGAN_PROBABILITY,
                numerator / denominator,
            )
            return self._explanation_factory.create_with_match(
                query_tid,
                term_freq.term_id,
                LrMatchType.NON_ROOT_COMMON_ANCESTOR,
                lr,
            )

        return self._explanation_factory.create(
            query_tid,
            LrMatchType.NO_MATCH_BELOW_ROOT,
            self.NO_MATCH_BELOW_ROOT_LR,
        )

    def lr_for_excluded_term(self, query_tid: str, idg: InducedDiseaseGraph) -> LrWithExplanation:
        disease = idg.disease
        if idg.is_exact_excluded_match(query_tid):
            return self._explanation_factory.create(
                query_tid,
                LrMatchType.EXCLUDED_QUERY_TERM_EXCLUDED_IN_DISEASE,
                self.EXCLUDED_IN_DISEASE_AND_EXCLUDED_IN_QUERY_PROBABILITY,
            )

        background_frequency = self._get_background_frequency(query_tid)
        if background_frequency > 0.99:
            logger.error(
                "Unusually high background frequency calculated for %s (%.5f).",
                query_tid,
                background_frequency,
            )
            return self._explanation_factory.create(
                query_tid,
                LrMatchType.UNUSUAL_BACKGROUND_FREQUENCY,
                1.0,
            )

        if not disease.is_annotated_to(query_tid, self._ontology):
            lr = 1.0 / (1.0 - background_frequency)
            return self._explanation_factory.create(
                query_tid,
                LrMatchType.EXCLUDED_QUERY_TERM_NOT_PRESENT_IN_DISEASE,
                lr,
            )

        frequency = self._get_frequency_of_term_in_disease_with_annotation_propagation(query_tid, disease)
        excluded_frequency = max(
            self.FALSE_NEGATIVE_OBSERVATION_OF_PHENOTYPE_PROB,
            1.0 - frequency,
        )
        lr = excluded_frequency / (1.0 - background_frequency)
        return self._explanation_factory.create(
            query_tid,
            LrMatchType.EXCLUDED_QUERY_TERM_PRESENT_IN_DISEASE,
            lr,
        )

    def _get_background_frequency(self, term_id: str) -> float:
        background_frequency = self._hpo_term_to_overall_frequency.get(term_id)
        if background_frequency is None:
            logger.error(
                "Background frequency missing for term %s. Using default.",
                term_id,
            )
            return self.DEFAULT_BACKGROUND_FREQUENCY
        return max(background_frequency, self.DEFAULT_BACKGROUND_FREQUENCY)

    def _no_common_organ_probability(self, term_id: str) -> float:
        f_value = self._hpo_term_to_overall_frequency.get(
            term_id,
            self.DEFAULT_FALSE_POSITIVE_NO_COMMON_ORGAN_PROBABILITY,
        )
        min_prob = 0.002
        max_prob = 0.10
        factor = (max_prob - min_prob) / (max_prob - self.DEFAULT_FALSE_POSITIVE_NO_COMMON_ORGAN_PROBABILITY)
        false_positive_penalty = min_prob + (f_value - self.DEFAULT_FALSE_POSITIVE_NO_COMMON_ORGAN_PROBABILITY) * factor
        return false_positive_penalty * f_value

    def _get_descendant_depth(self, ancestor: str, descendant: str) -> Optional[int]:
        """Shortest path length from ancestor to descendant (downward). Returns None if not descendant."""
        if ancestor == descendant:
            return 0
        from collections import deque
        graph = self._ontology.graph()
        q = deque([(ancestor, 0)])
        visited: Set[str] = {ancestor}
        while q:
            cur, d = q.popleft()
            for ch in graph.get_children(cur):
                if ch == descendant:
                    return d + 1
                if ch not in visited:
                    visited.add(ch)
                    q.append((ch, d + 1))
        return None

    def _get_proportion_in_children(self, query_tid: str, disease_tid: str) -> float:
        if query_tid == disease_tid:
            return 1.0
        children = self._ontology.graph().get_children(disease_tid)
        if not children:
            return 0.0
        if query_tid in children:
            return 1.0 / float(len(children))
        depth = self._get_descendant_depth(disease_tid, query_tid)
        if depth is None or depth < 2:
            return 0.0
        base = 1.0 / float(len(children))
        decay = 0.5 ** (depth - 1)  # grandchild 0.5, great‑grandchild 0.25, ...
        return base * decay

    def _get_frequency_of_term_in_disease_with_annotation_propagation(self, query_tid: str, disease: DiseaseModel) -> float:
        max_frequency = 0.0
        graph = self._ontology.graph()
        for annotation in disease.annotations_iter():
            annotated_term = annotation.term_id
            frequency = annotation.frequency
            if annotated_term == query_tid:
                max_frequency = max(max_frequency, frequency)
                continue
            ancestors = graph.get_ancestors(annotated_term, include_self=False)
            if query_tid in ancestors:
                max_frequency = max(max_frequency, frequency)
        return max_frequency

    @staticmethod
    def _initialize_frequency_map(ontology: MinimalOntology, diseases: Sequence[DiseaseModel]) -> Dict[str, float]:
        mp: Dict[str, float] = {term_id: 0.0 for term_id in ontology.non_obsolete_term_ids()}
        seen_diseases: Set[str] = set()
        graph = ontology.graph()
        for disease in diseases:
            if disease.disease_id in seen_diseases:
                continue
            seen_diseases.add(disease.disease_id)
            update_map: Dict[str, float] = {}
            for annotation in disease.annotations_iter():
                freq = annotation.frequency
                term_id = annotation.term_id
                update_map[term_id] = max(freq, update_map.get(term_id, 0.0))
                for ancestor in graph.get_ancestors(term_id, include_self=False):
                    update_map[ancestor] = max(freq, update_map.get(ancestor, 0.0))
            for term_id, value in update_map.items():
                mp[term_id] = mp.get(term_id, 0.0) + value
        if not seen_diseases:
            return mp
        return {term_id: value / len(seen_diseases) for term_id, value in mp.items()}


def _curie_from_iri(iri: str) -> str:
    if iri.startswith("http://purl.obolibrary.org/obo/"):
        frag = iri.rsplit("/", 1)[-1]
    else:
        frag = iri
    if frag.startswith("HP_"):
        return frag.replace("HP_", "HP:")
    if "_" in frag and ":" not in frag:
        prefix, suffix = frag.split("_", 1)
        return f"{prefix}:{suffix}"
    return frag


def _parse_obo_value(line: str) -> str:
    """Extract value from an OBO tag-value line; strip trailing ! comment."""
    if ":" not in line:
        return ""
    _, rest = line.split(":", 1)
    rest = rest.strip()
    if "!" in rest:
        rest = rest.split("!")[0].strip()
    return rest


def load_minimal_ontology_from_hp_json(obo_path: Optional[str | Path] = None) -> MinimalOntology:
    """
    Load HPO ontology from an OBO file (hp.obo) and return a MinimalOntology instance.

    Parses [Term] blocks: id, name, is_a (parents), alt_id (aliases). Obsolete terms are skipped.

    Args:
        obo_path: Path to hp.obo. If None, uses obo_file from prompt_config.json.
    """
    if obo_path is None:
        obo_path = OBO_DEFAULT_PATH
    path = Path(obo_path)
    if not path.exists():
        raise FileNotFoundError(f"OBO file not found: {path}")

    term_labels: Dict[str, str] = {}
    parents_map: Dict[str, Set[str]] = defaultdict(set)
    alias_map: Dict[str, str] = {}

    with path.open(encoding="utf-8") as f:
        in_term = False
        cur_id: Optional[str] = None
        cur_name: Optional[str] = None
        cur_is_a: List[str] = []
        cur_alt_id: List[str] = []
        cur_obsolete = False

        def flush_term() -> None:
            nonlocal cur_id, cur_name, cur_is_a, cur_alt_id, cur_obsolete
            if cur_id is None or cur_obsolete:
                cur_id = None
                cur_name = None
                cur_is_a = []
                cur_alt_id = []
                cur_obsolete = False
                return
            term_labels[cur_id] = cur_name or cur_id
            parents_map[cur_id].update(cur_is_a)
            for a in cur_alt_id:
                if a:
                    alias_map[a] = cur_id
            cur_id = None
            cur_name = None
            cur_is_a = []
            cur_alt_id = []
            cur_obsolete = False

        for line in f:
            line = line.rstrip("\n\r")
            s = line.strip()
            if s == "":
                flush_term()
                in_term = False
                continue
            if s.startswith("["):
                flush_term()
                in_term = s == "[Term]"
                continue
            if not in_term:
                continue
            if s.startswith("id:"):
                cur_id = _parse_obo_value(s)
            elif s.startswith("name:"):
                cur_name = _parse_obo_value(s)
            elif s.startswith("is_a:"):
                cur_is_a.append(_parse_obo_value(s))
            elif s.startswith("alt_id:"):
                cur_alt_id.append(_parse_obo_value(s))
            elif s.startswith("is_obsolete:") and _parse_obo_value(s).lower() == "true":
                cur_obsolete = True
        flush_term()

    return MinimalOntology(parents_map, term_labels, alias_map)


def _parse_frequency_ratio(is_negated: bool,
                           raw_frequency: str,
                           salvage_negated: bool,
                           cohort_size: int) -> Tuple[int, int]:
    """
    Parse the HPOA frequency field into a numerator/denominator pair following the logic in
    `HpoDiseaseLoaderDefault#parseFrequency`.
    """
    freq = (raw_frequency or "").strip()

    if not freq:
        numerator = 0 if is_negated else 1
        return numerator, 1

    token = freq.split(";", 1)[0].strip()

    if HPO_FREQUENCY_PATTERN.fullmatch(token):
        value = FREQUENCY_TERM_TO_VALUE.get(token)
        if value is None:
            raise ValueError(f"Unrecognized HPO frequency term {token}")
        numerator = 0 if is_negated else round(value * cohort_size)
        denominator = cohort_size
        return numerator, denominator

    ratio_match = RATIO_PATTERN.fullmatch(token)
    if ratio_match:
        denominator = int(ratio_match.group("denominator"))
        numerator_value = int(ratio_match.group("numerator"))
        if denominator == 0:
            denominator = cohort_size
        if is_negated:
            if numerator_value == 0 and salvage_negated:
                numerator = 0
            else:
                numerator = max(0, denominator - numerator_value)
        else:
            numerator = numerator_value
        numerator = min(max(numerator, 0), denominator)
        return numerator, denominator

    percentage_match = PERCENTAGE_PATTERN.fullmatch(token)
    if percentage_match:
        percentage = float(percentage_match.group("value"))
        numerator = round(percentage * cohort_size / 100.0)
        denominator = cohort_size
        numerator = min(max(numerator, 0), denominator)
        if is_negated and not salvage_negated:
            numerator = max(0, denominator - numerator)
        elif is_negated and salvage_negated and numerator != 0:
            numerator = max(0, denominator - numerator)
        return numerator, denominator

    raise ValueError(f"Unrecognized frequency value '{token}'")


def _parse_case_database_frequency_value(raw_frequency: str) -> Optional[float]:
    token = (raw_frequency or "").strip()
    if not token:
        return None
    normalized = token.upper()
    if normalized in CASE_DATABASE_UNKNOWN_TOKENS:
        return CASE_DATABASE_DEFAULT_FREQUENCY
    mapped = CASE_DATABASE_FREQUENCY_TEXT_TO_VALUE.get(normalized)
    if mapped is not None:
        return mapped
    ratio_match = RATIO_PATTERN.fullmatch(token)
    if ratio_match:
        denominator = int(ratio_match.group("denominator"))
        numerator = int(ratio_match.group("numerator"))
        if denominator <= 0:
            logger.warning("Skipping ratio with non-positive denominator '%s' in case database.", token)
            return None
        numerator = min(max(numerator, 0), denominator)
        return numerator / float(denominator)
    percentage_match = PERCENTAGE_PATTERN.fullmatch(token)
    if percentage_match:
        percentage = float(percentage_match.group("value"))
        return min(max(percentage / 100.0, 0.0), 1.0)
    try:
        numeric = float(token)
    except ValueError:
        logger.warning("Skipping unrecognized frequency token '%s' in case database.", raw_frequency)
        return None
    if 0.0 <= numeric <= 1.0:
        return numeric
    if 1.0 < numeric <= 100.0:
        return min(numeric / 100.0, 1.0)
    logger.warning("Numeric frequency '%s' out of expected range in case database.", raw_frequency)
    return None


def _parse_case_database_frequency_values(values: Sequence[str]) -> Optional[float]:
    max_frequency: Optional[float] = None
    for value in values:
        frequency = _parse_case_database_frequency_value(value)
        if frequency is None:
            continue
        if max_frequency is None or frequency > max_frequency:
            max_frequency = frequency
    return max_frequency


def load_disease_models_from_case_database(
    case_database_path: str | Path = CASE_DATABASE_DEFAULT_PATH,
    *,
    ontology: Optional[MinimalOntology] = None,
    include_databases: Optional[Set[str]] = None,
) -> List[DiseaseModel]:
    """
    Load disease models from the curated phenotype disease case database JSON.
    """
    path = Path(case_database_path)
    if not path.exists():
        raise FileNotFoundError(f"Case database not found at {path}")

    with path.open(encoding="utf-8") as handle:
        payload = json.load(handle)

    include_prefixes = {prefix.upper() for prefix in include_databases} if include_databases else None

    compiled_annotations: Dict[str, Dict[str, float]] = {}
    compiled_names: Dict[str, str] = {}
    compiled_ids: Dict[str, str] = {}

    for entry in payload.values():
        phenotypes = entry.get("phenotypes") or {}
        converted_annotations: Dict[str, float] = {}
        for raw_term_id, raw_values in phenotypes.items():
            if raw_values is None:
                continue
            if isinstance(raw_values, str):
                values: Sequence[str] = [raw_values]
            elif isinstance(raw_values, Sequence):
                values = list(raw_values)
            else:
                continue
            canonical_term_id = raw_term_id
            if ontology is not None:
                canonical_term_id = ontology.canonical_term_id(raw_term_id)
                if canonical_term_id == PHENOTYPIC_ABNORMALITY_TERM_ID:
                    continue
                if not ontology.has_term(canonical_term_id):
                    logger.debug("Skipping phenotype %s absent from ontology.", raw_term_id)
                    continue
            frequency = _parse_case_database_frequency_values(values)
            if frequency is None:
                continue
            existing = converted_annotations.get(canonical_term_id)
            if existing is None or frequency > existing:
                converted_annotations[canonical_term_id] = frequency

        standard_names = entry.get("standard_names") or {}
        for disease_id, disease_name in standard_names.items():
            if not disease_id:
                continue
            prefix = disease_id.split(":", 1)[0].upper()
            if include_prefixes is not None and prefix not in include_prefixes:
                continue
            key = disease_id.upper()
            annotations = compiled_annotations.setdefault(key, {})
            for term_id, frequency in converted_annotations.items():
                current = annotations.get(term_id)
                if current is None or frequency > current:
                    annotations[term_id] = frequency
            if disease_name:
                compiled_names[key] = disease_name
            compiled_ids[key] = disease_id

    models: List[DiseaseModel] = []
    for key, annotations in compiled_annotations.items():
        disease_id = compiled_ids.get(key, key)
        name = compiled_names.get(key, disease_id)
        models.append(
            DiseaseModel(
                disease_id=disease_id,
                name=name,
                annotations=dict(annotations),
                absent_annotations=set(),
                pretest_probability=None,
            )
        )

    return models


def augment_disease_models_with_case_database(
    diseases: Sequence[DiseaseModel],
    *,
    case_database_path: str | Path = CASE_DATABASE_DEFAULT_PATH,
    ontology: Optional[MinimalOntology] = None,
    include_databases: Optional[Set[str]] = None,
) -> List[DiseaseModel]:
    """
    Merge disease models with additional phenotype annotations from the case database.
    """
    try:
        case_models = load_disease_models_from_case_database(
            case_database_path,
            ontology=ontology,
            include_databases=include_databases,
        )
    except FileNotFoundError:
        logger.warning("Case database not found at %s; returning original disease models.", case_database_path)
        return [
            DiseaseModel(
                disease_id=d.disease_id,
                name=d.name,
                annotations=dict(d.annotations),
                absent_annotations=set(d.absent_annotations),
                pretest_probability=d.pretest_probability,
            )
            for d in diseases
        ]

    augmented: List[DiseaseModel] = []
    clones: Dict[str, DiseaseModel] = {}
    for disease in diseases:
        clone = DiseaseModel(
            disease_id=disease.disease_id,
            name=disease.name,
            annotations=dict(disease.annotations),
            absent_annotations=set(disease.absent_annotations),
            pretest_probability=disease.pretest_probability,
        )
        key = disease.disease_id.upper()
        clones[key] = clone
        augmented.append(clone)

    for case_model in case_models:
        key = case_model.disease_id.upper()
        if key in clones:
            target = clones[key]
            merged_annotations = dict(target.annotations)
            for term_id, frequency in case_model.annotations.items():
                existing = merged_annotations.get(term_id)
                if existing is None or frequency > existing:
                    merged_annotations[term_id] = frequency
            target.annotations = merged_annotations
            if case_model.absent_annotations:
                target.absent_annotations = set(target.absent_annotations).union(case_model.absent_annotations)
            if target.pretest_probability is None:
                target.pretest_probability = case_model.pretest_probability
        else:
            new_model = DiseaseModel(
                disease_id=case_model.disease_id,
                name=case_model.name,
                annotations=dict(case_model.annotations),
                absent_annotations=set(case_model.absent_annotations),
                pretest_probability=case_model.pretest_probability,
            )
            clones[key] = new_model
            augmented.append(new_model)

    return augmented


def load_disease_models_from_hpoa(
    hpoa_path: Optional[str | Path] = None,
    *,
    ontology: Optional[MinimalOntology] = None,
    pretest_probabilities: Optional[Mapping[str, float]] = None,
    loader_options: Optional[HpoDiseaseLoaderOptions] = None,
) -> List[DiseaseModel]:
    """
    Parse an HPO annotation (phenotype.hpoa) file and return a list of DiseaseModel objects.
    
    Args:
        hpoa_path: Path to the phenotype.hpoa file. If None, uses the path from prompt_config.json.
    """
    if hpoa_path is None:
        hpoa_path = HPOA_DEFAULT_PATH
    path = Path(hpoa_path)
    options = loader_options or HpoDiseaseLoaderOptions.default()
    included_databases = {db.upper() for db in options.included_databases}
    salvage_negated = options.salvage_negated_frequencies
    cohort_size = options.cohort_size

    diseases: Dict[str, Dict[str, object]] = {}
    pretest_map = {k.upper(): float(v) for k, v in (pretest_probabilities or {}).items()}

    with path.open(encoding="utf-8") as handle:
        finished_header = False
        for raw_line in handle:
            if raw_line.startswith("#"):
                continue
            line = raw_line.rstrip("\n")
            if not line:
                continue

            if not finished_header:
                if line.startswith("database_id"):
                    header_columns = line.split("\t")
                    if len(header_columns) != 12:
                        raise ValueError("Unexpected number of columns in HPOA header.")
                    finished_header = True
                    continue
                else:
                    # Skip legacy header lines.
                    continue

            fields = line.split("\t")
            if len(fields) < 12:
                fields.extend([""] * (12 - len(fields)))

            (
                disease_id,
                disease_name,
                qualifier,
                hpo_id,
                _reference,
                _evidence,
                _onset,
                frequency,
                _sex,
                _modifier,
                aspect,
                _biocuration,
            ) = fields[:12]

            if disease_id == "database_id":
                continue

            if not disease_id:
                continue

            disease_prefix = disease_id.split(":", 1)[0].upper()
            if disease_prefix not in included_databases:
                continue

            record = diseases.setdefault(
                disease_id,
                {
                    "name": disease_name,
                    "annotations": defaultdict(lambda: {"numerator": 0, "denominator": 0}),
                    "absent": set(),
                },
            )

            if not hpo_id or not hpo_id.startswith("HP:"):
                continue

            canonical_id = ontology.canonical_term_id(hpo_id) if ontology is not None else hpo_id
            if ontology is not None and not ontology.has_term(canonical_id):
                continue

            if aspect != "P":
                continue

            is_negated = qualifier.strip().upper().startswith("NOT")

            try:
                numerator, denominator = _parse_frequency_ratio(
                    is_negated,
                    frequency,
                    salvage_negated,
                    cohort_size,
                )
            except ValueError as exc:
                logger.warning("Skipping annotation with unrecognized frequency '%s': %s", frequency, exc)
                continue

            if is_negated:
                record["absent"].add(canonical_id)
                continue

            if denominator <= 0:
                continue

            stats = record["annotations"][canonical_id]
            stats["numerator"] += numerator
            stats["denominator"] += denominator

    if not diseases:
        return []

    default_pretest = 1.0 / len(diseases)
    models: List[DiseaseModel] = []
    for disease_id, payload in diseases.items():
        annotations: Dict[str, float] = {}
        for term_id, stats in payload["annotations"].items():
            denominator = stats["denominator"]
            numerator = stats["numerator"]
            if denominator <= 0:
                continue
            freq = 0.0 if numerator <= 0 else min(1.0, numerator / denominator)
            annotations[term_id] = freq

        absent = set(payload["absent"])
        disease_id_upper = disease_id.upper()
        if pretest_probabilities:
            pretest = pretest_map.get(disease_id_upper)
        else:
            pretest = default_pretest

        models.append(
            DiseaseModel(
                disease_id=disease_id,
                name=payload["name"],
                annotations=annotations,
                absent_annotations=absent,
                pretest_probability=pretest,
            )
        )

    return models


@dataclass(order=True)
class TestResult:
    """Outcome of ranking a single disease."""

    sort_index: float = field(init=False, repr=False)
    disease_id: str
    disease_name: str
    pretest_probability: float
    observed_results: Sequence[LrWithExplanation]
    excluded_results: Sequence[LrWithExplanation]
    composite_lr: float
    posttest_probability: float

    def __post_init__(self) -> None:
        self.sort_index = -self.posttest_probability


def _product(values: Iterable[float]) -> float:
    result = 1.0
    for value in values:
        result *= value
    return result


def _composite_lr_log_safe(
    observed_results: Sequence[LrWithExplanation],
    excluded_results: Sequence[LrWithExplanation],
    lr_floor: float = 0.3,
    lr_floor_threshold: float = 0.1,
) -> float:
    """Log-space product with floor on very low LRs to avoid underflow and over-penalty."""
    log_sum = 0.0
    for lr in observed_results:
        v = max(lr.lr, lr_floor) if lr.lr < lr_floor_threshold else lr.lr
        log_sum += math.log(max(v, 1e-10))
    for lr in excluded_results:
        v = max(lr.lr, lr_floor) if lr.lr < lr_floor_threshold else lr.lr
        log_sum += math.log(max(v, 1e-10))
    return math.exp(log_sum)


def _canonicalize_terms(terms: Sequence[str], ontology: MinimalOntology) -> List[str]:
    canonical_terms: List[str] = []
    seen: Set[str] = set()
    for term in terms:
        if not term:
            continue
        canonical = ontology.canonical_term_id(term)
        if canonical == PHENOTYPIC_ABNORMALITY_TERM_ID:
            continue
        if not ontology.has_term(canonical):
            logger.warning("Skipping term %s because it is not present in the ontology.", term)
            continue
        if canonical in seen:
            continue
        seen.add(canonical)
        canonical_terms.append(canonical)
    return canonical_terms


@dataclass
class AnalysisOptions:
    disease_databases: Optional[Set[str]] = None
    target_diseases: Optional[Set[str]] = None
    include_diseases_with_no_deleterious_variants: bool = True
    use_global: bool = False
    loader_options: Optional["HpoDiseaseLoaderOptions"] = None


@dataclass(frozen=True)
class HpoDiseaseLoaderOptions:
    included_databases: Set[str]
    salvage_negated_frequencies: bool = True
    cohort_size: int = 5

    @staticmethod
    def default() -> "HpoDiseaseLoaderOptions":
        return HpoDiseaseLoaderOptions(
            included_databases={"OMIM", "ORPHA", "DECIPHER"},
            salvage_negated_frequencies=True,
            cohort_size=5,
        )

    @staticmethod
    def omim_only() -> "HpoDiseaseLoaderOptions":
        return HpoDiseaseLoaderOptions(
            included_databases={"OMIM"},
            salvage_negated_frequencies=True,
            cohort_size=5,
        )


@dataclass
class PreparedPhenotypeRanking:
    ontology: MinimalOntology
    diseases: List[DiseaseModel]
    phenotype_lr: PhenotypeLikelihoodRatio
    options: AnalysisOptions


def _normalize_database_prefixes(prefixes: Optional[Set[str]]) -> Optional[Set[str]]:
    if prefixes is None:
        return None
    normalized = {prefix.split(":", 1)[0].upper() for prefix in prefixes if prefix}
    return normalized or None


def _normalize_target_diseases(targets: Optional[Set[str]]) -> Optional[Set[str]]:
    if targets is None:
        return None
    normalized = {target.upper() for target in targets if target}
    return normalized or None


def _load_disease_grouping_from_case_database(
    case_database_path: str | Path = CASE_DATABASE_DEFAULT_PATH,
) -> Dict[str, str]:
    """
    Load disease grouping information from case database.
    
    Returns a mapping from disease_id (upper case) to the preferred disease_id in the same group.
    Diseases in the same group (same entry's standard_names) will map to the same preferred ID.
    Priority: OMIM > ORPHA > DECIPHER > others (alphabetically)
    
    Args:
        case_database_path: Path to the case database JSON file.
        
    Returns:
        Dictionary mapping disease_id (upper) to preferred disease_id in the same group.
    """
    path = Path(case_database_path)
    if not path.exists():
        logger.debug("Case database not found at %s; no grouping information available.", case_database_path)
        return {}
    
    try:
        with path.open(encoding="utf-8") as handle:
            payload = json.load(handle)
    except Exception as exc:
        logger.warning("Failed to load case database for grouping: %s", exc)
        return {}
    
    # Priority order for disease ID prefixes
    PREFIX_PRIORITY = {"OMIM": 0, "ORPHA": 1, "DECIPHER": 2}
    
    grouping: Dict[str, str] = {}
    
    for entry in payload.values():
        standard_names = entry.get("standard_names") or {}
        if not standard_names:
            continue
        
        # Find the preferred disease ID in this group
        disease_ids = list(standard_names.keys())
        if not disease_ids:
            continue
        
        # Sort by priority: OMIM > ORPHA > DECIPHER > others (alphabetically)
        def get_priority(disease_id: str) -> Tuple[int, str]:
            prefix = disease_id.split(":", 1)[0].upper()
            priority = PREFIX_PRIORITY.get(prefix, 999)
            return (priority, disease_id.upper())
        
        preferred_id = min(disease_ids, key=get_priority)
        
        # Map all IDs in this group to the preferred one
        for disease_id in disease_ids:
            key = disease_id.upper()
            grouping[key] = preferred_id.upper()
    
    return grouping


def _merge_duplicate_diseases_by_grouping(
    diseases: Sequence[DiseaseModel],
    grouping: Dict[str, str],
) -> List[DiseaseModel]:
    """
    Merge diseases that belong to the same group according to the grouping dictionary.
    
    For diseases in the same group, only keep the one with the preferred ID (usually OMIM).
    Merge annotations from all diseases in the same group into the preferred one.
    
    Args:
        diseases: List of disease models to merge.
        grouping: Dictionary mapping disease_id (upper) to preferred disease_id (upper) in the same group.
        
    Returns:
        List of merged disease models.
    """
    if not grouping:
        # No grouping information, return as-is
        return list(diseases)
    
    # Map from preferred_id to list of diseases in that group
    groups: Dict[str, List[DiseaseModel]] = {}
    
    # First pass: group diseases by their preferred ID
    for disease in diseases:
        disease_id_upper = disease.disease_id.upper()
        preferred_id = grouping.get(disease_id_upper, disease_id_upper)
        
        if preferred_id not in groups:
            groups[preferred_id] = []
        groups[preferred_id].append(disease)
    
    # Second pass: merge diseases in each group
    merged: List[DiseaseModel] = []
    for preferred_id, group_diseases in groups.items():
        if len(group_diseases) == 1:
            # Only one disease in group, no merging needed
            merged.append(group_diseases[0])
            continue
        
        # Find the disease with the preferred ID (should be first, but check to be sure)
        preferred_disease = None
        other_diseases = []
        
        for d in group_diseases:
            if d.disease_id.upper() == preferred_id:
                preferred_disease = d
            else:
                other_diseases.append(d)
        
        # If no exact match, use the first one
        if preferred_disease is None:
            preferred_disease = group_diseases[0]
            other_diseases = group_diseases[1:]
        
        # Merge annotations from other diseases in the group
        merged_annotations = dict(preferred_disease.annotations)
        merged_absent = set(preferred_disease.absent_annotations)
        merged_name = preferred_disease.name
        
        for other in other_diseases:
            # Merge annotations (take maximum frequency)
            for term_id, frequency in other.annotations.items():
                existing = merged_annotations.get(term_id)
                if existing is None or frequency > existing:
                    merged_annotations[term_id] = frequency
            
            # Merge absent annotations
            merged_absent.update(other.absent_annotations)
            
            # Use the preferred disease's name, but log if different
            if other.name != merged_name:
                logger.debug(
                    "Merging disease %s (%s) into %s (%s); using name from preferred ID.",
                    other.disease_id,
                    other.name,
                    preferred_disease.disease_id,
                    merged_name,
                )
        
        # Create merged disease model
        merged.append(
            DiseaseModel(
                disease_id=preferred_disease.disease_id,
                name=merged_name,
                annotations=merged_annotations,
                absent_annotations=merged_absent,
                pretest_probability=preferred_disease.pretest_probability,
            )
        )
    
    logger.info(
        "Merged %d diseases into %d groups based on case database grouping.",
        len(diseases),
        len(merged),
    )
    
    return merged


def _filter_diseases(diseases: Sequence[DiseaseModel], options: AnalysisOptions) -> List[DiseaseModel]:
    if not diseases:
        return []

    disease_databases = _normalize_database_prefixes(options.disease_databases)
    if disease_databases is None:
        disease_databases = DEFAULT_DISEASE_DATABASES

    if options.target_diseases:
        targets = _normalize_target_diseases(options.target_diseases)
        filtered = [d for d in diseases if d.disease_id.upper() in targets]
        return filtered

    filtered = [
        d for d in diseases
        if d.disease_id.split(":", 1)[0].upper() in disease_databases
    ]
    return filtered or []


def _prepare_ranking_context(
    diseases: Sequence[DiseaseModel],
    ontology: MinimalOntology,
    options: AnalysisOptions,
) -> Optional[PreparedPhenotypeRanking]:
    include_prefixes = _normalize_database_prefixes(options.disease_databases)
    augmented_diseases = augment_disease_models_with_case_database(
        diseases,
        ontology=ontology,
        include_databases=include_prefixes,
    )
    filtered = _filter_diseases(augmented_diseases, options)
    if not filtered:
        return None

    # Load disease grouping information and merge duplicates
    grouping = _load_disease_grouping_from_case_database()
    merged_diseases = _merge_duplicate_diseases_by_grouping(filtered, grouping)

    prepared_diseases: List[DiseaseModel] = []

    uniform = 1.0 / len(merged_diseases)
    for disease in merged_diseases:
        prepared_diseases.append(
            DiseaseModel(
                disease_id=disease.disease_id,
                name=disease.name,
                annotations=dict(disease.annotations),
                absent_annotations=set(disease.absent_annotations),
                pretest_probability=uniform,
            )
        )

    if not prepared_diseases:
        return None

    phenotype_lr = PhenotypeLikelihoodRatio(ontology, prepared_diseases)
    return PreparedPhenotypeRanking(
        ontology=ontology,
        diseases=prepared_diseases,
        phenotype_lr=phenotype_lr,
        options=options,
    )


class PhenotypeRankingEngine:
    def __init__(
        self,
        ontology: MinimalOntology,
        diseases: Sequence[DiseaseModel],
        options: Optional[AnalysisOptions] = None,
    ):
        analysis_options = options or AnalysisOptions()
        context = _prepare_ranking_context(diseases, ontology, analysis_options)
        if context is None:
            raise ValueError("No diseases available for ranking with the supplied options.")
        self._context = context

    @property
    def context(self) -> PreparedPhenotypeRanking:
        return self._context

    def rank(
        self,
        observed_terms: Sequence[str],
        excluded_terms: Sequence[str],
    ) -> List[TestResult]:
        return rank_diseases_by_phenotypes(
            observed_terms,
            excluded_terms,
            self._context.diseases,
            self._context.ontology,
            options=self._context.options,
            prepared_ranking=self._context,
        )


def rank_diseases_by_phenotypes(
    observed_terms: Sequence[str],
    excluded_terms: Sequence[str],
    diseases: Sequence[DiseaseModel],
    ontology: MinimalOntology,
    options: Optional[AnalysisOptions] = None,
    prepared_ranking: Optional[PreparedPhenotypeRanking] = None,
) -> List[TestResult]:
    """
    Rank diseases according to observed and excluded phenotypes.

    Args:
        observed_terms: HPO term IDs observed in the individual.
        excluded_terms: HPO term IDs explicitly excluded in the individual.
        diseases: Iterable of DiseaseModel objects describing phenotype annotations.
        ontology: MinimalOntology instance containing the HPO graph.

    Returns:
        List of TestResult objects sorted by descending post-test probability.
    """
    if prepared_ranking is not None:
        context = prepared_ranking
    else:
        analysis_options = options or AnalysisOptions()
        context = _prepare_ranking_context(diseases, ontology, analysis_options)
        if context is None:
            logger.warning("No diseases available after applying analysis options.")
            return []

    ontology_for_ranking = context.ontology
    prepared_diseases = context.diseases
    phenotype_lr = context.phenotype_lr

    observed_terms_canonical = _canonicalize_terms(observed_terms, ontology_for_ranking)
    excluded_terms_canonical = _canonicalize_terms(excluded_terms, ontology_for_ranking)

    if not observed_terms_canonical and not excluded_terms_canonical:
        logger.warning("No valid phenotype terms supplied; returning empty results.")
        return []

    ranked_results: List[TestResult] = []
    for disease in prepared_diseases:
        pretest = disease.pretest_probability
        if pretest < 0.0 or pretest >= 1.0:
            raise ValueError(
                f"Pretest probability for disease {disease.disease_id} must be in [0, 1). Got {pretest}."
            )
        idg = InducedDiseaseGraph.create(disease, ontology_for_ranking)
        observed_results = [phenotype_lr.lr_for_observed_term(term, idg) for term in observed_terms_canonical]
        excluded_results = [phenotype_lr.lr_for_excluded_term(term, idg) for term in excluded_terms_canonical]
        composite_lr = _composite_lr_log_safe(observed_results, excluded_results)
        pretest_odds = pretest / (1.0 - pretest) if pretest > 0 else 0.0
        posttest_odds = pretest_odds * composite_lr
        posttest_probability = posttest_odds / (1.0 + posttest_odds) if posttest_odds > 0 else 0.0
        ranked_results.append(
            TestResult(
                disease_id=disease.disease_id,
                disease_name=disease.name,
                pretest_probability=pretest,
                observed_results=observed_results,
                excluded_results=excluded_results,
                composite_lr=composite_lr,
                posttest_probability=posttest_probability,
            )
        )
    ranked_results.sort()
    return ranked_results


__all__ = [
    "DiseaseModel",
    "InducedDiseaseGraph",
    "LrMatchType",
    "LrWithExplanation",
    "MinimalOntology",
    "PhenotypeAnnotation",
    "PhenotypeLikelihoodRatio",
    "HpoDiseaseLoaderOptions",
    "PreparedPhenotypeRanking",
    "PhenotypeRankingEngine",
    "TestResult",
    "CASE_DATABASE_DEFAULT_PATH",
    "OBO_DEFAULT_PATH",
    "HPOA_DEFAULT_PATH",
    "load_minimal_ontology_from_hp_json",
    "load_disease_models_from_case_database",
    "load_disease_models_from_hpoa",
    "augment_disease_models_with_case_database",
    "rank_diseases_by_phenotypes",
]

