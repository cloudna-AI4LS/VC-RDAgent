<h1 align="center">VC-RDAgent</h1>

<p align="center">
  <a href="https://doi.org/10.64898/2026.02.09.702153">Paper</a>
  &nbsp;·&nbsp;
  <a href="https://github.com/cloudna-AI4LS/VC-RDAgent/tree/main/rare-disease-chat">Demo</a>
</p>

A Virtual Case (VC) augmented reasoning framework for robust and privacy-preserving rare disease diagnosis. VC synthesizes *virtual standardized cases* from knowledge bases (HPO, Orphanet, OMIM, MONDO) instead of real clinical records, enabling privacy-conscious, evidence-grounded diagnostic reasoning.

**Features**

- **Interactive Chat & MCP Server** — [rare-disease-chat](rare-disease-chat/): phenotype extraction, disease diagnosis, disease info retrieval as MCP tools; LangGraph multi-agent chat (terminal & Web UI); Docker or local install.
- **Multi-step prompt generation** — 2/3-step chain-of-thought prompts with overlap, embedding, or hybrid case extraction.
- **Ensemble disease ranking** — Fuses IC-weighted similarity, annotation-frequency–weighted similarity, and embedding-based similar-case ranking via Z-statistics.
- **LLM evaluation** — Phenotype-to-disease prediction via local models (e.g. Qwen) or cloud APIs, with Top‑K and similarity metrics.
- **Standard biomedical ontologies** — HPO, Orphanet, MONDO, and custom disease–phenotype knowledge graphs.

---

## Installation

**Requirements:** Python 3.12.

```bash
uv venv --python 3.12 .venv
uv pip install -r requirements.txt
source .venv/bin/activate
```

---

## Configuration

Set the following parameters in each config file before running.

### `pho2disease/prompt_config.json`

| Parameter | Required | Description |
|-----------|----------|-------------|
| `base_path` | **Yes** | Project root (e.g. `/path/to/VC-RDAgent`). All other paths use `{base_path}/...`. |
| `input_file` | Optional | Evaluation input JSON; default `{base_path}/data/PUMCH-ADM.json`. Change if using another dataset. |
| `gpu_id` | Optional | GPU device ID for local models / embedding; default `2`. |

Other paths (`phenotype_hpoa`, `obo_file`, `case_library`, `embedding_file`, `ic_file`, Orphanet/MONDO paths, etc.) are derived from `base_path`; override only if your layout differs.

### `pho2disease/inference_config.json`

| Parameter | Required | Description |
|-----------|----------|-------------|
| `base_paths.base_path` | **Yes** | Project root (same as above). |
| `openrouter_config.api_key` | **Yes** if using OpenRouter | Your OpenRouter API key; leave empty only for local models. |
| `openrouter_config.model_name` | Optional | OpenRouter model (e.g. `qwen/qwen3-8b:free`). |
| `model_config.default_model_name` | Optional | `"openrouter"` or local model name (e.g. `"Qwen/Qwen3-8B"`). |
| `model_config.default_cache_dir` | Optional | Model cache dir; default `{base_path}/model_weight`. |

Other file paths use `${base_path}`; override only if your layout differs.

---

## Quick start

### Evaluation workflow (2 steps)

```bash
cd pho2disease
```

**Step 1 — Ensemble disease ranking** (fuses IC/frequency/embedding similarity; outputs final ranking and prompts):

```bash
python ensemble_disease_ranking.py --config prompt_config.json --prompt_steps 2
```

**Step 2 — LLM prediction** (local Qwen or OpenRouter; replace `xxx.json` with the prompt file from step 1):

```bash
# Local Qwen
python phenotype_to_disease_prediction_bysteps.py --prompts_file ./prompt/xxx.json --model_name "Qwen/Qwen3-8B" --gpu_id 0

# Or OpenRouter
python phenotype_to_disease_prediction_bysteps.py --prompts_file ./prompt/xxx.json --model_name openrouter
```

### Interactive chat (rare-disease-chat)

See the detailed setup and usage guide in [`rare-disease-chat/README.md`](rare-disease-chat/README.md) for Docker, local startup, and Web UI.

### Phenotype embeddings

```bash
cd hpo_embedding/poincare_model
python trainpoincare.py
python extract_phenotype_embeddings.py final_poincare_phenotype_embeddings.csv -o ../phe2embedding_recomputed.json
```

---

## More information

### Data sources and paths

#### 1. Evaluation / input data (`data/`)

Phenotype–disease pairs for inference and evaluation. JSON: one `[[phenotype IDs], [disease IDs]]` per sample.

| File | Description |
|------|-------------|
| `PUMCH-ADM.json` / `.jsonl` | PUMCH (Peking Union Medical College Hospital) admission cases |
| `HMS.json` / `.jsonl` | HMS cohort phenotype–disease benchmarks |
| `LIRICAL.json` / `.jsonl` | LIRICAL-related benchmark data |
| `mygene2.json` / `.jsonl` | MyGene2-derived phenotype–disease data |

#### 2. HPO annotations (`hpo_annotations/`)

[Human Phenotype Ontology](https://hpo.jax.org/data/annotations) and disease–phenotype / gene–phenotype mappings.

| File | Description |
|------|-------------|
| `hp.obo` | HPO ontology (OBO); phenotype hierarchy and ID–name mapping |
| `phenotype.hpoa` | Disease–phenotype annotations (OMIM, DECIPHER, ORPHANET) with frequency and evidence |
| `genes_to_phenotype.txt` | Gene → phenotype |
| `phenotype_to_genes.txt` | Phenotype → gene |
| `genes_to_disease.txt` | Gene → disease |
| `disease_descriptions_batch.json` | Disease descriptions; can be produced by `batch_disease_scraper` |

#### 3. Orphanet (`orphanet_annotations/`)

[Orphanet](https://sciences.orphadata.com/orphanet-scientific-knowledge-files/) rare diseases and HPO phenotype associations.

| File | Description |
|------|-------------|
| `Phenotypes_Associated_with_Rare_Disorders.xml` / `.json` | Orphanet Product 4: rare disease–HPO associations (use `xml_to_json_converter_product4.py` to produce JSON) |
| `rare_diseases_alignment.xml` / `.json` | Orphanet Product 1: disease alignment across sources (use `xml_to_json_converter_product1.py`) |
| `categorization_of_orphanet_diseases.csv` | **Produced by** `extract_disease_type_mapping.py` from Orphanet **Product 3** (Classifications of rare diseases) XMLs: OrphaNumber, Disorder_Name, Category (pipe-separated classification names) |
| `disease_type_mapping.json` | **Produced by** `extract_disease_type_mapping.py` from the same Product 3 XMLs: disorder→type and classification mappings |

#### 4. MONDO (`mondo_annotations/`)

[MONDO Disease Ontology](https://monarchinitiative.org/kg/downloads) for ID alignment and rare-disease tagging.

| File | Description |
|------|-------------|
| `mondo_parsed_full.json` | Parsed MONDO OBO (IDs, names, synonyms, equivalent IDs, rare subset, parents) via `parse_mondo_obo.py` |

#### 5. Disease–phenotype knowledge graph (`disease_phenotype_kg/`)

CSV-based graph: disease, phenotype, gene nodes and edges. **All of these files are produced by** `generate_prompts_bysteps.py` **when run with** `--save_case_library_only` (see Quick start / generate multi-step prompts).

**Upstream sources** (must be configured in `prompt_config.json` before running):

- **HPO:** `hp.obo` (phenotype hierarchy, `is_a`, `alt_id`, names, definitions, comments), `phenotype.hpoa` (disease–phenotype associations and frequencies), `genes_to_phenotype.txt`, `genes_to_disease.txt`
- **Orphanet:** `Phenotypes_Associated_with_Rare_Disorders.json` (disease–phenotype), `rare_diseases_alignment.json` (IDs and names), `categorization_of_orphanet_diseases.csv` (disease type, is_rare)
- **MONDO:** `mondo_parsed_full.json` (for ID alignment and merging)
- **Disease descriptions:** `disease_descriptions_batch.json`
- **Embeddings:** `phe2embedding_recomputed.json`

**Output:** `graph_output.output_dir` in `prompt_config.json` (e.g. `{base_path}/disease_phenotype_kg`). `graph_output.recomputed_ic_file` is where the recomputed IC is written.

| File | Description | Main upstream |
|------|-------------|---------------|
| `disease_nodes.csv` | Aggregated disease nodes (ID `D:X`, standard name, synonyms, type, is_rare, description) | phenotype.hpoa, Orphanet (names, categorization), disease_descriptions |
| `publicDisease_nodes.csv` | Concrete OMIM/ORPHA nodes (ID, standard name, link to omim.org / orpha.net) | phenotype.hpoa, Orphanet |
| `phenotype_nodes.csv` | Phenotype nodes (ID, name, synonyms, description, comment, associations, IC, embedding) | hp.obo, phenotype.hpoa, phe2embedding; IC recomputed from associations |
| `disease_to_phenotype_edges.csv` | Disease → phenotype (`has`), with frequency and source | phenotype.hpoa, Orphanet Phenotypes_Associated |
| `phenotype_to_phenotype_edges.csv` | Phenotype `is_a` and `phenotype_exact` (alt_id) | hp.obo |
| `disease_to_publicDisease_edges.csv` | Aggregated `D:X` → OMIM/ORPHA (`disease_exact`) | Merging by name from phenotype.hpoa and Orphanet |
| `gene_to_phenotype_edges.csv` | Gene–phenotype (disease–phenotype–gene, with expansion rules) | HPO `genes_to_phenotype.txt`; optional, skipped if not configured |
| `gene_to_disease_edges.csv` | Gene–disease | HPO `genes_to_disease.txt`; optional, skipped if not configured |

#### 6. Case library (`general_cases/`)

Virtual phenotype–disease cases for similar-case retrieval, embedding-based ranking, and few-shot.

| File | Description |
|------|-------------|
| `phenotype_disease_case_library.jsonl` | `{"Phenotype": [HPO IDs], "RareDisease": [disease IDs], "Department": ...}` per line |
| `phenotype_disease_case_library_with_high_freq.jsonl` | Filtered by high-frequency phenotypes |
| `phenotype_disease_case_database.json` | Aggregated by disease (standard name, aliases, phenotypes, frequencies) |
| `disease_ids_names.json` | Disease ID–name mapping |

#### 7. Phenotype embeddings (`hpo_embedding/`)

Phenotype vectors and information content for semantic similarity and weighted ranking.

| File | Description |
|------|-------------|
| `phe2embedding_recomputed.json` | Phenotype ID → embedding vector (from `poincare_model/extract_phenotype_embeddings.py`) |
| `ic_dict_recomputed.json` | Information content (IC) (from `pho2disease/generate_prompts_bysteps.py` with `--save_case_library_only`) |
| `poincare_model/trainpoincare.py` | Poincaré embedding training |
| `poincare_model/extract_phenotype_embeddings.py` | Extract HP phenotypes from embedding CSV → `phe2embedding_*.json` |

### Full usage and arguments

#### 1. Generate multi-step prompts — `generate_prompts_bysteps.py`

Build 2/3-step reasoning prompts from phenotype–disease JSON. Case extraction: `overlap`, `embedding`, or `both`.

```bash
cd pho2disease
```

**With config (recommended):**

```bash
# 2-step, top_k=50 from prompt_config.json
python generate_prompts_bysteps.py --config prompt_config.json --prompt_steps 2

# Export case library, phenotype_disease_case_database, disease_ids_names, and disease_phenotype_kg CSVs (no prompts)
python generate_prompts_bysteps.py --config prompt_config.json --save_case_library_only

# Custom input, output, top_k
python generate_prompts_bysteps.py --config prompt_config.json --input_file /path/to/PUMCH-ADM.json \
  --prompt_steps 2 --top_k 50 --output_file ./prompt/out.json

# Subset: first N or by indices
python generate_prompts_bysteps.py --config prompt_config.json --prompt_steps 2 --max_samples 10
python generate_prompts_bysteps.py --config prompt_config.json --prompt_steps 2 --sample_indices "0,5,10,19-24"
```

**Main arguments:**

| Argument | Description |
|----------|-------------|
| `--config` | Config path (default: `prompt_config.json`) |
| `--input_file` | Input JSON: `[[phenotypes],[diseases]]` per sample |
| `--obo_file`, `--phenotype_hpoa` | HPO OBO and annotations |
| `--disease_mapping` | Optional ID→name mapping |
| `--case_library` | Case library JSONL (required for `embedding`/`both` or `--use_samples`) |
| `--embedding_file`, `--ic_file` | Required for `embedding` / `both` |
| `--case_extraction_method` | `overlap` \| `embedding` \| `both` |
| `--prompt_steps` | 2 \| 3 |
| `--top_k` | Candidate diseases per sample |
| `--use_IC_weights`, `--use_frequency_weights`, `--use_score` | Ranking weights |
| `--use_samples` | Use true associations from case library (requires `--case_library`) |
| `--save_case_library_only` | Export case library, case_database, disease_ids_names, and `disease_phenotype_kg` CSVs only; no prompt generation |
| `--max_samples`, `--sample_indices` | Sample subset |
| `--output_file` | Output JSON path |

---

#### 2. Ensemble disease ranking — `ensemble_disease_ranking.py`

Fuses IC-weighted similarity, annotation-frequency–weighted similarity, and embedding-based similar-case ranking; produces a final ranking via Z-statistics and optionally multi-step prompts.

```bash
cd pho2disease
```

**Examples:**

```bash
# Config input_file, 2-step, with prompts (default)
python ensemble_disease_ranking.py --config prompt_config.json --prompt_steps 2

# Custom input, top_k, final_top_k
python ensemble_disease_ranking.py --input_file /path/to/PUMCH-ADM.json --config prompt_config.json \
  --prompt_steps 2 --top_k 100 --final_top_k 50 --output_file ensemble_out.json

# Ranking only, no prompts
python ensemble_disease_ranking.py --config prompt_config.json --prompt_steps 2 --no_prompt

# Subset by indices or count
python ensemble_disease_ranking.py --config prompt_config.json --prompt_steps 2 \
  --sample_indices "0,5,10,19-24" --output_file ensemble_selected.json
python ensemble_disease_ranking.py --config prompt_config.json --prompt_steps 2 --num_samples 10
```

**Arguments:**

| Argument | Description |
|----------|-------------|
| `--config` | Path to `prompt_config.json` |
| `--input_file` | Phenotype–disease JSON (or from config) |
| `--prompt_steps` | 2 or 3 (required) |
| `--top_k` | Candidates per strategy (default: 100) |
| `--final_top_k` | Final ranking size (default: 50) |
| `--no_prompt` | Output ranking only |
| `--sample_indices`, `--num_samples` | Sample subset |
| `--output_file` | Output path (default: `{base_path}/pho2disease/prompt/` with timestamp) |

---

#### 3. Phenotype→disease prediction (LLM) — `phenotype_to_disease_prediction_bysteps.py`

Runs multi-step inference on prompt JSON from `generate_prompts_bysteps` or `ensemble_disease_ranking`, using local Qwen or OpenRouter. Reports accuracy, Top‑K, rank, and similarity.

```bash
cd pho2disease
```

**Examples:**

```bash
# Local Qwen
python phenotype_to_disease_prediction_bysteps.py --prompts_file ./prompt/xxx.json \
  --model_name "Qwen/Qwen3-8B" --gpu_id 0

# Local Qwen + chain-of-thought
python phenotype_to_disease_prediction_bysteps.py --prompts_file ./prompt/xxx.json \
  --model_name "Qwen/Qwen3-8B" --gpu_id 0 --enable_thinking

# OpenRouter (uses inference_config or --openrouter_api_key)
python phenotype_to_disease_prediction_bysteps.py --prompts_file ./prompt/xxx.json --model_name openrouter

# OpenRouter with explicit model
python phenotype_to_disease_prediction_bysteps.py --prompts_file ./prompt/xxx.json \
  --model_name openrouter --api_model "qwen/qwen3-8b:free"

# Subset of samples
python phenotype_to_disease_prediction_bysteps.py --prompts_file ./prompt/xxx.json \
  --sample_indices "0,5-7,10,20-25"
python phenotype_to_disease_prediction_bysteps.py --prompts_file ./prompt/xxx.json --num_samples 50
```

**Arguments:**

| Argument | Description |
|----------|-------------|
| `--prompts_file` | Prompt JSON (required) |
| `--output_file` | Results JSON (default from `inference_config`) |
| `--model_name` | `"Qwen/Qwen3-8B"` or `openrouter` |
| `--api_model` | OpenRouter model override |
| `--gpu_id` | GPU for local Qwen |
| `--enable_thinking` | Enable thinking/CoT |
| `--sample_indices`, `--num_samples` | Sample subset |
| `--max_retries` | Retries on API failure |

### Orphanet & MONDO preprocessing

#### Orphanet XML → JSON

```bash
cd orphanet_annotations
python xml_to_json_converter_product4.py --input Phenotypes_Associated_with_Rare_Disorders.xml --output Phenotypes_Associated_with_Rare_Disorders.json
python xml_to_json_converter_product1.py --input rare_diseases_alignment.xml --output rare_diseases_alignment.json
```

#### Product 3 (Classifications of rare diseases)

Download XML package from Orphanet, extract, then:

```bash
cd orphanet_annotations
python extract_disease_type_mapping.py
# Custom: --input-dir /path/to/Classifications_of_rare_diseases --output disease_type_mapping.json --categories-output categorization_of_orphanet_diseases.csv
```
