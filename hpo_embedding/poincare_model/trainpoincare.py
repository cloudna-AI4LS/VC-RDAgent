import torch
import torch.nn as nn
import torch.optim as optim
import networkx as nx
import numpy as np
import pickle
import csv
import os
import pandas as pd
import random
import json
import geoopt
from geoopt import ManifoldParameter
from geoopt.manifolds import PoincareBall

# ==========================================
# 0. Global settings
# ==========================================
torch.set_default_dtype(torch.float64)

# ==========================================
# 0.2. Weight configuration
# ==========================================
# 1. Default weight for other edges (is_a, causes). Suggested: 1.0 (skeleton, strong connection)
DEFAULT_OTHER_WEIGHT = 1.0

# 2. Default when frequency is missing (has_symptom). Fill when frequency_max absent in CSV. Suggested: 0.3–0.5
DEFAULT_MISSING_FREQ = 0.5

# 3. Disease–phenotype edge scale (training). Amplifies disease–phenotype pull. Suggested: 5.0
DISEASE_EDGE_SCALE = 2

# 4. Knowledge graph dir (disease_phenotype_kg). Files: phenotype_to_phenotype_edges, disease_to_phenotype_edges, gene_to_disease_edges, phenotype_nodes (IC), optional ic_dict_recomputed.json
KG_DIR = os.path.normpath(os.path.join(os.path.dirname(__file__), "../../disease_phenotype_kg"))

def set_seed(seed=42):
    """Set all RNG seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# ==========================================
# 1. Data loading (using config variables)
# ==========================================
def _normalize_ic_dict(ic_dict, path, label):
    if not ic_dict:
        print(f"Loaded IC from {label}: {path} (empty)")
        return ic_dict
    vals = list(ic_dict.values())
    if not vals:
        print(f"Loaded IC from {label}: {path} (empty)")
        return ic_dict
    mi, ma = min(vals), max(vals)
    if ma > mi:
        ic_dict = {k: (v - mi) / (ma - mi) for k, v in ic_dict.items()}
        print(f"Loaded IC from {label}: {path} ({len(ic_dict)} phenotypes), normalized to [0,1] (original [{mi:.4f},{ma:.4f}])")
    else:
        ic_dict = {k: 0.5 for k in ic_dict}
        print(f"Loaded IC from {label}: {path} ({len(ic_dict)} phenotypes), all same -> 0.5")
    return ic_dict


def load_ic_dict(ic_file_path=None):
    """
    Load IC: 1) phenotype_nodes.csv (ID, IC) in KG_DIR; 2) ic_dict_recomputed.json in KG_DIR or ic_file_path.
    Returns {phenotype_id: ic_value} in [0,1]; empty dict if not found.
    """
    pn_path = os.path.join(KG_DIR, "phenotype_nodes.csv")
    if os.path.exists(pn_path):
        try:
            ic_dict = {}
            with open(pn_path, 'r', encoding='utf-8') as f:
                for row in csv.DictReader(f):
                    pid, ic = (row.get('ID') or '').strip(), (row.get('IC') or '').strip()
                    if pid and ic:
                        try:
                            ic_dict[pid] = float(ic)
                        except ValueError:
                            pass
            if ic_dict:
                return _normalize_ic_dict(ic_dict, pn_path, "phenotype_nodes")
        except Exception as e:
            print(f"Warning: Failed to load IC from phenotype_nodes {pn_path}: {e}")

    json_path = ic_file_path or os.path.join(KG_DIR, "ic_dict_recomputed.json")
    if json_path and os.path.exists(json_path):
        try:
            with open(json_path, 'r', encoding='utf-8-sig') as f:
                ic_dict = json.load(f)
            return _normalize_ic_dict(ic_dict, json_path, "ic_dict_recomputed.json")
        except Exception as e:
            print(f"Warning: Failed to load IC file {json_path}: {e}")
    print("Warning: IC file not found, will not use IC weighting")
    return {}

def build_medical_graph():
    G = nx.DiGraph()

    # 1. HPO hierarchy (is_a): parent -> child. sourceID is_a targetID => (targetID, sourceID) i.e. (parent, child)
    hpo_edges = []
    path_pp = os.path.join(KG_DIR, "phenotype_to_phenotype_edges.csv")
    if os.path.exists(path_pp):
        with open(path_pp, 'r', encoding='utf-8') as f:
            for row in csv.DictReader(f):
                if (row.get('relationship') or '').strip().lower() != 'is_a':
                    continue
                s, t = (row.get('sourceID') or '').strip(), (row.get('targetID') or '').strip()
                if s and t:
                    hpo_edges.append((t, s))  # (parent, child)

    # 2. Disease-phenotype (has_symptom): (Disease, HP, frequency_max). sourceID=disease, targetID=hp
    disease_edge_data = []
    path_dp = os.path.join(KG_DIR, "disease_to_phenotype_edges.csv")
    if os.path.exists(path_dp):
        with open(path_dp, 'r', encoding='utf-8') as f:
            for row in csv.DictReader(f):
                if (row.get('relationship') or '').strip().lower() != 'has':
                    continue
                d, hp = (row.get('sourceID') or '').strip(), (row.get('targetID') or '').strip()
                if not d or not hp:
                    continue
                v = (row.get('frequency_max') or '').strip()
                try:
                    freq = float(v) if v else DEFAULT_MISSING_FREQ
                    freq = freq if 0.0 <= freq <= 1.0 else DEFAULT_MISSING_FREQ
                except ValueError:
                    freq = DEFAULT_MISSING_FREQ
                disease_edge_data.append((d, hp, freq))

    # 3. Gene-disease (causes): Gene:ncbi_gene_id -> disease_id
    gene_edges = []
    path_gd = os.path.join(KG_DIR, "gene_to_disease_edges.csv")
    if os.path.exists(path_gd):
        with open(path_gd, 'r', encoding='utf-8') as f:
            for row in csv.DictReader(f):
                g = (row.get('ncbi_gene_id') or row.get('gene_symbol') or '').strip()
                d = (row.get('disease_id') or '').strip()
                if g and d:
                    gene_edges.append((f"Gene:{g}", d))
    
    # === Build graph (initial weights from config) ===
    for u, v in hpo_edges: 
        G.add_edge(u, v, type='is_a', weight=DEFAULT_OTHER_WEIGHT)
        
    for u, v, w in disease_edge_data: 
        G.add_edge(u, v, type='has_symptom', weight=w)
        
    for u, v in gene_edges: 
        G.add_edge(u, v, type='causes', weight=DEFAULT_OTHER_WEIGHT)
    
    print(f"Graph built: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges.")
    return G

# ==========================================
# 2. Poincaré embedding model
# ==========================================
class PoincareModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim=5, epsilon=1e-5):
        super(PoincareModel, self).__init__()
        self.epsilon = epsilon
        # Poincaré manifold
        self.manifold = PoincareBall()
        # Manifold parameter (constrained to Poincaré ball)
        embedding_weight = torch.randn(vocab_size, embedding_dim) * 0.01
        # Ensure inside ball at init
        embedding_weight = self.manifold.projx(embedding_weight)
        # Register as manifold parameter
        self.embeddings = ManifoldParameter(embedding_weight, manifold=self.manifold)

    def poincare_distance(self, u, v):
        sq_dist = torch.sum(torch.pow(u - v, 2), dim=-1, keepdim=True)
        sq_u = torch.sum(torch.pow(u, 2), dim=-1, keepdim=True)
        sq_v = torch.sum(torch.pow(v, 2), dim=-1, keepdim=True)
        boundary = 1 - 1e-4
        sq_u = torch.clamp(sq_u, max=boundary)
        sq_v = torch.clamp(sq_v, max=boundary)
        alpha = sq_dist / ((1 - sq_u) * (1 - sq_v))
        x = 1 + 2 * alpha
        x = torch.clamp(x, min=1 + 1e-5)
        dist = torch.log(x + torch.sqrt(torch.pow(x, 2) - 1))
        return dist.squeeze(-1)

    def forward(self, u_idx, v_idx):
        # Index embeddings from manifold parameter
        u = self.embeddings[u_idx]
        v = self.embeddings[v_idx]
        return self.poincare_distance(u, v)

# ==========================================
# 3. Helper functions (projection)
# ==========================================
def project_into_ball(embedding_weight):
    with torch.no_grad():
        norm = torch.norm(embedding_weight, p=2, dim=-1, keepdim=True)
        max_norm = 1.0 - 1e-3
        cond = (norm >= max_norm).float()
        scaling = max_norm / (norm + 1e-7)
        embedding_weight.data = embedding_weight.data * (1 - cond) + \
                                (embedding_weight.data * scaling) * cond

def einstein_midpoint(embeddings, weights=None, epsilon=1e-5):
    """
    Einstein midpoint (weighted center) in the Poincaré ball.

    Args:
        embeddings: [N, dim] or [Batch, N, dim] – phenotype vectors
        weights: [N, 1] or [Batch, N, 1] – (optional) per-phenotype weights, e.g. frequency
    Returns:
        midpoint: [1, dim] or [Batch, 1, dim]
    """
    # 1. Uniform weights if None
    if weights is None:
        weights = torch.ones_like(embeddings[..., :1])

    # Normalize weights to sum to 1
    weights = weights / torch.sum(weights, dim=-2, keepdim=True)

    # 2. Poincaré -> Klein: k = 2*u / (1 + ||u||^2)
    u_norm_sq = torch.sum(embeddings ** 2, dim=-1, keepdim=True)
    klein_vecs = 2 * embeddings / (1 + u_norm_sq)

    # 3. Weighted mean in Klein (Euclidean). Klein centroid = Poincaré Einstein midpoint.
    klein_mean = torch.sum(weights * klein_vecs, dim=-2, keepdim=True)

    # 4. Klein -> Poincaré: u = k / (1 + sqrt(1 - ||k||^2))
    k_norm_sq = torch.sum(klein_mean ** 2, dim=-1, keepdim=True)

    # Clamp for numerical stability (k_norm_sq >= 1 can happen from float error)
    k_norm_sq = torch.clamp(k_norm_sq, max=1.0 - epsilon)
    
    midpoint = klein_mean / (1 + torch.sqrt(1 - k_norm_sq))
    
    return midpoint

# ==========================================
# 4. Training (config-driven)
# ==========================================
def train_model(seed=42):
    set_seed(seed)
    
    G = build_medical_graph()
    nodes = list(G.nodes())
    node2id = {n: i for i, n in enumerate(nodes)}
    id2node = {i: n for i, n in enumerate(nodes)}
    edges = list(G.edges()) 
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    print(f"Config -> Other Weight: {DEFAULT_OTHER_WEIGHT}, Missing Freq: {DEFAULT_MISSING_FREQ}, Scale: {DISEASE_EDGE_SCALE}")
    
    # Load IC dict
    ic_dict = load_ic_dict()

    # Build loss weight vector (apply weights by edge type)
    edge_weights_list = []

    for u, v in edges:
        data = G[u][v]
        raw_weight = data.get('weight', DEFAULT_OTHER_WEIGHT)
        edge_type = data.get('type', 'unknown')

        if edge_type == 'has_symptom':
            # has_symptom: raw_freq * scale * IC
            final_weight = raw_weight * DISEASE_EDGE_SCALE
            # Weight by IC if phenotype node (usually v for Disease->HP)
            phenotype_node = v if v.startswith('HP:') else (u if u.startswith('HP:') else None)
            if phenotype_node and ic_dict:
                ic_value = ic_dict.get(phenotype_node, 0.5)  # default 0.5 if missing
                final_weight = final_weight * ic_value
        else:
            # Other edges: keep raw weight
            final_weight = raw_weight
            # is_a: use target (child) phenotype IC as weight (0–1)
            if edge_type == 'is_a' and ic_dict:
                if v.startswith('HP:'):
                    ic_value = ic_dict.get(v, 0.5)
                    final_weight = ic_value
                elif u.startswith('HP:'):
                    ic_value = ic_dict.get(u, 0.5)
                    final_weight = ic_value

        edge_weights_list.append(final_weight)
        
    edge_weights_tensor = torch.tensor(edge_weights_list, dtype=torch.float64).to(device)
    
    # Verify weights
    print("\n[Data Verification] Checking Scaled Edge Weights:")
    found_symptom = 0
    found_other = 0
    for i, (u, v) in enumerate(edges):
        data = G[u][v]
        etype = data.get('type')
        if etype == 'has_symptom' and found_symptom < 5:
            print(f"  [Symptom] {u} -> {v} | Raw: {data.get('weight')} | Scaled: {edge_weights_list[i]}")
            found_symptom += 1
        elif etype != 'has_symptom' and found_other < 2:
            print(f"  [ Other ] {u} -> {v} | Raw: {data.get('weight')} | Scaled: {edge_weights_list[i]}")
            found_other += 1
        if found_symptom >= 5 and found_other >= 2:
            break
            
    model = PoincareModel(len(nodes), embedding_dim=128).to(device)
    # RiemannianAdam (handles manifold projection)
    optimizer = geoopt.optim.RiemannianAdam([model.embeddings], lr=0.005, betas=(0.9, 0.999))
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=50
    )
    
    margin_val = 0.3

    # Preprocess positive pairs
    u_list, v_list = zip(*edges)
    pos_u = torch.tensor([node2id[n] for n in u_list], dtype=torch.long).to(device)
    pos_v = torch.tensor([node2id[n] for n in v_list], dtype=torch.long).to(device)
    
    print("Start Hierarchical Contrastive Training with Scaled Weights...")
    epochs = 1000
    
    rng = np.random.RandomState(seed)
    neg_v_indices_all = torch.tensor(
        rng.randint(0, len(nodes), (epochs, len(edges))), 
        dtype=torch.long, device=device
    )
    neg_u_indices_all = torch.tensor(
        rng.randint(0, len(nodes), (epochs, len(edges))), 
        dtype=torch.long, device=device
    )
    
    best_loss = float('inf')
    patience_counter = 0
    patience = 100
    
    for epoch in range(epochs):
        optimizer.zero_grad()

        # 1. Positive pair distances
        dist_pos = model(pos_u, pos_v)

        # 2. Negative sampling
        neg_v_indices = neg_v_indices_all[epoch]
        dist_neg_tail = model(pos_u, neg_v_indices)

        neg_u_indices = neg_u_indices_all[epoch]
        dist_neg_head = model(neg_u_indices, pos_v)

        # 3. Weighted margin ranking loss
        raw_loss_tail = torch.relu(dist_pos - dist_neg_tail + margin_val)
        raw_loss_head = torch.relu(dist_pos - dist_neg_head + margin_val)

        # Apply edge weights
        weighted_loss_tail = raw_loss_tail * edge_weights_tensor
        weighted_loss_head = raw_loss_head * edge_weights_tensor

        # Mean
        total_loss = torch.mean(weighted_loss_tail) + torch.mean(weighted_loss_head)

        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        # RiemannianAdam does manifold projection; no need for project_into_ball

        scheduler.step(total_loss.item())
        
        if total_loss.item() < best_loss:
            best_loss = total_loss.item()
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch}")
                break
        
        if epoch % 10 == 0:
            current_lr = optimizer.param_groups[0]['lr']
            print(f"Epoch {epoch}, Loss: {total_loss.item():.4f}, LR: {current_lr:.6f}")

    return model, node2id, id2node

# ==========================================
# 5. Validation and save
# ==========================================
def inspect_hierarchy(model, node2id, id2node):
    print("\n=== Hierarchy Inspection (Norm Analysis) ===")
    model.eval()
    with torch.no_grad():
        all_indices = torch.arange(len(node2id), device=next(model.parameters()).device)
        vecs = model.embeddings[all_indices]
        norms = torch.norm(vecs, dim=1).cpu().numpy()
        data = []
        for i, norm in enumerate(norms):
            data.append((id2node[i], norm))
        data.sort(key=lambda x: x[1])
        
        total_nodes = len(data)
        if total_nodes <= 20:
            selected_data = data
        else:
            selected_data = data[:10] + data[-10:]
        
        print(f"{'Node Name':<20} | {'Norm (Depth)':<10} | {'Type'}")
        print("-" * 45)
        for name, norm in selected_data:
            n_type = "Root/Abstract" if norm < 0.5 else "Leaf/Specific"
            print(f"{name:<20} | {norm:.4f}     | {n_type}")

def save_checkpoint(model, node2id, id2node, filename_prefix="poincare_model"):
    torch.save(model.state_dict(), f"{filename_prefix}.pth")
    vocab_data = {"node2id": node2id, "id2node": id2node}
    with open(f"{filename_prefix}_vocab.pkl", "wb") as f:
        pickle.dump(vocab_data, f)
    print(f"Model saved to {filename_prefix}.pth")

# ==========================================
# 6. Inference
# ==========================================
def get_all_diseases():
    diseases = set()
    path_dp = os.path.join(KG_DIR, "disease_to_phenotype_edges.csv")
    if os.path.exists(path_dp):
        with open(path_dp, 'r', encoding='utf-8') as f:
            for row in csv.DictReader(f):
                d = (row.get('sourceID') or '').strip()
                if d:
                    diseases.add(d)
    path_gd = os.path.join(KG_DIR, "gene_to_disease_edges.csv")
    if os.path.exists(path_gd):
        with open(path_gd, 'r', encoding='utf-8') as f:
            for row in csv.DictReader(f):
                d = (row.get('disease_id') or '').strip()
                if d:
                    diseases.add(d)
    return sorted(list(diseases))

def diagnose_patient(model, node2id, patient_phenotypes, all_diseases, G=None):
    device = next(model.parameters()).device 
    model.eval()
    results = []
    p_indices = [node2id[p] for p in patient_phenotypes if p in node2id]
    if not p_indices: return []
    p_tensor = torch.tensor(p_indices, dtype=torch.long).to(device)
    
    if G is None:
        disease_to_phenotypes = {}
        path_dp = os.path.join(KG_DIR, "disease_to_phenotype_edges.csv")
        if os.path.exists(path_dp):
            with open(path_dp, 'r', encoding='utf-8') as f:
                for row in csv.DictReader(f):
                    if (row.get('relationship') or '').strip().lower() != 'has':
                        continue
                    d = (row.get('sourceID') or '').strip()
                    hp = (row.get('targetID') or '').strip()
                    if hp and d and d in node2id and hp in node2id:
                        disease_to_phenotypes.setdefault(d, []).append(hp)
    else:
        disease_to_phenotypes = {}
        for disease in all_diseases:
            if disease in G:
                phenotypes = [n for n in G.successors(disease) if n.startswith("HP:") and n in node2id]
                if len(phenotypes) > 0: disease_to_phenotypes[disease] = phenotypes
    
    with torch.no_grad():
        pheno_vectors = model.embeddings[p_tensor]
        p_vector = torch.mean(pheno_vectors, dim=0, keepdim=True)
        for disease in all_diseases:
            if disease not in node2id: continue
            if disease not in disease_to_phenotypes or len(disease_to_phenotypes[disease]) == 0: continue
            disease_phenotypes = disease_to_phenotypes[disease]
            disease_pheno_indices = [node2id[p] for p in disease_phenotypes if p in node2id]
            if len(disease_pheno_indices) == 0: continue
            disease_pheno_tensor = torch.tensor(disease_pheno_indices, dtype=torch.long).to(device)
            disease_pheno_embs = model.embeddings[disease_pheno_tensor]
            d_vector = torch.mean(disease_pheno_embs, dim=0, keepdim=True)
            p_vector_exp = p_vector.unsqueeze(0)
            d_vector_exp = d_vector.unsqueeze(0)
            score = model.poincare_distance(p_vector_exp, d_vector_exp).item()
            
            results.append((disease, score))
    results.sort(key=lambda x: x[1])
    return results

def calculate_chamfer_distance(model, node2id, target_phenos, candidate_phenos):
    device = next(model.parameters()).device
    t_indices = [node2id[p] for p in target_phenos if p in node2id]
    c_indices = [node2id[p] for p in candidate_phenos if p in node2id]
    if not t_indices or not c_indices: return float('inf')
    t_tensor = torch.tensor(t_indices, dtype=torch.long).to(device)
    c_tensor = torch.tensor(c_indices, dtype=torch.long).to(device)
    with torch.no_grad():
        u = model.embeddings[t_tensor]
        v = model.embeddings[c_tensor]
        u_exp = u.unsqueeze(1)
        v_exp = v.unsqueeze(0)
        dist_matrix = model.poincare_distance(u_exp, v_exp)
        min_dist_A_to_B, _ = torch.min(dist_matrix, dim=1)
        term1 = torch.mean(min_dist_A_to_B)
        min_dist_B_to_A, _ = torch.min(dist_matrix, dim=0)
        term2 = torch.mean(min_dist_B_to_A)
        total_dist = term1 + term2
        return total_dist.item()

def find_similar_patients(model, node2id, target_p, candidate_db):
    results = []
    for candidate in candidate_db:
        c_p = candidate['phenotypes']
        dist = calculate_chamfer_distance(model, node2id, target_p, c_p)
        results.append((candidate['id'], dist, c_p))
    results.sort(key=lambda x: x[1])
    return results

# ==========================================
# 7. Main
# ==========================================
if __name__ == "__main__":
    print("=== Step 1: Training Poincaré Model with Frequency Info ===")
    trained_model, n2i, i2n = train_model()

    inspect_hierarchy(trained_model, n2i, i2n)
    save_checkpoint(trained_model, n2i, i2n)

    target_patient = [
      "HP:0000722", "HP:0000750", "HP:0001319", "HP:0001350", "HP:0001530",
      "HP:0001558", "HP:0001612", "HP:0002033", "HP:0002577", "HP:0004324",
      "HP:0011951", "HP:0030748", "HP:0040217"
    ]
    print(f"\nTarget Patient: {target_patient}")

    # 2. Diagnosis
    print("\n=== Step 2: Diagnosis ===")
    candidate_diseases = get_all_diseases()
    print(f"Total diseases in database: {len(candidate_diseases)}")
    diag_results = diagnose_patient(trained_model, n2i, target_patient, candidate_diseases)
    
    for rank, (disease, score) in enumerate(diag_results[:10]):
        print(f"Rank {rank+1}: {disease} | Dist: {score:.4f}")

    # 2.5. Output all phenotype embeddings
    print("\n=== Step 2.5: Output All Phenotype Embeddings ===")
    phenotype_nodes = [node for node in n2i.keys() if node.startswith("HP:")]
    if not phenotype_nodes:
        print("Warning: No phenotype nodes found")
    else:
        device = next(trained_model.parameters()).device
        trained_model.eval()
        results = []
        with torch.no_grad():
            phenotype_indices = torch.tensor([n2i[phenotype] for phenotype in phenotype_nodes], dtype=torch.long).to(device)
            emb_matrix = trained_model.embeddings[phenotype_indices].cpu().numpy()
            for idx, phenotype in enumerate(phenotype_nodes):
                vector = emb_matrix[idx]
                vec_str = ",".join([f"{v:.5f}" for v in vector])
                results.append({'node_type': 'phenotype', 'id': phenotype, 'embedding': vec_str})

        output_file = 'final_poincare_phenotype_embeddings.csv'
        df_res = pd.DataFrame(results)
        df_res.to_csv(output_file, index=False)
        print(f"Saved {len(df_res)} phenotype embeddings to '{output_file}'")

    # 3. Similar case search
    print("\n=== Step 3: Similar Case Search ===")
    patient_db = [
        {"id": "Case_1 (Match)", "phenotypes": ["HP:0000365", "HP:0001263"]},
        {"id": "Case_2 (Noise)", "phenotypes": ["HP:0001250"]},
        {"id": "Case_3 (Mixed)", "phenotypes": ["HP:0000400", "HP:0001250"]}
    ]
    sim_results = find_similar_patients(trained_model, n2i, target_patient, patient_db)
    for rank, (pid, dist, phenos) in enumerate(sim_results):
        print(f"Rank {rank+1}: {pid} | Dist: {dist:.4f} | Phenos: {phenos}")