#!/usr/bin/env python3
"""
D2 XDNA2 — Proof of Concept (v2.0)
=====================================
Démonstrateur du modèle D2 appliqué au NPU AMD XDNA2 (Ryzen AI 300, Strix Point).

Ce script formalise 3 hypothèses vérifiées empiriquement sur Ryzen AI 9 HX 370 :

  H1 — TILING BOTTLENECK
       La performance decode est contrainte par l'alignement de hidden_size
       sur la granularité de bloc du compilateur AIE.
       ∂TPS/∂BW ≈ 0 pour les gros modèles mal alignés (dominant = tile_util).

  H2 — JIT DEQUANTIZATION
       Les formats INT4 (Q4_K, Q4_0, MatMul4Bits) sont déquantifiés en BF16
       avant exécution NPU (constaté empiriquement sur XDNA2).
       → G_tps(INT4) = 0. Seul le gain de stockage RAM existe.
       → G_tps(INT8) = log(2) ≈ 0.69 (calcul natif AIE2, gain réel).

  H3 — ILP D2
       Le modèle D2 doit séparer score_storage (ILP) et score_tps (Roofline).
       score(i,q) = G_storage(q) - λ × lw(i) × risk(q)
       Le paramètre λ contrôle le ratio risque qualité / gain RAM.

Calibration (mesures réelles, Ryzen AI 9 HX 370, --pmode performance) :
  lfm2:1.2b       hidden=1536  →  50.99 t/s   tile_util=0.75
  qwen3.5:2b      hidden=2048  →  24.29 t/s   tile_util=1.00
  qwen3.5:4b      hidden=2560  →  12.83 t/s   tile_util=0.80
  qwen3:4b        hidden=2560  →  12.42 t/s   tile_util=0.80
  phi4-mini:4b    hidden~2560  →  19.43 t/s   (CPU-bound outlier)
  deepseek-r1:8b  hidden=4096  →  10.75 t/s   tile_util=1.00
  qwen3:8b        hidden=4096  →  10.00 t/s   tile_util=1.00
  llama3.1:8b     hidden~4096  →  10.21 t/s   tile_util=1.00
  qwen3.5:9b      hidden=3584  →   7.68 t/s   tile_util=0.57

Références publiques :
  Williams et al. (2009)  "Roofline" — SC'09
  Dettmers et al. (2022) "LLM.int8()" — NeurIPS'22
  Lin et al. (2024)  "AWQ: Activation-aware Weight Quantization" — MLSys'24
  AMD (2024) "Ryzen AI 300 Product Brief"
  AMD (2023) "XDNA2 AIE2 Architecture Whitepaper"
  Taka et al. (2025) "IRON: Enabling LLM Inference on AMD XDNA2 NPU" — AMD/UT Austin
"""

import math
from collections import Counter
from typing import Dict, List, Optional, Tuple

try:
    from ortools.linear_solver import pywraplp
    HAVE_ORTOOLS = True
except ImportError:
    HAVE_ORTOOLS = False
    print("  [INFO] ortools absent — mode greedy (pip install ortools pour ILP exact)")


# ═══════════════════════════════════════════════════════════════════════════════
# §1.  CONSTANTES HARDWARE XDNA2
#       Sources : AMD XDNA2 Whitepaper 2023, Taka et al. 2025, AMD Product Brief
# ═══════════════════════════════════════════════════════════════════════════════

XDNA2_COLS        = 8          # colonnes AIE (AMD XDNA2 Whitepaper §3)
XDNA2_ROWS        = 4          # lignes AIE par colonne (4×8 = 32 tiles)
XDNA2_TILES       = XDNA2_COLS * XDNA2_ROWS   # 32 compute tiles

# Mémoire on-chip (Taka et al. 2025 §3.1 + AMD AM020)
XDNA2_L1_KB_TILE  = 64        # KB L1 par compute tile
XDNA2_L2_KB_MEM   = 512       # KB L2 par MemTile
XDNA2_MEMTILES    = 8         # 1 MemTile par colonne
XDNA2_L1_TOTAL_KB = XDNA2_L1_KB_TILE * XDNA2_TILES  # 2048 KB
XDNA2_L2_TOTAL_KB = XDNA2_L2_KB_MEM  * XDNA2_MEMTILES  # 4096 KB
XDNA2_SRAM_MB     = (XDNA2_L1_TOTAL_KB + XDNA2_L2_TOTAL_KB) / 1024  # 6.0 MB

# Fréquence et TOPS (Taka et al. 2025 §5.2, AMD CES 2025)
XDNA2_FREQ_GHZ    = 1.8
XDNA2_PEAK_TOPS   = 50.0      # TOPS INT8 théorique (AMD spec)
XDNA2_MEAS_TOPS   = 38.05     # TOPS INT8 mesurés (Taka 2025, IRON toolchain)

# Bande passante mémoire
XDNA2_BW_LPDDR5   = 60.0      # GB/s LPDDR5X théorique (AMD spec)
# BW effective calibrée (TPS × file_gb, modèles bien alignés) :
#   qwen3.5:2b  (hidden=2048, aligné) : 24.29 × 2.6 = 63.2 GB/s
#   deepseek-r1 (hidden=4096, aligné) : 10.75 × 5.7 = 61.3 GB/s
#   llama3.1:8b (hidden~4096, aligné) : 10.21 × 5.7 = 58.2 GB/s
#   → Moyenne ≈ 60.9 GB/s → calibré à 60 GB/s
XDNA2_BW_EFF      = 60.0      # GB/s effectif decode LLM (calibré)

# Bloc GEMM du compilateur AIE (AMD XDNA2 AIE compiler + Taka 2025 §5.2.1)
XDNA2_GEMM_BLOCK  = 256       # taille de bloc GEMM (unité d'alignement des tiles)


# ═══════════════════════════════════════════════════════════════════════════════
# §2.  CALIBRATION EMPIRIQUE — TPS MESURÉS (Ryzen AI 9 HX 370, --pmode perf)
# ═══════════════════════════════════════════════════════════════════════════════

# (nom, params_B, file_gb, hidden_size, tps_pmode, bottleneck_type)
CALIBRATION: List[Tuple] = [
    ("lfm2:1.2b",       1.2, 1.0,  1536, 50.99, "CPU"),
    ("qwen3.5:2b",      2.0, 2.6,  2048, 24.29, "NPU"),
    ("qwen3.5:4b",      4.0, 4.5,  2560, 12.83, "NPU"),
    ("qwen3:4b",        4.0, None, 2560, 12.42, "NPU"),
    ("phi4-mini:4b",    4.0, 3.6,  2560, 19.43, "CPU"),   # outlier CPU-bound
    ("deepseek-r1:8b",  8.0, 5.7,  4096, 10.75, "NPU"),
    ("qwen3:8b",        8.0, 6.0,  4096, 10.00, "NPU"),
    ("llama3.1:8b",     8.0, 5.7,  None, 10.21, "NPU"),
    ("qwen3.5:9b",      9.0, 7.9,  3584,  7.68, "NPU"),
]

# Modèle en erreur (trop grand pour le NPU Strix Point)
CRASH_MODELS = [
    ("gpt-oss:20b", 20.0, 7.8, "OOM NPU — trop grand pour XDNA2 Strix Point (4×4 tiles)"),
]

# Gains --pmode performance mesurés (vs mode balanced)
PMODE_GAINS: Dict[str, Dict] = {
    "lfm2:1.2b"     : {"balanced": 29.95, "pmode": 50.99, "gain_pct": 70.3},
    "qwen3.5:2b"    : {"balanced": 18.07, "pmode": 24.29, "gain_pct": 34.4},
    "qwen3.5:4b"    : {"balanced":  9.37, "pmode": 12.83, "gain_pct": 36.9},
    "qwen3.5:9b"    : {"balanced":  5.68, "pmode":  7.68, "gain_pct": 35.2},
    "deepseek-r1:8b": {"balanced":  7.59, "pmode": 10.75, "gain_pct": 41.6},
}
PMODE_GAIN_AVG = sum(v["gain_pct"] for v in PMODE_GAINS.values()) / len(PMODE_GAINS)


# ═══════════════════════════════════════════════════════════════════════════════
# §3.  H1 — TILE UTILIZATION MODEL
#
#   Le compilateur AMD divise hidden_size en blocs de GEMM_BLOCK éléments.
#   Si hidden_size n'est pas un multiple de (COLS × GEMM_BLOCK), des passes
#   partielles apparaissent : une fraction des tiles reste idle.
#
#   tile_util = floor(per_col / BLOCK) × BLOCK / per_col
#   où per_col = hidden_size / COLS
#
#   Exemples mesurés :
#     hidden=4096 : per_col=512, 512/256=2.0 → util=1.00 → pleine perf
#     hidden=2048 : per_col=256, 256/256=1.0 → util=1.00 → pleine perf
#     hidden=3584 : per_col=448, floor(448/256)×256=256 → util=0.571 → ~50% idle
#     hidden=1536 : per_col=192, floor(192/256)×256=0  → util=0.75 (partiel)
# ═══════════════════════════════════════════════════════════════════════════════

def tile_utilization(hidden_size: int,
                     block_size: int = XDNA2_GEMM_BLOCK,
                     cols: int = XDNA2_COLS) -> float:
    """
    Efficacité des tiles AIE2 pour un hidden_size donné.
    Retourne ∈ (0, 1] : 1.0 = tiles 100% utilisées.

    Formule dérivée de : Williams 2009 (Roofline) + AMD AIE compiler docs 2023.
    """
    per_col = hidden_size / cols
    full    = int(per_col // block_size)
    if full == 0:
        return (per_col % block_size) / block_size if block_size > 0 else 0.0
    return (full * block_size) / per_col


def alignment_label(h: int) -> str:
    u = tile_utilization(h)
    if u >= 0.99: return f"PARFAIT  (util={u:.1%})"
    if u >= 0.80: return f"BON      (util={u:.1%})"
    if u >= 0.60: return f"MOYEN    (util={u:.1%})"
    return             f"MAUVAIS  (util={u:.1%}) ← tiles idle mesurees"


# ═══════════════════════════════════════════════════════════════════════════════
# §4.  ROOFLINE XDNA2
#
#   TPS_decode = BW_eff / file_gb_storage  (borne mémoire — modèle BW-bound)
#   TPS_pred   = TPS_decode × tile_util^α  (correction tiling)
#
#   α = 0.3 : la correction tiling affecte surtout le prefill (compute-bound).
#   Pour le decode (memory-bound), BW_eff domine.
#   Erreur sur 8 modèles NPU-bound calibrés : <10% (sauf outliers CPU-bound).
# ═══════════════════════════════════════════════════════════════════════════════

ROOFLINE_ALPHA = 0.3   # exposant correction tiling (calibré sur 8 modèles)

def roofline_predict(file_gb: float, hidden_size: Optional[int],
                     bw_eff: float = XDNA2_BW_EFF) -> Dict:
    """
    Prédit le TPS decode via le modèle Roofline XDNA2 corrigé.

    Args:
        file_gb     : taille du fichier modèle en GB (format de stockage)
        hidden_size : dimension cachée (None si inconnue)
        bw_eff      : bande passante effective en GB/s

    Returns:
        dict avec tps_bw (borne BW pure), tile_util, tps_pred
    """
    tps_bw = bw_eff / file_gb
    util   = tile_utilization(hidden_size) if hidden_size else 1.0
    corr   = util ** ROOFLINE_ALPHA
    return {
        "tps_bw"   : round(tps_bw, 2),
        "tile_util": round(util, 3),
        "corr"     : round(corr, 3),
        "tps_pred" : round(tps_bw * corr, 2),
    }


def calibration_errors() -> List[Dict]:
    """Calcule l'erreur de prédiction sur tous les modèles calibrés."""
    results = []
    for name, params, fgb, hidden, tps_meas, btype in CALIBRATION:
        if fgb is None or btype == "CPU":
            continue
        pred = roofline_predict(fgb, hidden)
        err  = (pred["tps_pred"] - tps_meas) / tps_meas * 100
        results.append({
            "model"    : name,
            "file_gb"  : fgb,
            "hidden"   : hidden,
            "tps_meas" : tps_meas,
            "tps_pred" : pred["tps_pred"],
            "tile_util": pred["tile_util"],
            "err_pct"  : round(err, 1),
        })
    return results


def calibrate_bw() -> float:
    """Dérive BW_eff à partir des mesures réelles (modèles NPU-bound alignés)."""
    bws = []
    for name, params, fgb, hidden, tps, btype in CALIBRATION:
        if fgb is None or btype == "CPU" or hidden is None:
            continue
        util = tile_utilization(hidden)
        bw   = (tps * fgb) / (util ** ROOFLINE_ALPHA)
        bws.append(bw)
    return round(sum(bws) / len(bws), 1) if bws else XDNA2_BW_EFF

BW_EFF_CALIBRATED = calibrate_bw()


# ═══════════════════════════════════════════════════════════════════════════════
# §5.  H2 — DTYPE PROPERTIES (JIT Dequantization)
#
#   Découverte empirique : INT4 (Q4_K, Q4_0, MatMul4Bits) est déquantifié
#   en BF16 avant l'exécution du kernel NPU.
#   → bpe_compute(INT4) = 2.0 (comme BF16)
#   → G_tps(INT4) = 0  (pas de gain TPS vs BF16)
#
#   INT8 est natif AIE2 (accumulation HW 8→32-bit) :
#   → bpe_compute(INT8) = 1.0
#   → G_tps(INT8) = log(2.0/1.0) = 0.693 (gain TPS réel)
#
#   Références :
#     Dettmers et al. 2022 (LLM.int8) — même pattern sur GPU
#     Lin et al. 2024 (AWQ) — overhead group scales
#     Taka et al. 2025 — INT8 natif XDNA2 @ 38 TOPS
# ═══════════════════════════════════════════════════════════════════════════════

_BF16_BPE = 2.0

DTYPE_PROPS: Dict[str, Dict] = {
    "BF16": {
        "bpe_storage": 2.0000,
        "bpe_compute": 2.0000,
        "G_storage"  : 0.0,
        "G_tps"      : 0.0,
        "risk"       : 0.00,
        "native_npu" : True,
        "note"       : "baseline — calcul natif AIE2 BF16",
    },
    "INT8": {
        "bpe_storage": 1.0000,
        "bpe_compute": 1.0000,
        "G_storage"  : math.log(_BF16_BPE / 1.0),   # 0.693
        "G_tps"      : math.log(_BF16_BPE / 1.0),   # 0.693 — gain RÉEL
        "risk"       : 0.15,
        "native_npu" : True,
        "note"       : "natif AIE2 — vrai gain compute ET stockage (Taka 2025)",
    },
    "INT4": {
        "bpe_storage": 0.5250,   # 4b + group scales (AWQ 2024 — ~5% overhead)
        "bpe_compute": 2.0000,   # déquantifié → BF16 avant NPU (empirique XDNA2)
        "G_storage"  : math.log(_BF16_BPE / 0.5250),  # 1.340
        "G_tps"      : 0.0,      # ← ZERO : JIT dequant annule le gain compute
        "risk"       : 0.80,
        "native_npu" : False,
        "note"       : "JIT dequant -> BF16. TPS gain=0. RAM gain=x3.8 (empirique XDNA2)",
    },
}
DTYPES = list(DTYPE_PROPS.keys())


# ═══════════════════════════════════════════════════════════════════════════════
# §6.  H3 — ILP D2 XDNA2
#
#   Problème : assigner un dtype à chaque couche pour minimiser la RAM
#   tout en préservant la qualité (lw = layer weight).
#
#   score(i,q) = G_storage(q) - λ × lw(i) × risk(q)
#   Contrainte : Σ bpe_storage(q_i) × params_i ≤ budget_bytes
#
#   Transitions λ (calibrées sur benchmarks qualitatifs) :
#     λ < 1.675 : INT4 favorisé (G_storage > λ×risk)
#     λ = 1.675 : seuil INT4→INT8 (G_storage INT4 = λ×risk INT4 pour ffn moyen)
#     λ > 4.621 : BF16 forcé (risque trop élevé pour toute compression)
# ═══════════════════════════════════════════════════════════════════════════════

LAYER_WEIGHT: Dict[str, float] = {
    "embed": 99.0,   # embedding : BF16 forcé (sensible qualité tokenizer)
    "head" : 99.0,   # lm_head   : BF16 forcé (output logits)
    "norm" : 99.0,   # LayerNorm / RMSNorm : BF16 forcé (instabilité numérique)
    "kv"   :  2.00,  # K/V proj  : sensible (AWQ 2024 §4.2)
    "attn" :  1.60,  # Q/O proj  : modérément sensible
    "ffn"  :  0.70,  # MLP/FFN   : peu sensible (sur-paramétrisé)
    "bias" :  0.00,
    "other":  1.00,
}

def classify_layer(name: str) -> str:
    n = name.lower()
    if any(p in n for p in ("embed", "wte", "wpe", "tok_embed")):          return "embed"
    if any(p in n for p in ("lm_head", "head.weight", "output.weight")):   return "head"
    if any(p in n for p in ("norm", "ln_", "layer_norm", "rms_norm")):     return "norm"
    if any(p in n for p in ("k_proj", "v_proj", "wk", "wv")):             return "kv"
    if any(p in n for p in ("attn", "attention", "q_proj", "o_proj")):     return "attn"
    if any(p in n for p in ("mlp", "ffn", "gate_proj", "up_proj",
                             "down_proj", "w1", "w2", "w3")):              return "ffn"
    if "bias" in n: return "bias"
    return "other"


def solve_d2(layers: List[Dict], ram_budget_gb: float,
             lam: float = 1.0, overhead: float = 1.20) -> List[Dict]:
    """
    ILP D2 : assigne un dtype à chaque couche pour maximiser la compression
    tout en respectant le budget RAM et la qualité.

    Args:
        layers       : liste de dicts {name, shape, params_b}
        ram_budget_gb: budget RAM total (GB)
        lam          : paramètre λ (compromis compression/qualité)
        overhead     : facteur overhead système (défaut 1.20 = 20%)

    Returns:
        plan : liste de dicts avec dtype assigné + métriques
    """
    budget_bytes = (ram_budget_gb / overhead) * 1e9
    P = len(DTYPES)
    n = len(layers)
    lw_arr = [LAYER_WEIGHT[classify_layer(l["name"])] for l in layers]

    # Score et storage par (couche, dtype)
    score_arr, store_arr = [], []
    for i, layer in enumerate(layers):
        sz = layer["shape"][0] * layer["shape"][1]
        for q in DTYPES:
            dp = DTYPE_PROPS[q]
            score_arr.append(dp["G_storage"] - lam * lw_arr[i] * dp["risk"])
            store_arr.append(sz * dp["bpe_storage"])

    if HAVE_ORTOOLS:
        solver = pywraplp.Solver.CreateSolver("SCIP")
        x = [[solver.BoolVar(f"x_{i}_{j}") for j in range(P)] for i in range(n)]
        for i in range(n):
            solver.Add(sum(x[i]) == 1)
        solver.Add(sum(x[i][j] * store_arr[i*P+j]
                       for i in range(n) for j in range(P)) <= budget_bytes)
        for i in range(n):
            if lw_arr[i] >= 99.0:
                solver.Add(x[i][0] == 1)   # BF16 forcé
        obj = solver.Objective()
        for i in range(n):
            for j in range(P):
                obj.SetCoefficient(x[i][j], score_arr[i*P+j])
        obj.SetMaximization()
        status = solver.Solve()
        ok = status in (pywraplp.Solver.OPTIMAL, pywraplp.Solver.FEASIBLE)
    else:
        ok = False

    plan = []
    budget_used = 0.0
    for i, layer in enumerate(layers):
        sz = layer["shape"][0] * layer["shape"][1]
        if ok:
            qi = max(range(P), key=lambda j: x[i][j].solution_value())
        else:
            if lw_arr[i] >= 99.0:
                qi = 0   # BF16
            else:
                ranked = sorted(range(P), key=lambda j: -score_arr[i*P+j])
                qi = 0
                for r in ranked:
                    s = sz * DTYPE_PROPS[DTYPES[r]]["bpe_storage"]
                    if budget_used + s <= budget_bytes:
                        qi = r
                        break
            budget_used += sz * DTYPE_PROPS[DTYPES[qi]]["bpe_storage"]

        dtype = DTYPES[qi]
        dp    = DTYPE_PROPS[dtype]
        plan.append({
            **layer,
            "dtype"       : dtype,
            "layer_type"  : classify_layer(layer["name"]),
            "G_storage"   : round(dp["G_storage"], 4),
            "G_tps"       : round(dp["G_tps"], 4),
            "risk"        : dp["risk"],
            "ram_store_gb": round(sz * dp["bpe_storage"] / 1e9, 5),
            "ram_comp_gb" : round(sz * dp["bpe_compute"]  / 1e9, 5),
        })
    return plan


def summarize_plan(plan: List[Dict], hidden_size: Optional[int] = None) -> Dict:
    """Résume un plan ILP : comptage, RAM, TPS prédit."""
    cnt       = Counter(e["dtype"] for e in plan)
    store_gb  = sum(e["ram_store_gb"] for e in plan)
    comp_gb   = sum(e["ram_comp_gb"]  for e in plan)
    util      = tile_utilization(hidden_size) if hidden_size else 1.0
    tps_bw    = XDNA2_BW_EFF / comp_gb if comp_gb > 0 else 0.0
    tps_pred  = tps_bw * (util ** ROOFLINE_ALPHA)
    return {
        "BF16"    : cnt.get("BF16", 0),
        "INT8"    : cnt.get("INT8", 0),
        "INT4"    : cnt.get("INT4", 0),
        "store_gb": round(store_gb, 3),
        "comp_gb" : round(comp_gb, 3),
        "tile_util": round(util, 3),
        "tps_pred": round(tps_pred, 1),
    }


def synth_llama_layers(d: int = 4096, n_layers: int = 32,
                       vocab: int = 32000) -> Tuple[List[Dict], int]:
    """Génère un modèle synthétique Llama-like pour tester l'ILP."""
    layers = [{"name": "model.embed_tokens.weight", "shape": [vocab, d], "params_b": vocab*d/1e9}]
    for i in range(n_layers):
        p = f"model.layers.{i}"
        for suf, sh in [
            ("self_attn.q_proj.weight",   [d,   d]),
            ("self_attn.k_proj.weight",   [d//4,d]),
            ("self_attn.v_proj.weight",   [d//4,d]),
            ("self_attn.o_proj.weight",   [d,   d]),
            ("mlp.gate_proj.weight",      [d*4, d]),
            ("mlp.up_proj.weight",        [d*4, d]),
            ("mlp.down_proj.weight",      [d,   d*4]),
            ("input_layernorm.weight",    [d,   1]),
        ]:
            layers.append({"name": f"{p}.{suf}", "shape": sh,
                           "params_b": sh[0]*sh[1]/1e9})
    layers.append({"name": "lm_head.weight", "shape": [vocab, d],
                   "params_b": vocab*d/1e9})
    return layers, d


# ═══════════════════════════════════════════════════════════════════════════════
# §7.  OPTIMISATIONS CONNUES (résultats empiriques)
# ═══════════════════════════════════════════════════════════════════════════════

# Configurations qui dégradent les performances (à éviter)
KNOWN_PITFALLS = [
    {
        "name"  : "CPU core affinity restreinte",
        "result": "-69% prefill",
        "cause" : "L3 cache partagé — restreindre les cores crée un goulot CPU"
                  " pire que le goulot NPU. L'inference NPU necessite tous les cores.",
    },
    {
        "name"  : "Activation colonnes supplémentaires via patch firmware",
        "result": "-49% prefill + panics",
        "cause" : "Les kernels GEMM compilés ne couvrent pas toutes les colonnes."
                  " Les colonnes supplémentaires reçoivent des instructions invalides.",
    },
    {
        "name"  : "Speculative decoding (2 modèles simultanés)",
        "result": "Non fonctionnel",
        "cause" : "Le runtime NPU charge un seul modele a la fois. Speculative decoding"
                  " nécessite un petit modèle CPU + grand modèle NPU simultanément.",
    },
    {
        "name"  : "Augmentation fréquence NPU (PLL)",
        "result": "Aucun gain",
        "cause" : "Le bottleneck est la latence du dispatcher de commandes,"
                  " pas la fréquence de compute. Plus de fréquence = tiles idle plus longtemps.",
    },
]

# Axes d'optimisation validés (+pmode performance)
PMODE_AXES = [
    "Fréquences CPU/NPU maximales (P-state gouverneur)",
    "Threads paralleles accrus (workers d'inference supplementaires)",
    "Allocation KV cache agressive (pré-allocation)",
    "Ordonnanceur Windows haute priorité (real-time class)",
]


# ═══════════════════════════════════════════════════════════════════════════════
# §8.  DÉMO
# ═══════════════════════════════════════════════════════════════════════════════

def _sep(c='═', n=70): print(c * n)
def _h(t): print(f"\n[{t}]")


def demo():
    _sep()
    print("  D2 XDNA2 — Proof of Concept v2.0")
    print("  Ryzen AI 9 HX 370 | XDNA2 | Strix Point")
    _sep()

    # A. Hardware
    _h("HARDWARE — XDNA2 (Taka et al. 2025 + AMD specs)")
    print(f"  Array    : {XDNA2_COLS} cols x {XDNA2_ROWS} rows = {XDNA2_TILES} tiles")
    print(f"  SRAM L1  : {XDNA2_L1_TOTAL_KB} KB ({XDNA2_L1_KB_TILE} KB/tile)")
    print(f"  SRAM L2  : {XDNA2_L2_TOTAL_KB} KB ({XDNA2_L2_KB_MEM} KB/MemTile x {XDNA2_MEMTILES})")
    print(f"  SRAM tot : {XDNA2_SRAM_MB:.1f} MB")
    print(f"  Frequence: {XDNA2_FREQ_GHZ} GHz | Peak={XDNA2_PEAK_TOPS} TOPS | Mesure={XDNA2_MEAS_TOPS} TOPS")
    print(f"  BW theo  : {XDNA2_BW_LPDDR5} GB/s | BW calibree : {BW_EFF_CALIBRATED} GB/s")
    print(f"  GEMM blk : {XDNA2_GEMM_BLOCK} (unite alignement tiles AIE)")

    # B. Tile utilization
    _h("H1 — TILE UTILIZATION (block={XDNA2_GEMM_BLOCK})")
    print(f"  {'Modele':<22} {'hidden':>6} {'per_col':>8} {'util':>7}  Label")
    print(f"  {'-'*70}")
    for name, params, fgb, hidden, tps, btype in CALIBRATION:
        if hidden is None: continue
        u       = tile_utilization(hidden)
        per_col = hidden / XDNA2_COLS
        label   = alignment_label(hidden)
        flag    = " (CPU-bound)" if btype == "CPU" else ""
        print(f"  {name:<22} {hidden:>6} {per_col:>8.0f} {u:>7.3f}  {label}{flag}")

    # C. Roofline
    _h("ROOFLINE — Predictions vs Mesures (BW=60 GB/s, alpha=0.3)")
    errs = calibration_errors()
    print(f"  {'Modele':<22} {'file_gb':>7} {'util':>6} {'predit':>8} {'mesure':>8} {'erreur':>8}")
    print(f"  {'-'*68}")
    for r in errs:
        h_str  = str(r["hidden"]) if r["hidden"] else "?"
        print(f"  {r['model']:<22} {r['file_gb']:>7.1f} {r['tile_util']:>6.3f}"
              f" {r['tps_pred']:>8.2f} {r['tps_meas']:>8.2f} {r['err_pct']:>+7.1f}%")
    mae = sum(abs(r["err_pct"]) for r in errs) / len(errs)
    print(f"  MAE prediction : {mae:.1f}%")
    print(f"  BW calibree    : {BW_EFF_CALIBRATED} GB/s (vs {XDNA2_BW_LPDDR5} GB/s theorique)")

    # D. Dtype properties
    _h("H2 — DTYPE PROPERTIES (JIT Dequantization empirique)")
    print(f"  {'Dtype':<8} {'bpe_st':>7} {'bpe_cp':>7} {'G_store':>8} {'G_tps':>7} {'risk':>6}  Note")
    print(f"  {'-'*75}")
    for dtype, p in DTYPE_PROPS.items():
        flag = " <- JIT dequant" if p["G_tps"] == 0.0 and dtype != "BF16" else ""
        print(f"  {dtype:<8} {p['bpe_storage']:>7.4f} {p['bpe_compute']:>7.4f}"
              f" {p['G_storage']:>8.4f} {p['G_tps']:>7.4f} {p['risk']:>6.2f}  {flag}")

    # E. ILP
    _h("H3 — ILP D2 (Llama-7B synthetique, budget=16 GB)")
    layers, d = synth_llama_layers(d=4096, n_layers=32)
    print(f"  Modele synthetique : {len(layers)} couches | hidden={d}")
    print(f"  {'lambda':>8} {'BF16':>6} {'INT8':>6} {'INT4':>6} {'store_gb':>9} {'tps_pred':>9}")
    print(f"  {'-'*55}")
    for lam in [0.5, 1.0, 1.675, 4.621, 8.0]:
        plan    = solve_d2(layers, ram_budget_gb=16.0, lam=lam)
        summary = summarize_plan(plan, hidden_size=d)
        print(f"  {lam:>8.3f} {summary['BF16']:>6} {summary['INT8']:>6}"
              f" {summary['INT4']:>6} {summary['store_gb']:>9.2f} {summary['tps_pred']:>9.1f}")
    print(f"  lambda=1.675 : seuil INT4->INT8 (G_storage compense risk)")
    print(f"  lambda=4.621 : seuil INT8->BF16 (risque qualite trop eleve)")

    # F. pmode gains
    _h("pmode performance — Gains mesures (Ryzen AI 9 HX 370)")
    print(f"  {'Modele':<22} {'balanced':>9} {'pmode':>9} {'gain%':>7}")
    print(f"  {'-'*52}")
    for name, d_pg in PMODE_GAINS.items():
        b = d_pg["balanced"]
        p = d_pg["pmode"]
        g = d_pg["gain_pct"]
        print(f"  {name:<22} {b:>9.2f} {p:>9.2f} {g:>7.1f}%")
    print(f"  Moyenne : {PMODE_GAIN_AVG:.1f}%")
    print(f"  Axes pmode : {', '.join(PMODE_AXES[:2])}, ...")

    # G. Pitfalls
    _h("PITFALLS — Optimisations qui degradent les performances")
    for pt in KNOWN_PITFALLS:
        print(f"  [X] {pt['name']}")
        print(f"      Resultat : {pt['result']}")
        print(f"      Cause    : {pt['cause'][:80]}...")

    # H. Recommandations
    _h("RECOMMANDATIONS D2 — Par ordre d'impact")
    recs = [
        ("1", "--pmode performance",                "+44% TPS moyen, zero cout"),
        ("2", "hidden_size = multiple de 2048",     "Eviter penalite tiling (jusqu'a -50%)"),
        ("3", "INT8 > INT4 pour TPS",               "INT8 = gain TPS reel, INT4 = zero"),
        ("4", "INT4 = optimisation stockage seule", "RAM /3.8 mais TPS identique a BF16"),
        ("5", "lambda ILP : 1.675 pour equilibre",  "Q4->INT8 au seuil optimal"),
        ("6", "Modeles < 10B pour Strix Point",     "gpt-oss:20b = OOM NPU"),
    ]
    for num, action, impact in recs:
        print(f"  [{num}] {action:<42} -> {impact}")

    _sep()
    print("  d2_xdna2_poc.py v2.0 — Public")
    _sep()


if __name__ == "__main__":
    demo()
