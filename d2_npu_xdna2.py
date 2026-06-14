#!/usr/bin/env python3
"""
D2 NPU XDNA2 — Quantization Planner for AMD Ryzen AI (XDNA2)
==============================================================
Port du framework D2 Production vers l'architecture NPU AMD XDNA2
(Ryzen AI 300 Series, Strix Point — AIE2P tiles).

Différences fondamentales vs GPU (d2_production.py)
────────────────────────────────────────────────────
1. DTYPE_PROPS recalibrés pour AIE2 :
   • BF16  = baseline (AIE2P supporte BF16 nativement via vecteurs 32b)
   • INT8  = datapath NATIF (vecteurs 8b×8b → 32b accumulateur)
             risque réduit vs GPU : accumulation HW évite saturation (Dettmers 2022)
   • INT4  = groupwise quantization (group_size=32, ~5 % metadata overhead)

2. Roofline NPU (Williams et al. SC'09) :
   • P_peak(INT8) = 50 TOPS        — Ryzen AI 300 Product Brief (AMD 2024)
   • BW théorique = 60 GB/s        — LPDDR5X spec Ryzen AI 300 (AMD 2024)
   • BW_eff       ≈ 30 GB/s        — SW efficiency ~50 % (Hauswald et al. 2015,
                                      ARM 2022, typique NPU consumer SW stack)
   • OI_decode ≈ 1–2 FLOPs/byte   → toujours memory-bound pour S=1
   • G(q) = log(BPE_BF16/BPE(q)) reste analytiquement valide

3. Budget RAM partagé (pas de VRAM dédié) :
   • Shared LPDDR5X, typiquement 16–32 GB
   • Overhead KV cache + activations ≈ 20 % du budget modèle
   • Contrainte ajustée : budget_eff = ram_budget / overhead_factor

4. Export ONNX + OGA (ONNX Runtime GenAI) + Olive-AI :
   • Pas de GGUF / llama.cpp
   • Cible : onnxruntime-genai model_builder, olive-ai MatMul4BitsQuantization

5. Seuils λ différents (INT8 natif → risque plus faible) :
   • λ_indiff(INT4) ≈ 1.675  (vs 1.27 GPU)
   • λ_indiff(INT8) ≈ 4.620  (vs 1.81 GPU, car risk(INT8_NPU)=0.15 vs 0.35)

Références
──────────
  Williams et al. (2009) "Roofline: An Insightful Visual Performance Model"
      SC'09. https://doi.org/10.1145/1654059.1654124
  Dettmers et al. (2022) "LLM.int8(): 8-bit Matrix Multiplication for
      Transformers at Scale." NeurIPS'22. arXiv:2208.07339
  Frantar et al. (2022) "GPTQ: Accurate Post-Training Quantization for
      Generative Pre-trained Transformers." ICLR'23. arXiv:2210.17323
  AMD (2023) "Versal XDNA AI Engine Architecture." Xilinx/AMD whitepaper.
  AMD (2024) "Ryzen AI 300 Series Product Brief." amd.com/en/products/ryzen-ai
  Hauswald et al. (2015) "DjiNN and Tonic: DNN as a Service and Its
      Implications for Future Warehouse Scale Computers." ISCA'15.
  Lin et al. (2024) "AWQ: Activation-aware Weight Quantization for On-Device
      LLM Compression and Acceleration." MLSys'24. arXiv:2306.00978
"""

import json
import math
import re as _re
from collections import Counter
from typing import Dict, List, Optional

from ortools.linear_solver import pywraplp

# ═══════════════════════════════════════════════════════════════════════════════
# 1. CONSTANTES XDNA2
# ═══════════════════════════════════════════════════════════════════════════════

# Sources : AMD Ryzen AI 300 Series Product Brief (2024)
#           AMD XDNA2 / AIE2P Architecture whitepaper (2023)
XDNA2_TOPS_INT8       : float = 50.0    # TOPS peak INT8 (Ryzen AI 300, Strix Point)
XDNA2_TOPS_BF16       : float = 12.5   # TOPS BF16 (AIE2P vecteur 32-bit)
XDNA2_BW_THEORETICAL  : float = 60.0   # GB/s LPDDR5X théorique
# SW efficiency factor : 50 % du débit théorique en workloads réels pour les
# stacks NPU consumer actuels (Hauswald 2015 ; ARM NN 2022 benchmark notes).
XDNA2_BW_EFF_FACTOR   : float = 0.50
XDNA2_BW_EFF          : float = XDNA2_BW_THEORETICAL * XDNA2_BW_EFF_FACTOR  # 30 GB/s

# Intensité opérationnelle seuil (Roofline, Williams 2009)
# OI* = P_peak / BW_eff = 50T / 30G ≈ 1 667 FLOPs/byte
# Decode LLM S=1 : OI ≈ 1–2 → memory-bound, G(q) analytiquement valide
XDNA2_OI_THRESHOLD : float = (XDNA2_TOPS_INT8 * 1e12) / (XDNA2_BW_EFF * 1e9)

# Tile SRAM (AIE2 core local memory) : 32 KB/tile
# Source : AMD XDNA2 Architecture Reference, section "Local Memory"
XDNA2_TILE_SRAM_KB : int = 32


# ═══════════════════════════════════════════════════════════════════════════════
# 2. DTYPE_PROPS XDNA2
# ═══════════════════════════════════════════════════════════════════════════════
#
# bpe (bytes per element) pour décodage S=1, mémoire-bound :
#
#   BF16  : 2.0000 bpe  (baseline — référence Roofline NPU)
#   INT8  : 1.0000 bpe  (exact, format natif AIE2, pas d'overhead groupwise)
#           G(INT8)  = log(2.0/1.0)  = 0.6931
#           risk     = 0.15  ← RÉDUIT vs GPU (0.35) : accumulation HW 8→32 bit
#                               évite saturation (Dettmers 2022 ; AMD AIE2 ISA 2023)
#   INT4  : 0.5250 bpe  (4b weights + ~5 % overhead group scales, group_size=32)
#           group overhead : 2B fp16 scale/32 elts → 2/32 = 0.0625 bpe extra
#           bpe total ≈ 0.500 + 0.0625/2 ≈ 0.525  (Lin et al. AWQ 2024)
#           G(INT4)  = log(2.0/0.525) ≈ 1.3403
#           risk     = 0.80
#
# Seuils d'indifférence λ (lw=1.0) :
#   λ_indiff(INT4) = G(INT4) / (lw · risk(INT4)) = 1.3403 / 0.80 ≈ 1.675
#   λ_indiff(INT8) = G(INT8) / (lw · risk(INT8)) = 0.6931 / 0.15 ≈ 4.621
#
# Règle opérationnelle XDNA2 :
#   λ < 1.675         → INT4 ET INT8 > BF16
#   1.675 ≤ λ < 4.621 → INT8 seul (INT4 rejeté)
#   λ ≥ 4.621         → BF16 conservateur

_BPE_BF16 : float = 2.0

DTYPE_PROPS : Dict[str, Dict] = {
    'BF16': {
        'bpe'     : 2.0000,
        'G'       : 0.0,
        'risk'    : 0.00,
        'onnx'    : 'float16',   # ONNX RT: BF16 → FP16 si BF16 EP absent
        'oga_type': None,        # natif (pas de quantif)
    },
    'INT8': {
        'bpe'     : 1.0000,
        'G'       : math.log(_BPE_BF16 / 1.0000),  # 0.6931
        'risk'    : 0.15,
        'onnx'    : 'uint8',
        'oga_type': 'u8',
    },
    'INT4': {
        'bpe'     : 0.5250,
        'G'       : math.log(_BPE_BF16 / 0.5250),  # 1.3403
        'risk'    : 0.80,
        'onnx'    : 'uint4',
        'oga_type': 'u4',
    },
}
DTYPES = list(DTYPE_PROPS.keys())  # ['BF16', 'INT8', 'INT4']

# Seuils analytiques précompilés
LAM_INDIFF_INT4 = DTYPE_PROPS['INT4']['G'] / DTYPE_PROPS['INT4']['risk']   # ≈ 1.675
LAM_INDIFF_INT8 = DTYPE_PROPS['INT8']['G'] / DTYPE_PROPS['INT8']['risk']   # ≈ 4.621


# ═══════════════════════════════════════════════════════════════════════════════
# 3. CLASSIFICATION DES COUCHES
# ═══════════════════════════════════════════════════════════════════════════════
#
# Spécificités NPU XDNA2 :
#   - KV projections (k_proj, v_proj) : catégorie dédiée, lw=2.0
#     Justification : erreurs quantif sur K/V propagées sur toute la séquence
#     via softmax → sensibilité amplifiée (Lin et al. AWQ 2024, section 3.2)
#   - Attention Q/O   : lw=1.60 (erreurs ×softmax, + qu'un FFN)
#   - FFN gate/up/dn  : lw=0.70 (GELU/SiLU lisse, très robuste INT4)
#   - Embed/Head/Norm : lw=99  → forcé BF16 (contrainte dure ILP)

LAYER_WEIGHT : Dict[str, float] = {
    'embed': 99.0,   # forcé BF16
    'head' : 99.0,   # forcé BF16
    'norm' : 99.0,   # forcé BF16 (RMSNorm scale : 1 valeur/dim critique)
    'kv'   :  2.00,  # K/V projections ultra-sensibles (AWQ 2024)
    'attn' :  1.60,  # Q/O projections (erreurs ×softmax)
    'ffn'  :  0.70,  # Gate/Up/Down (GELU robuste, cible INT4)
    'bias' :  0.00,  # négligeable
    'other':  1.00,
}


def classify(name: str) -> str:
    """Classifie un tensor HF par type de couche."""
    n = name.lower()
    if any(p in n for p in ('embed', 'wte', 'wpe', 'tok_embed', 'position_embed')):
        return 'embed'
    if any(p in n for p in ('lm_head', 'head.weight', 'output.weight', 'cls')):
        return 'head'
    if any(p in n for p in ('norm', 'ln_', 'layer_norm', 'rms_norm', 'layernorm')):
        return 'norm'
    # KV séparé des autres projections attention
    if any(p in n for p in ('k_proj', 'v_proj', 'wk', 'wv',
                             '.key.', '.value.', 'k_cache', 'v_cache')):
        return 'kv'
    if any(p in n for p in ('attn', 'attention', 'q_proj', 'o_proj',
                             'c_attn', 'c_proj', 'wq', 'wo', 'query',
                             'self_attention')):
        return 'attn'
    if any(p in n for p in ('mlp', 'ffn', 'fc', 'gate_proj', 'up_proj',
                             'down_proj', 'c_fc', 'w1', 'w2', 'w3', 'dense',
                             'intermediate', 'feed_forward')):
        return 'ffn'
    if 'bias' in n:
        return 'bias'
    return 'other'


# ═══════════════════════════════════════════════════════════════════════════════
# 4. ROOFLINE XDNA2
# ═══════════════════════════════════════════════════════════════════════════════

def roofline_tps(
    model_params_b: float,
    dtype: str,
    bw_gb_s: float = XDNA2_BW_EFF,
) -> float:
    """
    Estime TPS (tokens/s) via Roofline pour decode S=1 (Williams 2009).

    T_decode = model_bytes / BW_eff
    TPS      = 1 / T_decode

    Memory-bound car OI_decode ≈ 1–2 « OI* = 1667 FLOPs/B.
    """
    bpe         = DTYPE_PROPS[dtype]['bpe']
    model_bytes = model_params_b * 1e9 * bpe
    t_s         = model_bytes / (bw_gb_s * 1e9)
    return 1.0 / t_s


def roofline_table(model_params_b: float = 7.0) -> str:
    """Retourne un tableau Roofline formaté."""
    header = (
        f"\n{'─'*62}\n"
        f"  Roofline XDNA2 — {model_params_b}B params\n"
        f"  P_peak(INT8)={XDNA2_TOPS_INT8} TOPS  "
        f"BW_eff={XDNA2_BW_EFF:.0f} GB/s  "
        f"OI*={XDNA2_OI_THRESHOLD:.0f} FLOPs/B\n"
        f"  (Decode S=1 : OI≈1 → memory-bound, G(q) valide)\n"
        f"{'─'*62}\n"
        f"  {'Dtype':<8} {'bpe':>6} {'G(q)':>8} {'risk':>6}  TPS_est\n"
        f"  {'─'*52}"
    )
    rows = []
    for dt in DTYPES:
        tps = roofline_tps(model_params_b, dt)
        rows.append(
            f"  {dt:<8} {DTYPE_PROPS[dt]['bpe']:>6.4f} "
            f"{DTYPE_PROPS[dt]['G']:>8.4f} "
            f"{DTYPE_PROPS[dt]['risk']:>6.2f}  "
            f"{tps:>6.1f} t/s"
        )
    return header + '\n' + '\n'.join(rows) + f"\n{'─'*62}"


# ═══════════════════════════════════════════════════════════════════════════════
# 5. ILP SOLVER
# ═══════════════════════════════════════════════════════════════════════════════

def solve_quantization_plan(
    layers            : List[Dict],
    ram_budget_gb     : float,
    lam               : float = 1.0,
    w_risk            : float = None,    # alias compat d2_production
    overhead_factor   : float = 1.20,    # 20 % réservé KV cache + activations
) -> List[Dict]:
    """
    ILP via OR-Tools SCIP pour NPU XDNA2.

    Objectif :
      max Σ_{i,q} x[i,q] · score(i,q)
      score(i,q) = G(q) - λ · lw(i) · risk(q)

    Contraintes :
      C1 : Σ_q x[i,q] = 1           ∀ i   (un seul dtype par couche)
      C2 : Σ_{i,q} x[i,q]·bpe(q)·sz(i) ≤ budget_eff_bytes
      C3 : x[i,'BF16'] = 1          si lw(i) ≥ 99  (hard — Embed/Head/Norm)

    Budget ajusté :
      budget_eff = ram_budget_gb / overhead_factor
      Réserve explicite pour KV cache et activations (~20 %).

    Seuils λ (lw=1.0) :
      λ < 1.675         → INT4 et INT8 > BF16
      1.675 ≤ λ < 4.621 → INT8 seul
      λ ≥ 4.621         → BF16
    """
    if w_risk is not None:
        lam = w_risk

    budget_bytes = (ram_budget_gb / overhead_factor) * 1e9
    n = len(layers)
    P = len(DTYPES)

    lw    = [LAYER_WEIGHT[classify(l['name'])] for l in layers]
    score = []
    vram  = []

    for i, layer in enumerate(layers):
        sz = layer['shape'][0] * layer['shape'][1]
        for q in DTYPES:
            s = DTYPE_PROPS[q]['G'] - lam * lw[i] * DTYPE_PROPS[q]['risk']
            score.append(s)
            vram.append(sz * DTYPE_PROPS[q]['bpe'])

    # ── OR-Tools SCIP ────────────────────────────────────────────────────
    solver = pywraplp.Solver.CreateSolver('SCIP')

    x = [[solver.BoolVar(f'x_{i}_{q}') for q in range(P)] for i in range(n)]

    # C1 : unicité dtype
    for i in range(n):
        solver.Add(sum(x[i]) == 1)

    # C2 : RAM ≤ budget_eff
    solver.Add(
        sum(x[i][q] * vram[i * P + q]
            for i in range(n) for q in range(P)) <= budget_bytes
    )

    # C3 : hard BF16 (embed / head / norm)
    for i in range(n):
        if lw[i] >= 99.0:
            solver.Add(x[i][0] == 1)   # index 0 = BF16

    # Objectif maximisation
    obj = solver.Objective()
    for i in range(n):
        for q in range(P):
            obj.SetCoefficient(x[i][q], score[i * P + q])
    obj.SetMaximization()

    status = solver.Solve()
    ok = status in (pywraplp.Solver.OPTIMAL, pywraplp.Solver.FEASIBLE)

    plan = []
    for i, layer in enumerate(layers):
        if ok:
            qi = max(range(P), key=lambda q: x[i][q].solution_value())
        else:
            qi = max(range(P), key=lambda q: score[i * P + q])
        dtype = DTYPES[qi]
        sz    = layer['shape'][0] * layer['shape'][1]
        plan.append({
            **layer,
            'dtype'       : dtype,
            'layer_type'  : classify(layer['name']),
            'layer_weight': lw[i],
            'score'       : round(score[i * P + qi], 4),
            'G'           : round(DTYPE_PROPS[dtype]['G'], 4),
            'risk'        : round(DTYPE_PROPS[dtype]['risk'], 3),
            'ram_gb'      : round(sz * DTYPE_PROPS[dtype]['bpe'] / 1e9, 5),
        })
    return plan


# ═══════════════════════════════════════════════════════════════════════════════
# 6. EXPORT ONNX / OGA / OLIVE-AI
# ═══════════════════════════════════════════════════════════════════════════════

def export_onnx_config(
    plan : List[Dict],
    path : str = 'xdna2_quant_config.json',
    lam  : float = 1.0,
) -> str:
    """
    Exporte le plan en JSON compatible ONNX Runtime GenAI (OGA) et Olive-AI.

    Sorties :
      <path>                 — config principale (oga_cmd inclus)
      <path>_olive.json      — config Olive-AI pour quantification mixte

    Usage OGA :
      python -m onnxruntime_genai.tools.model_builder \\
          --model <hf_id_or_path> --precision u4 \\
          --execution_provider qnn --output <out_dir>

    Usage Olive-AI :
      olive run --config xdna2_quant_config_olive.json
    """
    counts      = Counter(e['dtype'] for e in plan)
    total_ram   = sum(e['ram_gb'] for e in plan)
    dominant    = max(DTYPES, key=lambda d: counts.get(d, 0))

    layer_overrides = []
    for e in plan:
        dt = e['dtype']
        layer_overrides.append({
            'name'      : e['name'],
            'layer_type': e['layer_type'],
            'dtype'     : dt,
            'onnx_dtype': DTYPE_PROPS[dt]['onnx'],
            'oga_type'  : DTYPE_PROPS[dt]['oga_type'],
            'score'     : e['score'],
            'risk'      : e['risk'],
        })

    oga_cmd = _build_oga_cmd(dominant, total_ram)

    config = {
        'format'        : 'd2_npu_xdna2_v1',
        'target'        : 'AMD_XDNA2_Ryzen_AI_300',
        'lambda'        : lam,
        'default_quant' : DTYPE_PROPS[dominant]['oga_type'] or 'bf16',
        'total_ram_gb'  : round(total_ram, 3),
        'distribution'  : {dt: counts.get(dt, 0) for dt in DTYPES},
        'xdna2_hw'      : {
            'tops_int8'       : XDNA2_TOPS_INT8,
            'bw_theoretical_gb': XDNA2_BW_THEORETICAL,
            'bw_eff_gb'       : XDNA2_BW_EFF,
            'oi_threshold'    : round(XDNA2_OI_THRESHOLD, 1),
            'tile_sram_kb'    : XDNA2_TILE_SRAM_KB,
        },
        'lam_thresholds': {
            'int4_vs_bf16': round(LAM_INDIFF_INT4, 4),
            'int8_vs_bf16': round(LAM_INDIFF_INT8, 4),
        },
        'oga_cmd'       : oga_cmd,
        'layers'        : layer_overrides,
    }

    with open(path, 'w') as f:
        json.dump(config, f, indent=2)

    # Config Olive-AI compagnon
    olive_path = path.replace('.json', '_olive.json')
    _export_olive_config(config, dominant, olive_path)

    return oga_cmd


def _build_oga_cmd(dominant: str, total_ram_gb: float) -> str:
    oga_type = DTYPE_PROPS[dominant]['oga_type'] or 'bf16'
    return '\n'.join([
        "# ── ONNX Runtime GenAI (OGA) — AMD Ryzen AI ─────────────────",
        "# CPU EP (baseline) :",
        "python -m onnxruntime_genai.tools.model_builder \\",
        "    --model <hf_model_id_or_path> \\",
        f"   --precision {oga_type} \\",
        "    --execution_provider cpu \\",
        "    --output <output_dir>",
        "",
        "# QNN EP (Ryzen AI NPU) :",
        "python -m onnxruntime_genai.tools.model_builder \\",
        "    --model <hf_model_id_or_path> \\",
        f"   --precision {oga_type} \\",
        "    --execution_provider qnn \\",
        "    --qnn_context_binary_path <qnn_context.bin> \\",
        "    --output <output_dir>",
        "",
        f"# RAM estimée : {total_ram_gb:.2f} GB (overhead KV non inclus)",
    ])


def _export_olive_config(config: Dict, dominant: str, path: str) -> None:
    """Génère un workflow Olive-AI pour quantification mixte sur Ryzen AI."""
    oga_type = DTYPE_PROPS[dominant]['oga_type'] or 'bf16'
    is_int4  = dominant == 'INT4'
    olive = {
        "description" : "D2 XDNA2 Mixed-Precision Quantization (Olive-AI)",
        "model"       : {"type": "HfModel"},
        "data_config" : {"name": "default"},
        "passes"      : {
            "matmul_nbits" : {
                "type"       : "MatMul4BitsQuantization" if is_int4
                               else "OnnxDynamicQuantization",
                "bits"       : 4 if is_int4 else 8,
                "group_size" : 32,
                "accuracy_level" : 4,
            },
            "ryzenai_quant" : {
                "type"      : "RyzenAIQuantization",
                "quant_mode": oga_type,
                "vai_q_onnx_config": {
                    "quant_format"     : "QDQ",
                    "activation_type"  : "uint8",
                    "weight_type"      : "uint4" if is_int4 else "uint8",
                    "per_channel"      : False,
                    "reduce_range"     : False,
                },
            },
        },
        "engine" : {
            "provider"    : "ryzenai",
            "target"      : "amd_xdna2",
            "cache_dir"   : ".olive_cache",
            "output_dir"  : "olive_output",
        },
        "_d2_meta" : {
            "lambda"    : config.get('lambda', 1.0),
            "ram_gb"    : config['total_ram_gb'],
            "bw_eff_gb" : config['xdna2_hw']['bw_eff_gb'],
            "generator" : "d2_npu_xdna2.py",
        },
    }
    with open(path, 'w') as f:
        json.dump(olive, f, indent=2)


# ═══════════════════════════════════════════════════════════════════════════════
# 7. SUMMARY
# ═══════════════════════════════════════════════════════════════════════════════

def summarize(
    plan             : List[Dict],
    ram_budget_gb    : float,
    lam              : float = 1.0,
    w_risk           : float = None,
    model_params_b   : float = None,
    overhead_factor  : float = 1.20,
) -> str:
    if w_risk is not None:
        lam = w_risk

    counts  = Counter(e['dtype'] for e in plan)
    by_type = Counter(f"{e['layer_type']}:{e['dtype']}" for e in plan)
    ram_gb  = sum(e['ram_gb'] for e in plan)
    avg_G   = sum(DTYPE_PROPS[e['dtype']]['G'] for e in plan) / len(plan)

    regime = (
        'INT4-dominant'      if lam < LAM_INDIFF_INT4 else
        'INT8-dominant'      if lam < LAM_INDIFF_INT8 else
        'BF16-conservateur'
    )

    by_type_str = '  '.join(
        f"{lt}→{dt}:{c}"
        for lt_dt, c in sorted(by_type.items())
        for lt, dt in [lt_dt.split(':')]
    )

    tps_lines = []
    if model_params_b:
        for dt in DTYPES:
            tps = roofline_tps(model_params_b, dt)
            tps_lines.append(f"  TPS_roofline {dt:<5}: {tps:>6.1f} t/s")

    budget_eff = ram_budget_gb / overhead_factor
    lines = [
        f"\n{'═'*58}",
        f"  D2 NPU XDNA2 — Mixed-Precision Plan (Ryzen AI 300)",
        f"{'═'*58}",
        f"  Layers       : {len(plan)}   λ={lam:.3f}  [{regime}]",
        f"  Budget RAM   : {ram_budget_gb:.1f} GB  (eff={budget_eff:.1f} GB, -20% KV/act)",
        f"  RAM utilisé  : {ram_gb:.3f} GB",
        f"  BF16         : {counts.get('BF16', 0)}",
        f"  INT8         : {counts.get('INT8', 0)}   (natif AIE2, risk=0.15)",
        f"  INT4         : {counts.get('INT4', 0)}   (groupwise g=32, bpe=0.525)",
        f"  Avg G(q)     : {avg_G:.4f}  (log-TPS gain vs BF16)",
        f"  λ_indiff     : INT4~{LAM_INDIFF_INT4:.3f}  INT8~{LAM_INDIFF_INT8:.3f}",
        f"  BW_eff       : {XDNA2_BW_EFF:.0f} GB/s   OI*={XDNA2_OI_THRESHOLD:.0f} FLOPs/B",
        f"  By type      : {by_type_str}",
    ] + tps_lines + [f"{'═'*58}"]

    return '\n'.join(lines)


# ═══════════════════════════════════════════════════════════════════════════════
# 8. DEMO / MAIN
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == '__main__':
    import sys

    try:
        from tabulate import tabulate
        TAB = True
    except ImportError:
        TAB = False

    CACHE_DIR = '/tmp/hf_cache'

    # ── Loaders ─────────────────────────────────────────────────────────────

    def _load_hf(model_id: str) -> List[Dict]:
        from huggingface_hub import hf_hub_download, list_repo_files
        import safetensors.torch as st
        files = [f for f in list_repo_files(model_id)
                 if f.endswith('.safetensors') and 'onnx' not in f]
        if not files:
            raise FileNotFoundError(f"Pas de .safetensors pour {model_id}")
        local = hf_hub_download(model_id, files[0], cache_dir=CACHE_DIR)
        tensors = st.load_file(local)
        layers  = []
        for name, tensor in tensors.items():
            if tensor.ndim < 2: continue
            t = tensor.float().numpy()
            if t.ndim > 2: t = t.reshape(t.shape[0], -1)
            m, n = t.shape
            if m < 8 or n < 8: continue
            layers.append({'name': name, 'shape': [m, n]})
        return layers

    def _synthetic_llama(n_layers: int = 32, d_model: int = 4096) -> List[Dict]:
        """Modèle synthétique LLaMA-7B-like (pour tests hors ligne)."""
        ls = [{'name': 'model.embed_tokens.weight', 'shape': [32000, d_model]}]
        for i in range(n_layers):
            p = f'model.layers.{i}'
            ls += [
                {'name': f'{p}.self_attn.q_proj.weight',         'shape': [d_model,       d_model]},
                {'name': f'{p}.self_attn.k_proj.weight',         'shape': [d_model // 4,  d_model]},
                {'name': f'{p}.self_attn.v_proj.weight',         'shape': [d_model // 4,  d_model]},
                {'name': f'{p}.self_attn.o_proj.weight',         'shape': [d_model,       d_model]},
                {'name': f'{p}.mlp.gate_proj.weight',            'shape': [d_model * 4,   d_model]},
                {'name': f'{p}.mlp.up_proj.weight',              'shape': [d_model * 4,   d_model]},
                {'name': f'{p}.mlp.down_proj.weight',            'shape': [d_model,       d_model * 4]},
                {'name': f'{p}.input_layernorm.weight',          'shape': [d_model, 1]},
                {'name': f'{p}.post_attention_layernorm.weight', 'shape': [d_model, 1]},
            ]
        ls.append({'name': 'lm_head.weight', 'shape': [32000, d_model]})
        return ls

    # ── Charger ──────────────────────────────────────────────────────────────
    model_id = sys.argv[1] if len(sys.argv) > 1 else 'gpt2'
    print(f"\n  Modèle : {model_id}")
    try:
        layers = _load_hf(model_id)
        print(f"  Chargé depuis HuggingFace : {len(layers)} couches")
    except Exception as e:
        print(f"  [HF indisponible : {e}]")
        print(f"  → Synthétique LLaMA-7B-like (32 layers, d=4096)")
        layers = _synthetic_llama()

    n_params = sum(l['shape'][0] * l['shape'][1] for l in layers) / 1e9
    print(f"  Paramètres : ~{n_params:.2f}B")

    # ── Roofline ─────────────────────────────────────────────────────────────
    print(roofline_table(model_params_b=n_params))

    # ── Divergence aux seuils λ XDNA2 ───────────────────────────────────────
    print(f"\n{'#'*58}")
    print(f"  TEST : divergence λ (seuils XDNA2 — INT8 natif NPU)")
    print(f"  G(INT4)={DTYPE_PROPS['INT4']['G']:.4f}  G(INT8)={DTYPE_PROPS['INT8']['G']:.4f}")
    print(f"  λ_indiff : INT4~{LAM_INDIFF_INT4:.3f}  INT8~{LAM_INDIFF_INT8:.3f}")
    print(f"{'#'*58}")

    test_lams = [0.5, LAM_INDIFF_INT4, LAM_INDIFF_INT8, 6.0]
    results   = {}
    RAM_BUDGET = 16.0  # GB

    for lv in test_lams:
        plan = solve_quantization_plan(layers, ram_budget_gb=RAM_BUDGET, lam=lv)
        c    = Counter(e['dtype'] for e in plan)
        rg   = sum(e['ram_gb'] for e in plan)
        results[lv] = (c, rg, plan)

    rows = [
        [f"lam={lv:.3f}", c.get('BF16', 0), c.get('INT8', 0), c.get('INT4', 0),
         f"{v:.3f}"]
        for lv, (c, v, _) in results.items()
    ]
    if TAB:
        print(tabulate(rows, headers=['Lambda', 'BF16', 'INT8', 'INT4', 'RAM_GB'],
                       tablefmt='psql'))
    else:
        print(f"  {'Lambda':<14} BF16  INT8  INT4  RAM_GB")
        for r in rows:
            print(f"  {r[0]:<14} {r[1]:>4}  {r[2]:>4}  {r[3]:>4}  {r[4]:>7}")

    # ── Summary & Export ─────────────────────────────────────────────────────
    plan_mid = results[LAM_INDIFF_INT4][2]
    print(summarize(plan_mid, RAM_BUDGET, lam=LAM_INDIFF_INT4,
                    model_params_b=n_params))

    out_path = f'{CACHE_DIR}/xdna2_quant_config.json'
    cmd = export_onnx_config(plan_mid, path=out_path, lam=LAM_INDIFF_INT4)
    print(f"\n  === OGA Command ===")
    for line in cmd.split('\n')[:8]:
        print(f"  {line}")
    print(f"\n  → {out_path}")
    print(f"  → {out_path.replace('.json', '_olive.json')}")
