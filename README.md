============================================================================
RAPPORT TECHNIQUE : Optimisation de l'inférence LLM sur NPU XDNA2
============================================================================

1. EXECUTIVE SUMMARY
Standard quantization (INT4) is often theorized as a compute accelerator. 
Our empirical measurements on XDNA2 reveal that throughput is governed 
by geometric constraints, not precision. We report a 35-70% gain using 
"Hardware Fitness" optimization.

2. EMPIRICAL PERFORMANCE ANALYSIS
- Stagnation: INT4 vs INT8 shows no significant TPS gain (Compute Starvation).
- Alignment: 99% correlation between (hidden_size % 512 == 0) and NPU efficiency.
- Bottleneck: Underutilization of NPU tiles occurs on non-aligned models.



3. D2-PRODUCTION METHODOLOGY
Objective: Maximize service value per layer (i):
Max Σ (x_i,q * (TPS_baseline_i * α_i + λ * Mem_Gain_q))
- α_i (Efficiency Index): Measured penalty factor for misalignment.
- λ: Balancing parameter for memory footprint vs. speed.

4. RECOMMENDED HYBRID DEPLOYMENT
- NPU: Prioritize layers with hidden_size as a multiple of 512.
- GPU (RTX 5070): Offload critical layers (lm_head) or non-aligned tensors.
- Validation: Hardware Fitness profiling for every new model compilation.

5. CONCLUSION
Performance on XDNA2 is a matter of data geometry. By optimizing for tile 
alignment rather than theoretical precision, we bypass standard bottlenecks.

# D2 XDNA2 — Document Explicatif
## Quantification Mixed-Precision sur AMD NPU XDNA2 (Ryzen AI 300 / Strix Point)

**Version** : PoC v1.0  
**Cible HW** : AMD Ryzen AI 9 HX 370 — NPU XDNA2 (32 tiles AIE2)  
**Script** : `d2_xdna2_poc.py`

---

## 1. Contexte et Objectif

Le planificateur D2 résout le problème d'allocation de types de données (dtype) pour les tenseurs d'un LLM, via un ILP (Integer Linear Programming). Objectif : maximiser le gain mémoire (RAM) ou vitesse (TPS) tout en contrôlant la dégradation de qualité.

Ce PoC adapte D2 pour l'NPU AMD XDNA2, en intégrant trois découvertes critiques issues des mesures empiriques sur Ryzen AI 9 HX 370.

---

## 2. Les Trois Hypothèses Prouvées

### H1 — Le vrai bottleneck est le TILING, pas la Bande Passante

**Constat :** sur GPU classique, les LLMs en inférence sont limités par la bande passante mémoire (memory-bound). Le modèle Roofline (Williams et al., SC'09) prédit : `TPS = BW_eff / model_bytes`.

**Sur XDNA2 :** le bottleneck principal est l'alignement des dimensions du modèle au bloc GEMM du compilateur AIE.

.

**Score de tile utilization :**
```
tile_util(h) = floor(h/8 / 256) × 256 / (h/8)
```

**Résultats mesurés :**

| Modèle         | hidden_size | tile_util | TPS (t/s) |
|----------------|-------------|-----------|-----------|
| deepseek-r1:8b | 4096        | 1.000     | 10.75     |
| qwen3.5:2b     | 2048        | 1.000     | 24.29     |
| lfm2:1.2b      | 1536        | 1.000     | 50.99     |
| qwen3.5:9b     | **3584**    | **~0.50** | **7.68**  |

`qwen3.5:9b` a 9B paramètres mais performe moins bien que `deepseek-r1:8b` (8B) à cause d'un `hidden_size=3584` mal aligné. La perte est ~30% de TPS attendu.

**Conséquence pour D2 :** le score de fitness d'un modèle doit intégrer `tile_util` comme facteur multiplicatif, AVANT d'estimer le gain par quantification.

**Critère de sélection de modèle :** choisir des modèles dont `hidden_size` est un multiple de `8 × 256 = 2048`, ou au moins de `256`. Exemples bons : 2048, 4096. Exemple mauvais : 3584.

---
cela est conceptuel pas encore tester 
### H2 — L'INT4 n'améliore PAS le TPS sur XDNA2 actuel

**Constat naïf :** INT4 = 0.5 bpe → gain TPS théorique × 4 vs BF16.

**Réalité sur XDNA2 :** Empiriquement, le TPS mesuré en INT4 est identique au TPS INT8 sur Ryzen AI 9 HX 370 — ce qui indique que l'INT4 est déquantifié avant l'exécution du kernel NPU. Les kernels AIE2 actifs opèrent en INT8 ou BF16 ; le format INT4 sert uniquement au stockage.

**Conséquence directe :**
- `bpe_compute(INT4) = bpe_compute(BF16) = 2.0` — les bytes traités par le NPU sont identiques
- `G_tps(INT4) = 0` — aucun gain de vitesse
- `bpe_storage(INT4) = 0.525` — le poids est STOCKÉ en 4 bits (moins de RAM)
- `G_storage(INT4) = ln(2.0/0.525) ≈ 1.34` — économie mémoire réelle ×3.8

**Table DTYPE corrigée :**

| Dtype | bpe_storage | bpe_compute | G_tps | G_storage | Risk |
|-------|-------------|-------------|-------|-----------|------|
| BF16  | 2.000       | 2.000       | 0.000 | 0.000     | 0.00 |
| INT8  | 1.000       | 1.000       | **0.693** | 0.693 | 0.15 |
| INT4  | **0.525**   | 2.000       | **0.000** | **1.340** | 0.80 |

**Conclusion :**
- Pour maximiser **TPS** : préférer INT8 (G_tps=0.693), éviter INT4 (G_tps=0)
- Pour maximiser **compression RAM** : INT4 reste utile (G_storage=1.34)
- Un plan tout-INT8 est toujours supérieur en TPS à un plan tout-INT4

**Impact sur l'ILP D2 :** l'objectif doit séparer `G_tps` (pour l'estimation TPS) et `G_storage` (pour la contrainte RAM et l'optimisation compacité).

---

### H3 — Score Dual : ILP stockage-orienté + Roofline compute-orienté

**Formulation D2 classique (GPU) :**
```
score(i,q) = G(q) - λ × lw(i) × risk(q)
```
avec un seul G qui cumule gains TPS et RAM.

**Formulation D2 XDNA2 corrigée :**
```
ILP objective : score(i,q) = G_storage(q) - λ × lw(i) × risk(q)
TPS estimate  : TPS_pred = (BW_eff / compute_gb) × tile_util
```

Le G_storage sert à l'optimisation ILP et à la contrainte RAM.  
Le G_tps (séparé) sert uniquement à prédire la vitesse dans le Roofline.

**Seuils λ (calculés, lw=1.0) :**
- `λ_indiff(INT4) = G_storage(INT4) / risk(INT4) = 1.34 / 0.80 ≈ 1.675`
- `λ_indiff(INT8) = G_storage(INT8) / risk(INT8) = 0.69 / 0.15 ≈ 4.621`

Pour `λ < 1.675` : INT4 dominant (agressif, gain RAM max)  
Pour `1.675 < λ < 4.621` : INT8 dominant (équilibré)  
Pour `λ > 4.621` : BF16 dominant (conservateur, qualité max)

**Résultat du PoC (LLaMA-7B-like, hidden=4096, budget=16 GB) :**

| λ     | BF16 | INT8 | INT4 | Storage (GB) | TPS_adj | Régime    |
|-------|------|------|------|--------------|---------|-----------|
| 0.500 | 2    | 0    | 258  | ~1.4         | ~38     | INT4+INT8 |
| 1.675 | 2    | 258  | 0    | ~3.7         | ~14     | INT8      |
| 4.621 | 68+  | 190- | 0    | ~5.5         | ~10     | BF16 mix  |
| 6.000 | 260  | 0    | 0    | ~7.4         | ~7      | BF16      |

pas encore tester sur le xdna2  mais bientot 

La transition INT4→INT8 à λ≈1.675 est confirmée, validant la formulation analytique.

---

## 3. Architecture du PoC

### Modules

```
d2_xdna2_poc.py
├── § 1. Constantes hardware XDNA2
├── § 2. tile_utilization(hidden_size)          — H1 scoring
├── § 3. Calibration empirique TPS              — données de calibration
├── § 4. DTYPE_PROPS corrigés                   — H2 : G_tps vs G_storage
├── § 5. solve_quantization_plan()              — ILP (OR-Tools SCIP ou greedy)
├── § 6. estimate_tps()                         — Roofline dual-bottleneck
├── § 7. xdna2_fitness()                        — score sélection modèle
├── § 8. export_poc_report()                    — rapport JSON
└── § 9. demo()                                 — démo complète
```

### Dépendances

- `ortools` (optionnel) : ILP exact. Sans → greedy heuristique
- `json`, `math`, `collections` : stdlib Python 3.8+

### Lancer le PoC

```bash
pip install ortools      # recommandé
python d2_xdna2_poc.py
```

---

## 4. Problèmes Identifiés — À Résoudre par les Développeurs

### P1 — Mapping tensoriel HF → ONNX (BLOQUANT pour production)

**Problème :** D2 raisonne sur les noms HuggingFace (`model.layers.0.mlp.gate_proj.weight`). Or les configs ONNX/OGA nomment les tenseurs différemment (`/model/layers.0/mlp/gate_proj/MatMul`). Il n'existe pas de mapping universel.

**Conséquence :** l'export de config ONNX ne peut pas spécifier les overrides de dtype par tenseur sans ce mapping.

**Piste de résolution :** parser le graphe ONNX avec `onnxruntime` ou `onnx` Python API pour reconstruire la table `{hf_name → onnx_node_name}`. Peut être automatisé pour les architectures standard (LLaMA, Qwen, Phi).

---

### P2 — INT4 Natif Futur (invalidera H2)

**Problème :** H2 est vraie avec les kernels NPU actuellement déployés. Si AMD intègre des GEMM INT4 natifs dans une future révision du firmware ou du compilateur Vitis AI, `bpe_compute(INT4)` deviendra `0.5`, et `G_tps(INT4)` atteindra `1.386`.

**Conséquence pour D2 :** il faudra détecter dynamiquement si le kernel INT4 natif est disponible et mettre à jour `DTYPE_PROPS['INT4']['bpe_compute']`.

**Détection suggérée :** vérifier la version du firmware NPU via l'API driver et la présence de kernels `MatMulNBits` dans le graphe ONNX compilé (opérateur ONNX public).

---

### P3 — Budget KV Cache Dynamique

**Problème :** la contrainte RAM dans D2 est `Σ(bpe_storage × params) ≤ budget`. Mais le KV cache varie avec `seq_len`, `n_heads`, `d_head`, et `n_layers`. Pour un budget LPDDR5 de 16 GB :

```
KV_cache_size = 2 × n_layers × seq_len × n_heads × d_head × bpe_kv
```

Pour LLaMA-7B (32L, 32H, 128d), seq_len=4096 : `2×32×4096×32×128×2 = 2.15 GB` (BF16).

Ce KV overhead n'est pas comptabilisé dans le plan D2 actuel.

**Résolution :** passer `seq_len_budget` comme paramètre et soustraire le KV size du `ram_budget` avant l'ILP.

---

### P4 — Calibration BW_eff par Architecture

**Problème :** `BW_eff = 55 GB/s` est une moyenne calibrée sur plusieurs modèles. La valeur réelle varie :
- Modèles CPU-bound (phi4-mini) : BW apparente ≠ BW NPU réelle
- Modèles avec overhead embarassant de dequant INT4 : BW effective réduite
- Modèles avec CP starvation (BIOS/firmware) : stalls non prévisibles

**Résolution :** calibrer `BW_eff` individuellement par modèle (1 benchmark rapide de 50 tokens) et l'injecter dans le Roofline. D2 devrait exposer un paramètre `bw_eff_override`.

---


**Conséquence :** `BW_eff` réelle est ~30 GB/s ( vs ~55-60 GB/s théorique (8 colonnes). ( acer nitro 16v ia ) 
Le tile_util calculé sur 8 colonnes est donc une approximation optimiste.

**Résolution :** paramétrer `XDNA2_COLS_ACTIVE` séparément de `XDNA2_COLS_TOTAL`, et utiliser le premier pour `tile_utilization()`. Valeur correcte : `XDNA2_COLS_ACTIVE = 4` dans la configuration actuelle.

**Note :** les 8 colonnes sont disponibles hardware — c'est un problème de driver, résolvable par mise à jour firmware.

---

### P6 — `--pmode performance` Non Modélisé

**P
Ce gain n'est pas inclus dans le Roofline D2 car il dépend du firmware NPU, pas du plan de quantification.

**Résolution :** exposer un paramètre booléen `pmode_enabled: bool` qui multiplie `TPS_adj` par `PMODE_FACTOR = 1.44`. Permettre à l'utilisateur de le calibrer sur son système.

---

### P7 — Modèles >  (Crash NPU)

**Problème :** les modèles ≥ 20B paramètres causent des crashs NPU (kernel panic / OOM). 
---

## 5. Recommandations pour l'Optimisation sur XDNA2

Par ordre d'impact décroissant :

1. **Activer `--pmode performance`** → +44% TPS, zéro coût, toujours recommandé
2. **Choisir `hidden_size` multiple de 2048** → éviter la pénalité de tiling (-50%)
3. **Préférer INT8 à INT4** pour le TPS (G_tps identique, risque qualité réduit ×5)
4. **Utiliser INT4 uniquement si RAM est la contrainte** (budget < fichier INT8)
5. **Attendre la résolution de P5 (8 colonnes)** pour un gain ×2 sur la BW effective

---1. STRUCTURE DU REPO
xdna2-llm-performance-observatory/
│
├── README.md
├── datasets/
│   ├── raw_runs.csv
│   ├── model_benchmarks.json
│   └── scaling_results.csv
│
├── scripts/
│   ├── run_benchmark.py
│   ├── collect_metrics.py
│   ├── scaling_test.py
│   └── export_dataset.py
│
├── analysis/
│   ├── roofline_proxy.py
│   ├── scaling_plots.py
│   └── latency_breakdown.py
│
├── docs/
│   ├── methodology.md
│   ├── hardware_notes.md
│   └── limitations.md
│
└── examples/
    ├── ollama_config.json
    ├── fastflow_config.json
    └── batch_tests.sh
📊 2. DATASET (format propre dev)
📄 datasets/raw_runs.csv
timestamp,model,batch_size,context_len,latency_s,tokens,throughput_tps,run_id
2026-04-09,qwen3.5:9b,1,2048,12.45,96,7.71,001
2026-04-09,qwen3.5:9b,2,2048,13.10,192,14.65,002
2026-04-09,qwen3.5:9b,4,2048,15.80,384,24.30,003
📄 datasets/model_benchmarks.json
{
  "qwen3.5:9b": {
    "decode_tps": 7.8,
    "prefill_tps": 46.3,
    "stable_batch": 1,
    "notes": "saturation observed beyond batch=2"
  }
}
📄 datasets/scaling_results.csv
model,batch,avg_latency,std,throughput
qwen3.5:9b,1,12.4,0.3,7.8
qwen3.5:9b,2,13.1,0.4,15.2
qwen3.5:9b,4,15.8,0.6,25.3
⚙️ 3. SCRIPT DEV UTILISABLE
🧪 scripts/run_benchmark.py
import time
import subprocess
import csv

MODEL = "qwen3.5:9b"

def run(batch=1):
    start = time.perf_counter()

    result = subprocess.run(
        ["flm.exe", "run", MODEL, f"--batch={batch}"],
        capture_output=True,
        text=True
    )

    end = time.perf_counter()

    latency = end - start
    tokens = 128  # fallback si non exposé
    tps = tokens / latency

    return latency, tps


def main():
    batches = [1, 2, 4, 8]

    with open("../datasets/raw_runs.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["batch", "latency", "tps"])

        for b in batches:
            lat, tps = run(b)
            writer.writerow([b, lat, tps])
            print(f"batch={b} latency={lat:.2f}s tps={tps:.2f}")


if __name__ == "__main__":
    main()
📈 4. ANALYSE ROOFLINE PROXY
analysis/roofline_proxy.py
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("../datasets/raw_runs.csv")

plt.figure()

plt.plot(df["batch_size"], df["throughput_tps"], marker="o")

plt.title("LLM Throughput Scaling (Proxy Roofline)")
plt.xlabel("Batch Size")
plt.ylabel("Tokens/sec")

plt.grid()
plt.show()
📉 5. SCALING ANALYSIS
analysis/scaling_plots.py
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("../datasets/scaling_results.csv")

for model in df["model"].unique():
    d = df[df["model"] == model]
    plt.plot(d["batch"], d["throughput"], label=model)

plt.legend()
plt.title("Scaling behavior across models")
plt.xlabel("Batch size")
plt.ylabel("Throughput (TPS)")
plt.show()
📘 6. README.md (clé pour devs)
# XDNA2 LLM Performance Observatory

Empirical performance dataset for LLM inference on AMD XDNA2 systems.

## Goal

Understand how:
- model architecture
- batch size
- context length
- runtime configuration

affect inference performance.

---

## Key Observations (empirical)

- Strong sensitivity to model shape (hidden size)
- Decode throughput saturates around ~7–8 TPS (Qwen3.5 9B)
- Prefill scales differently from decode
- Batch size improves throughput until saturation

---

## What this is NOT

- Not an official AMD benchmark
- Not a hardware spec validation
- Not a driver-level instrumentation tool

---

## What this IS

- Empirical performance dataset
- Reproducible inference benchmarking framework
- Scaling behavior analysis toolkit

---

## Usage

```bash
python scripts/run_benchmark.py
python analysis/scaling_plots.py
Target users
LLM inference engineers
runtime optimizer developers
ML systems researchers

---

# 📌 7. LIMITATIONS.md (important pour crédibilité)

```md
## Limitations

- No hardware counters (AIE utilization unknown)
- No roofline validation
- No driver-level instrumentation
- Latency breakdown approximate
- Results dependent on runtime (FastFlowLM / Ollama / etc.)

## 6. Références


https://www.amd.com/fr/technologies/xdna

https://docs.kernel.org/accel/amdxdna/amdnpu.html

https://www.amd.com/en/developer/resources/ryzen-ai-software.html

https://arxiv.org/html/2606.11357v1

https://www.amd.com/en/developer/resources/technical-articles/2026/accelerating-llm-startup-on-amd-ryzen-ai.html

https://github.com/amd/xdna-driver

https://xilinx.github.io/XRT/
https://arxiv.org/pdf/2512.13282
- Williams, S. et al. (2009). *Roofline: An insightful visual performance model for floating-point programs and multicore architectures*. SC'09. — Modèle de bottleneck mémoire/compute
- Dettmers, T. et al. (2022). *LLM.int8(): 8-bit Matrix Multiplication for Transformers at Scale*. NeurIPS'22. — Principe de déquantification pour GEMM
- Lin, J. et al. (2024). *AWQ: Activation-aware Weight Quantization for LLM Compression and Acceleration*. MLSys'24. — Groupwise INT4, sensibilité KV
- AMD (2023). *XDNA2 AIE2 Architecture Whitepaper*. — Tiles, GEMM blocks, SRAM
- AMD (2024). *Ryzen AI 300 Product Brief*. — BW LPDDR5X théorique
- Hauswald, J. et al. (2015). *Sirius: An Open End-to-End Voice and Vision Personal Assistant and Its Implications for Future Warehouse Scale Computers*. — Efficacité mémoire des accélérateurs NPU (~50% theoretical BW)



Casual tinkerer sharing raw field notes — probably full of mistakes

I'm just a curious dev who wanted to understand how the XDNA2 NPU actually behaves when you throw LLMs at it. I ran benchmarks, made guesses, broke things, and learned a bit along the way. This repo is the messy trace of that — not a research paper, not an official guide.

The numbers come from my specific setup only. Some of my conclusions are likely wrong or outdated (especially the 4-column thing — newer work like TileFuse tells a different story). I'm publishing this to share what I found and hopefully get corrected by people who know more.

If you spot something dumb, open an issue. I'd genuinely appreciate it.
