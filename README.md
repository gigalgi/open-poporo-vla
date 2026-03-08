# 🧠 open-poporo-vla

**Biologically-grounded Vision-Language-Action architecture mapping four neurobiological mechanisms from primate cortex directly onto state-of-the-art robotics AI frameworks. Independent open-source research. In development.**

[![License](https://img.shields.io/badge/License-Apache%202.0-blue)](LICENSE)
[![Status](https://img.shields.io/badge/Status-In%20Development-orange)]()

---

> *The Quimbaya Poporo (c. 301 BC – 600 AD) is a masterpiece of pre-Columbian goldsmithing that symbolizes wisdom, fertility, and elevated thought through the ritual of "mambeo" — the chewing of coca leaf. This gold vessel, used to store lime, represents cosmic balance, spiritual connection, and ancestral knowledge.*



<div align="center">
  <img src="docs/images/architecture-overview.png" width="780" alt="open-poporo-vla architecture overview"/>
  <br><em>open-poporo-vla — four biologically-grounded pillars unified into a smolVLA backbone for compositional, continual, and physically precise manipulation.</em>
</div>


---

## Overview

**open-poporo-vla** is a modular Vision-Language-Action architecture that directly maps four neurobiological mechanisms — identified in primate prefrontal and motor cortices by Tafazoli et al. (*Nature*, 2026) [1] — onto four state-of-the-art AI and robotics frameworks. The design principle is not biological *inspiration* but biological *grounding*: every architectural decision is traceable to a specific empirical finding in systems neuroscience, with a precise computational analogue derived from current literature.

**Primary test platform: dual-arm [SO-ARM101](https://github.com/TheRobotStudio/SO-ARM100) setup emulating a microfactory assembly environment.** The dual-arm configuration exercises the full compositional and continual learning capabilities of the architecture — bimanual coordination, sequential sub-skill composition, and physical precision — which are exactly the behaviors the four pillars are designed to enable.

The architecture addresses a fundamental limitation in the current generation of robotic VLAs:

- **Large foundation VLAs** (RT-2, π₀, OpenVLA) generalize broadly but fail at compositional new-task construction from known sub-skills, and suffer catastrophic forgetting when fine-tuned on new tasks.
- **Modular skill systems** support continual learning but lack the physical precision and multi-task routing needed for contact-rich manipulation.
- **Neurosymbolic architectures** (cf. [neurosymbolic-vla](https://github.com/yourusername/neurosymbolic-vla)) solve compositional reasoning explicitly through symbolic graphs and planners, but require manual domain engineering.

This work proposes a fully neural solution: a **smolVLA** [3] backbone enriched with four biologically-grounded modules — **V-JEPA 2** for task belief [2], **DGMoE** for active suppression [6], **LoRAC** for shared subspace skill storage [7][8], and **Residual RL / PLD** for cerebellar-style motor correction [5][9] — producing a system that acquires compositional manipulation skills continually without symbolic scaffolding.

---

## Architectural Foundations

### 1. Backbone — smolVLA

The inference backbone is **smolVLA** [3], chosen because it achieves near-frontier manipulation performance within a resource envelope accessible to independent researchers, and because its modular architecture exposes clean injection points for the four biological pillars.

| Property | Value |
|---|---|
| Parameters | ~450 M |
| Visual tokens | 64 (PixelShuffle pooling) |
| Transformer acceleration | L/2 layer skipping |
| Action head | Flow Matching [11] |
| Target inference rate | 50 Hz+ on consumer CPU |
| LIBERO benchmark | 80.3% |

PixelShuffle token compression reduces spatial redundancy in the visual stream while preserving manipulation-relevant features. The Flow Matching action head provides smooth, multi-modal trajectory distributions — critical for the stochastic contact-rich tasks of the microfactory assembly benchmark. The **GR00T** hierarchical control architecture [4] separates the VLA (high-level intent, ~50 Hz) from a Whole-Body Controller (WBC) trained via Isaac Lab RL (low-level physics).

---

### 2. Pillar I — Task Belief & Skill Cartridge Creation (V-JEPA 2 + LoRA)

**Neuroscience source:** Primate prefrontal cortex (PFC) neurons maintain an *internal task belief* — a continuously updated probabilistic representation of the current task context that biases downstream motor representations *before* any movement begins [1]. This is anticipatory, not reactive: the PFC conditions the motor system on the expected task structure before sensory feedback arrives.

**Computational analogue — V-JEPA 2 / V-JEPA 2-AC [2]:** V-JEPA 2 learns latent world models from unlabeled video by predicting future latent representations. Its encoder embeddings capture structured temporal predictions — what the world *should look like* after an action sequence — constituting a predictive prior that mirrors the PFC's internal task belief. V-JEPA 2-AC extends this to closed-loop action conditioning, enabling planning without environment interaction.

#### The Skill Cartridge Pipeline

Pillar I is also where the entire **skill acquisition process begins**. The four-step pipeline below describes how a new manipulation skill enters the architecture, is trained, stored, and eventually deployed — with each subsequent pillar playing its role in that flow.

**Step 1 — Freeze the core backbone (the shared brain).** The pre-trained smolVLA is locked entirely. `requires_grad = False` is set on all original parameters. The backbone is mathematically frozen and cannot experience catastrophic forgetting regardless of how many skills are subsequently trained on top of it.

**Step 2 — Attach a blank LoRA cartridge.** Two small matrices `A` and `B` are injected alongside the frozen layers. At inference, their product is added to the frozen layer's output:

```
Output = W_frozen(x) + (A × B)(x)
```

The frozen backbone provides the generalizable prior — physics, spatial reasoning, language grounding — while the cartridge learns to steer it toward the specific skill. Pillar I's V-JEPA 2 task belief embedding provides the *anticipatory context* that primes which parts of the backbone are relevant before training on demonstrations begins.

**Step 3 — Train the specific skill.** Human teleoperation demonstrations are collected for one task (e.g., *"insert connector onto PCB"*). Backpropagation only updates `A` and `B`. Pillar II (DGMoE) routes the training signal through the relevant expert subset. The cartridge learns to steer the backbone's outputs to execute the skill successfully.

**Step 4 — Save and swap.** Only `A` and `B` are saved as a standalone file. A full smolVLA backbone is ~900 MB; a single skill cartridge is ~20 MB. At deployment, cartridges are loaded and composed without reloading the backbone — just as the monkeys in [1] flexibly engaged the shared sensory and motor subspaces relevant to the current task. Pillar III (LoRAC) then enforces that the subspaces of different cartridges remain mathematically orthogonal, and Pillar IV (PLD) distills any residual physical correction back into the cartridge before it is finalized.

#### V-JEPA 2 Fusion

The V-JEPA 2 task belief embedding is fused into the smolVLA transformer backbone via **Flamingo-style gated cross-attention** [10] inserted every 8 transformer layers:

```python
class GatedCrossAttention(nn.Module):
    """Flamingo-style fusion of V-JEPA 2 task belief into backbone.

    Biological analogue: PFC task belief gating downstream motor representations.
    Gate initialized to zero — fusion is transparent at training start.
    """
    def __init__(self, dim: int, n_heads: int):
        super().__init__()
        self.attn = nn.MultiheadAttention(dim, n_heads)
        self.gate = nn.Parameter(torch.zeros(1))  # tanh gate, init = 0

    def forward(self, x: Tensor, jepa_context: Tensor) -> Tensor:
        attn_out, _ = self.attn(x, jepa_context, jepa_context)
        return x + torch.tanh(self.gate) * attn_out
```

Key design decisions derived from the neuroscientific finding:
- Gate initialized to **zero**: fusion is off at training start, preventing early interference — analogous to the PFC task belief requiring a learning period to become reliably predictive.
- Fusion layers trained at **0.1× backbone LR**: preserves pre-trained backbone knowledge during fusion integration.
- V-JEPA 2 encoder **frozen**: the world model prior is treated as fixed structured knowledge, not a trainable component.

**Empirical anchor [2]:** 80% zero-shot cup-lift success vs. 15% for Octo; 16 s/action vs. 4 min for Cosmos world model.

---

### 3. Pillar II — Active Suppression (DGMoE)

**Neuroscience source:** PFC and premotor neurons do not merely *attend* to task-relevant information — they actively *suppress* task-irrelevant neural dimensions [1]. This is a hard mechanism: irrelevant pathway activations are driven near zero, not merely down-weighted. Suppression is context-dependent, switches rapidly between tasks, and precedes movement execution — indicating it is predictively invoked.

**Computational analogue — DGMoE (Decoupled Gating Mixture of Experts) [6]:** Standard MoE uses soft routing — all experts receive a weighted combination. DGMoE decouples expert **selection** (hard binary: use / don't use) from expert **weighting** (continuous scale for selected experts). For unselected experts, the effective contribution is driven to zero — a direct implementation of hard active suppression.

```
Standard MoE:
    output = Σᵢ softmax(gate)ᵢ · expertᵢ(x)         ← soft, all experts contribute

DGMoE:
    selected = top_k( selection_gate(x) )             ← hard binary selection
    weights  = softmax( weighting_gate(x)[selected] )
    output   = Σᵢ∈selected weightsᵢ · expertᵢ(x)    ← unselected: weight = 0
```

The **AdaMoE scale adapter** provides per-expert learnable scaling to correct load imbalance without auxiliary routing losses, which can destabilize training in multi-task settings. The decoupling mirrors the neuroscientific finding: separate mechanisms control *which* circuits to engage (binary selection) and *how strongly* to engage them (continuous modulation).

---

### 4. Pillar III — Shared Subspaces (LoRAC)

**Neuroscience source:** Motor cortex (M1) neurons encode compositionally related tasks (reach-then-grasp, reach-then-place, reach-then-insert) in **shared low-dimensional subspaces** [1]. The subspace geometry itself is reused across task variants; task specificity is captured by a small residual projection *within* the subspace. This is the primary mechanism enabling rapid compositional generalization to novel task combinations.

**Computational analogue — LoRAC / QR-LoRA [7][8]:** LoRAC parameterizes each skill's weight update using QR decomposition, enforcing mathematical orthogonality between skill subspaces:

```
For each skill t:
    Aₜ = Qₜ · Rₜ          (QR decomposition of adapter matrix)
    Qₜ frozen after init   (shared subspace basis — the biological "subspace")
    Only ΔRₜ is trained    (task-specific residual — the biological "specialization")

Orthogonality constraint:
    Aᵢᵀ · Aⱼ ≈ 0  for i ≠ j    (enforced by QR; prevents cross-skill interference)
```

The biological mapping is precise at two levels:
- `Qₜ` (frozen after init) = shared neural subspace: the structural prior that makes compositional generalization possible.
- `ΔRₜ` (trained per skill) = task-specific residual within the subspace: specialization on top of shared structure.
- Cross-skill orthogonality = suppression of inter-task interference in parameter space, directly preventing catastrophic forgetting.

Each trained `(Qₜ, Rₜ)` pair constitutes a **skill cartridge** — a self-contained, storable, composable unit of manipulation knowledge that can be loaded, merged, and swapped at inference time without full model reloading.

**Parameter efficiency:** LoRAC trains approximately 50% fewer parameters than standard LoRA for equivalent rank, because only the upper-triangular `Rₜ` (r×r) is updated rather than both A (d×r) and B (r×d) factors.

**Empirical results [8]:** +6.35% baseline accuracy, −3.24% catastrophic forgetting on Split CIFAR-100 continual learning benchmark vs. standard LoRA.

---

### 5. Pillar IV — Cerebellar Correction (Residual RL / PLD)

**Neuroscience source:** The cerebellum acts as a high-frequency error corrector for cortical motor commands [1]. It receives an *efference copy* of the cortical motor plan and generates a **residual correction term** added to the final motor output — operating at a finer temporal granularity than the cortex, and particularly active for fast, precise movements requiring sub-millimeter accuracy.

**Computational analogue — Residual RL / PLD (Probe–Learn–Distill) [5][9]:** The final motor action is the sum of the base VLA policy and a lightweight RL-trained residual agent:

```
a_final = a_base + a_res
```

Where `a_base` is the smolVLA output (the "cortical command") and `a_res` is a small RL-trained network that takes the physical state and `a_base` as input (the "efference copy"), outputting a bounded correction. `a_res` is initialized to zero-mean so the system degrades gracefully to the base policy.

The **PLD training protocol** provides the learning mechanism for this correction:

```
Stage 1 — PROBE
    Execute a_base in environment. Record failure states and outcomes.
    Identify where cortical commands are systematically insufficient.

Stage 2 — LEARN / Hybrid Rollout
    Train residual RL agent a_res on identified failure states.
    Hybrid rollout: a_final = a_base + β·a_res, with β annealed 1.0 → 0.0.
    Prevents the residual agent from diverging from the base policy regime.

Stage 3 — DISTILL
    Distill (a_base + a_res) back into VLA backbone via SFT.
    Deployed model: a_final = a_base only (cerebellar correction absorbed).
    Net result: a VLA that has internalized the physical correction.
```

This mirrors biological motor consolidation: the cerebellum's correction is progressively internalized by the cortex through long-term synaptic plasticity, until the correction is no longer needed as a separate signal.

**COMPASS [9]** demonstrates that residual RL generalizes across robot morphologies without full retraining — enabling cross-embodiment transfer from SO-ARM101 to open-kumanday-humanoid without relearning from scratch.

**Empirical anchors [5]:** ~99% task success on LIBERO benchmark; 100% success rate on 1-hour GPU insertion/unplugging stress cycle (Franka + YAM bimanual).

---

## The Four Biological-to-Computational Mappings

| Neurobiological Mechanism | Brain Region | Computational Analogue | Key Reference |
|---|---|---|---|
| **Internal Task Belief** — iterative update of task context | Prefrontal cortex | V-JEPA 2 latent predictive embeddings + Flamingo cross-attention fusion | [2][10] |
| **Active Suppression** — hard gating drives irrelevant neural dims → 0 | PFC / premotor | DGMoE decoupled hard selection drives unselected expert weights → 0 | [6] |
| **Shared Neural Subspaces** — low-dim subspaces reused across related tasks | Motor cortex (M1) | LoRAC QR-decomposed orthogonal skill cartridges; Q frozen = shared subspace | [7][8] |
| **Cerebellar Error Correction** — residual correction term added to cortical commands | Cerebellum → M1 | Residual RL / PLD: a_final = a_base + a_res; distilled back via SFT | [5][9] |

---

## Skill Creation and Compositional Learning

This section describes the concrete workflow for creating new skills and how the system uses skill subsets to perform compositional tasks it has never seen before. It is the operational heart of the architecture — the mechanism that makes the biological analogy actionable.

### What Is a Skill?

A **skill** in open-poporo-vla is a named, self-contained unit of manipulation knowledge defined by three things:

| Component | What it is | Stored as |
|---|---|---|
| **Language tag** | Human-readable name and trigger description | String + embedding |
| **LoRAC cartridge** | `(Qₜ, Rₜ)` adapter pair encoding the motor subspace for this skill | `.pt` file |
| **Residual correction** | PLD-distilled physical correction for this skill's failure modes | Absorbed into cartridge after distillation |

A skill is **not** a full model. It is a lightweight delta on top of the frozen smolVLA backbone — typically a few MB per skill regardless of how many skills are accumulated.

#### Skill boundary definition

A skill boundary is defined by the **homogeneity of the required motor subspace**, not by task semantics. Concretely, two action sequences belong to the same skill if their neural trajectories in M1 — and by analogy, their LoRAC adapter activations — lie in the same low-dimensional subspace. In practice, this means:

- `pick_pcb`, `pick_bolt`, `pick_connector` → **three separate skills** (different grasp geometries → different subspaces)
- `place_on_bracket`, `place_in_tray`, `place_at_station` → **three separate skills** (different placement constraints → different subspaces)
- `pick_pcb` executed with left arm vs. right arm → **same skill, different embodiment** (COMPASS cross-embodiment transfer applies)

This definition is deliberately conservative: finer skill granularity means smaller, more orthogonal cartridges and better compositional generalization.

---

### Skill Creation Workflow

Creating a new skill follows a four-stage pipeline

This architecture is grounded in the following research:

**Primary neuroscience paper (architectural foundation):**
> S. Tafazoli et al., "Building compositional tasks with shared neural subspaces," *Nature*, vol. 650, pp. 164–172, Feb. 2026. DOI: 10.1038/s41586-025-09805-2 [1]

The central empirical contribution of [1] is the simultaneous identification and dissociation of all four mechanisms in the same primate motor system during compositional task learning. Critically, the paper demonstrates that disrupting any single mechanism degrades performance in a qualitatively distinct way: loss of task belief produces misrouted execution; loss of active suppression produces task confusion; loss of shared subspaces eliminates compositional transfer; loss of cerebellar correction degrades physical precision without affecting task-level planning. This four-way dissociation directly motivates the four-pillar architecture — each pillar targets a distinct failure mode.

**Component findings that motivate this architecture:**
- Assran et al. [2] show V-JEPA 2-AC achieves **80% zero-shot** manipulation success vs. 15% for the best prior method, confirming that latent predictive priors dramatically reduce data required for new skill acquisition.
- Yang et al. [8] show QR-decomposed LoRA reduces catastrophic forgetting by **3.24 percentage points** on Split CIFAR-100, confirming that orthogonal subspace enforcement is sufficient to prevent inter-skill interference without replay or regularization.
- Xiao et al. [5] show the PLD protocol achieves **~99%** task success on LIBERO and **100%** on a 1-hour physical stress test, confirming that residual RL corrections generalize beyond the training distribution when distilled back into the base policy.

These findings establish that each pillar has been independently validated for the specific capability it provides, motivating their integration into a unified architecture.

---

## System Architecture

```
 Natural Language Instruction + Goal Image
                    │
                    ▼
 ┌──────────────────────────────────────────────────────────────────┐
 │           PILLAR I — TASK BELIEF (V-JEPA 2 / V-JEPA 2-AC)       │
 │   Video latent world model · predictive prior over task context  │
 │   Fused into backbone via Flamingo gated cross-attn (every 8L)  │
 │   tanh gate init=0 · frozen encoder · 0.1× LR for fusion layers │
 └──────────────────────────────┬───────────────────────────────────┘
                                │  task belief embedding
                                ▼
 ┌──────────────────────────────────────────────────────────────────┐
 │                    smolVLA BACKBONE (~450M)                       │
 │   RGB-D → PixelShuffle (64 tokens) → Transformer (L/2 skip)     │
 │                                                                  │
 │   ┌──────────────────────────────────────────────────────────┐   │
 │   │        PILLAR II — ACTIVE SUPPRESSION (DGMoE)            │   │
 │   │  Replaces FFN layers · decoupled selection + weighting   │   │
 │   │  Unselected experts: weight → 0 (hard suppression)       │   │
 │   │  AdaMoE scale adapter · top-k routing per token          │   │
 │   └──────────────────────────────────────────────────────────┘   │
 │                                                                  │
 │   ┌──────────────────────────────────────────────────────────┐   │
 │   │       PILLAR III — SHARED SUBSPACES (LoRAC)              │   │
 │   │  QR adapters: Aₜ = Qₜ·Rₜ                                │   │
 │   │  Q frozen (shared subspace) · ΔR trained (skill delta)   │   │
 │   │  AᵢᵀAⱼ ≈ 0 → no catastrophic forgetting                 │   │
 │   │  Cartridge manager: load / save / compose at runtime     │   │
 │   └──────────────────────────────────────────────────────────┘   │
 │                                                                  │
 │              Flow Matching Action Expert → a_base               │
 └──────────────────────────────┬───────────────────────────────────┘
                                │  a_base (50 Hz+)
                                ▼
 ┌──────────────────────────────────────────────────────────────────┐
 │         PILLAR IV — CEREBELLAR CORRECTION (Residual RL / PLD)    │
 │   Residual agent: (state, a_base) → a_res · init: a_res = 0     │
 │   a_final = a_base + a_res                                       │
 │   PLD: Probe → Learn (hybrid rollout) → Distill (SFT)           │
 └──────────────────────────────┬───────────────────────────────────┘
                                │  a_final
                                ▼
 ┌──────────────────────────────────────────────────────────────────┐
 │              GR00T WHOLE-BODY CONTROLLER (WBC)                   │
 │   Isaac Lab RL policy · joint torque commands                    │
 │   Required for open-kumanday-humanoid deployment                 │
 └──────────────────────────────┬───────────────────────────────────┘
                                │
                                ▼
                         Robot Actuators
```

---

## Roadmap

### Phase 0 — Architecture Design & Foundation
| Component | Status |
|---|---|
| Architecture design — four pillars and integration strategy | ✅ Done |
| Biological-to-computational mapping documentation | ✅ Done |
| smolVLA backbone selection and interface analysis | ✅ Done |
| Component paper survey (V-JEPA 2, DGMoE, LoRAC, PLD) | ✅ Done |

### Phase 1 — Simulation Environment Setup
| Component | Status |
|---|---|
| SO-ARM101 dual-arm URDF / USD import for Isaac Lab | ⏳ Planned |
| Dual-arm microfactory scene construction | ⏳ Planned |
| Bimanual action space and multi-camera observation definition | ⏳ Planned |
| LIBERO evaluation harness integration | ⏳ Planned |
| Teleoperation pipeline for demonstration collection | ⏳ Planned |

### Phase 2 — Individual Pillar Implementation
| Component | Status |
|---|---|
| Pillar I — V-JEPA 2 encoder integration + Flamingo gated cross-attention | ⏳ Planned |
| Pillar II — DGMoE layer + AdaMoE adapter replacing backbone FFN layers | ⏳ Planned |
| Pillar III — LoRAC QR decomposition + cartridge manager + orthogonality tests | ⏳ Planned |
| Pillar IV — Residual RL agent + full PLD Probe → Learn → Distill pipeline | ⏳ Planned |
| Unit tests: Q orthogonality, zero-init a_res, DGMoE hard suppression | ⏳ Planned |

### Phase 3 — Integration & Ablation
| Component | Status |
|---|---|
| End-to-end forward pass with all four pillars active simultaneously | ⏳ Planned |
| LIBERO benchmark: per-pillar ablation study | ⏳ Planned |
| Multi-skill continual learning experiment (5 tasks sequential) | ⏳ Planned |
| GR00T WBC integration for humanoid deployment | ⏳ Planned |
| Ablation: task belief on/off · DGMoE vs standard MoE · LoRAC vs LoRA · PLD vs no correction | ⏳ Planned |

### Phase 4 — Hardware Validation
| Component | Status |
|---|---|
| SO-ARM101 dual-arm microfactory assembly evaluation (primary) | ⏳ Planned |
| open-pquaca-arm single-arm tabletop skill acquisition | ⏳ Planned |
| 1-hour physical stress test (insertion / unplugging cycle) | ⏳ Planned |
| Cross-embodiment transfer via COMPASS (SO-ARM101 → open-kumanday) | ⏳ Planned |

### Phase 5 — Hybrid Architecture & Long-Horizon Evaluation
| Component | Status |
|---|---|
| Integration bridge with neurosymbolic-vla (symbolic decomposition + neural execution) | ⏳ Planned |
| Long-horizon evaluation (>10 sequential sub-skills) on custom microfactory benchmark | ⏳ Planned |
| open-kumanday-humanoid full-body loco-manipulation deployment | ⏳ Planned |
| Technical report / preprint | ⏳ Planned |

---

## References

[1] S. Tafazoli, K. K. Bhatt, A. Bhattacharyya, M. G. Esteban, D. Milstein, E. E. Bouchacourt, J. Zhu, and C. D. Harvey, "Building compositional tasks with shared neural subspaces," *Nature*, vol. 650, pp. 164–172, Feb. 2026. DOI: [10.1038/s41586-025-09805-2](https://doi.org/10.1038/s41586-025-09805-2)

[2] M. Assran *et al.*, "V-JEPA 2: Self-Supervised Video Models Enable Understanding, Prediction and Planning," *arXiv preprint arXiv:2506.09985*, 2025.

[3] H. Cadene *et al.*, "SmolVLA: A Vision-Language-Action Model for Affordable and Efficient Robotics," *arXiv preprint arXiv:2506.01844*, 2025.

[4] NVIDIA, "Building Generalist Humanoid Capabilities with NVIDIA Isaac GR00T N1.6," NVIDIA Developer Blog, 2025. [Online]. Available: https://developer.nvidia.com/blog/

[5] W. Xiao, Y. Mao, S. Zhao, and P. Stone, "Self-Improving Vision-Language-Action Models with Data Generation via Residual RL," *arXiv preprint arXiv:2511.00091*, 2024.

[6] Z. Wang *et al.*, "FedVLA: Federated Vision-Language-Action Learning with Dual Gating Mixture-of-Experts for Robotic Manipulation," *arXiv preprint arXiv:2508.02190*, 2025.

[7] X. Yang, Z. Chen, B. Ding, X. Tang, H. Chen, L. Zhao, and F. Wu, "Orthogonal Subspace Learning for Language Model Continual Learning," in *Findings of the Assoc. for Computational Linguistics: EMNLP 2023*, pp. 715–730, 2023. DOI: [10.18653/v1/2023.findings-emnlp.715](https://doi.org/10.18653/v1/2023.findings-emnlp.715)

[8] Y. Yang *et al.*, "QR-LoRA: Efficient and Disentangled Fine-tuning via QR Decomposition," in *Proc. Int. Conf. Computer Vision (ICCV)*, 2025.

[9] J. Alonso *et al.*, "COMPASS: Cross-embodiment Mobility Policy via Residual RL and Skill Synthesis," *arXiv preprint arXiv:2502.16372*, 2025.

[10] J.-B. Alayrac *et al.*, "Flamingo: a Visual Language Model for Few-Shot Learning," in *Advances in Neural Information Processing Systems (NeurIPS)*, vol. 35, 2022.

[11] Y. Lipman, R. T. Q. Chen, H. Ben-Hamu, M. Nickel, and M. Le, "Flow Matching for Generative Modeling," in *Proc. Int. Conf. Learning Representations (ICLR)*, 2023.

---

## Author

**Gilberto Galvis Giraldo**
M.Sc. Electrical and Computer Engineering — Sungkyunkwan University

---

## License

Apache License, Version 2.0 — see [LICENSE](LICENSE) for details.

This repository contains no proprietary code from any third party. All implementations are original works of the author released under the Apache 2.0 License.
