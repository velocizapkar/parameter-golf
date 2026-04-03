# Thought Bubbles for Parameter Golf

## Core Idea

Insert **B learnable "thought bubble" compute tokens** into each training sequence so the model gets extra forward-pass compute to refine representations before predicting. The transformer always processes a fixed-length sequence (2048), but B of those positions are learned embeddings rather than real tokens.

The model is free to learn whatever representation management strategies it wants in the thought bubble positions — they receive gradient from all downstream real tokens via causal attention.

---

## Architecture

**Base stack:** 11L 512d transformer, Parallel Muon, LeakyReLU², XSA-all, Full GPTQ (int6 + Hessian-aware + AR self-gen calibration), partial RoPE, ValueEmbedding, EMA, selective pruning. Current best BPB: ~1.1147.

**Thought bubble parameters:** `nn.Parameter(B, model_dim)` initialized to zero. Tiny footprint (B=128 = 65K params = ~48KB at int6).

**Hyperparameters:**
| Env var | Default | Description |
|---------|---------|-------------|
| `THOUGHT_BUBBLE_SIZE` | `0` | Number of thought tokens B (0 = disabled) |

Insertion position is always computed automatically as `real_seq_len - 64` (i.e. 64 tokens before the end).

**Sweep:** `B` in {0, 32, 64, 128, 256}, 3 seeds each.

---

## Training

### Data Loading

With B > 0, each training chunk uses **2048 - B real tokens** instead of 2048:

```
Standard:    take 2048 contiguous tokens from stream
With TB:     take 2048-B contiguous tokens from stream
```

### Sequence Assembly

The 2048-B real tokens are assembled into a 2048-length transformer input:

```
Position:  [0 ................. 2048-B-64)  [2048-B-64 ... 2048-64)  [2048-64 ... 2048)
Content:   real context tokens               B thought bubbles        64 real tokens
Count:     2048 - B - 64                     B                        64
           \___________ context ___________/ \____ compute ____/     \____ tail ____/
```

Total: `(2048 - B - 64) + B + 64 = 2048` -- **fixed sequence length, no recompile, flash attention happy.**

### Loss

- Computed on **all 2048 - B real token positions** (standard next-token prediction)
- Thought bubble positions are **masked** — no loss, no targets
- The B thought tokens receive gradient indirectly through causal attention from every real token that follows them (the last 64)

### Gradient Flow

```
Context tokens (2048-B-64)
    |
    v  [causal attention]
Thought bubbles (B)  <-- no loss, but gradient flows back from...
    |
    v  [causal attention]
Tail tokens (64)     <-- loss here (and on all context tokens too)
```

All real tokens get loss. The thought bubbles are just B learned positions the model can use as a scratchpad — the model is free to learn whatever representation strategies it wants there.

---

## Evaluation

### Sliding Window (Primary — produces submission score)

Each window consumes **2048 - B real tokens** from the validation stream, assembled identically to training:

```
Window k:  real[64k : 64k + 2048-B]
Assembly:  [context: 2048-B-64 tokens] + [B thought bubbles] + [64 scoring tokens]
Score:     last 64 real tokens only
Stride:    64 real tokens
```

Window progression over real token stream:
```
Window 0:  real[0    : 1920]     score real[1856 : 1920]
Window 1:  real[64   : 1984]     score real[1920 : 1984]
Window 2:  real[128  : 2048]     score real[1984 : 2048]
...
```

(Example numbers for B=128)

The previously-scored 64 tokens slide left into context, new 64 arrive on the right. Thought bubbles are freshly inserted every window at the same relative position.

### Standard Eval (Diagnostic)

Same assembly as training — 2048-B real tokens per chunk, thought bubbles inserted, loss on all real positions. Used for quick validation during training, not for final scoring.

---

## B = 0 Baseline

When `THOUGHT_BUBBLE_SIZE=0`:
- Data loader takes 2048 tokens (unchanged)
- No insertion, no masking
- Loss on all 2048 positions
- Sliding window eval unchanged
- **Identical to original behavior** — clean baseline

---

## Budget

| B | Thought params | Param bytes (int6) | Seq overhead |
|---|---------------|-------------------|--------------|
| 0 | 0 | 0 | 0% |
| 32 | 16,384 | ~12 KB | context shrinks by 32 |
| 64 | 32,768 | ~24 KB | context shrinks by 64 |
| 128 | 65,536 | ~48 KB | context shrinks by 128 |
| 256 | 131,072 | ~96 KB | context shrinks by 256 |

The transformer always processes 2048 tokens — no throughput change. The only cost is reduced real context (2048-B-64 instead of 2048-64).

---

## Key Design Decisions

1. **Learned embeddings, zero-init.** Start from nothing; let gradient shape them.
2. **Fixed sequence length.** No shape changes = torch.compile + flash attn work perfectly.
3. **Full causal attention.** Thought tokens attend to all prior context. Real tokens after the bubble attend to thought tokens. This is the mechanism — the model reads its own "thoughts."
4. **Loss on all real tokens.** Not just the 64 after the bubble. The model trains normally on everything; the bubble is just B masked positions spliced in.
5. **Position 2048-B-64.** Thought bubbles sit right before the final 64 tokens, maximizing context available to the bubble while giving the tail tokens the freshest "thoughts."
6. **Optimizer:** Thought embeddings go in the scalar AdamW group (same LR as scale params, with distributed all-reduce).
