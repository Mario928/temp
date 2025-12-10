# Experimental Evaluation
## Expert Routing Prediction for Speculative Pipeline Parallelism

---

## Slide 1: Experimental Setup

### Objective
Analyze MoE routing decisions to find predictable patterns for speculative expert pre-loading

### Dataset
| Property | Value |
|----------|-------|
| Model | Mixtral 8x7B |
| Dataset | HumanEval (code generation) |
| MoE Layers | 32 |
| Experts per Layer | 8 (top-2 selected) |
| Tokens Analyzed | 20 (10 from problem 0, 10 from problem 1) |

### Methodology
```
1. Extract token journeys (32-layer routing decisions per token)
2. Create prediction rules from training tokens
3. Validate rules on unseen test tokens
4. Measure accuracy across 5 prediction strategies
```

### Train/Test Split
- **Training**: First 5 tokens from problem 0 (p0_token_0 to p0_token_4)
- **Testing**: 15 tokens (p0_token_5-9 + p1_token_0-9)

---

## Slide 2: Prediction Strategies (5 Levels)

### Level 1: Single Expert Context
```
Input:  (Layer N, Expert E)
Output: Predicted Expert at Layer N+1
Rules:  107 unique patterns
```

### Level 2: Two-Expert History
```
Input:  (Layer N, Expert at N-1, Expert at N)
Output: Predicted Expert at Layer N+1
Rules:  138 unique patterns
```

### Level 3: Three-Expert History (BEST)
```
Input:  (Layer N, Expert at N-2, Expert at N-1, Expert at N)
Output: Predicted Expert at Layer N+1
Rules:  139 unique patterns
```

### Level 4: Expert Pair (Primary + Secondary)
```
Input:  (Layer N, Primary Expert, Secondary Expert)
Output: Predicted Primary at Layer N+1
Rules:  138 unique patterns
```

### Level 5: Skip-Layer Prediction
```
Input:  (Layer N, Expert E)
Output: Predicted Expert at Layer N+K (K = 2, 4, 8)
```

---

## Slide 3: Results

### Accuracy by Prediction Level

| Level | Strategy | Accuracy | vs Random (12.5%) |
|-------|----------|----------|-------------------|
| 1 | (Layer, Expert) | 30.1% | 2.4x |
| 2 | (Layer, Prev, Curr) | 44.0% | 3.5x |
| **3** | **(Layer, E-2, E-1, E)** | **65.8%** | **5.3x** |
| 4 | (Layer, Primary, Secondary) | 43.7% | 3.5x |
| 5 | Skip Layers (K=2,4,8) | 26-28% | 2.2x |

### Cross-Problem Generalization

| Test Set | Level 3 Accuracy | Observations |
|----------|------------------|--------------|
| Same problem (p0 tokens 5-9) | 33.3% | Different token positions = different patterns |
| Different problem (p1 tokens 0-9) | **75.9%** | Same token positions = similar routing |
| Combined (all 15 tokens) | 65.8% | Weighted average |

### Key Result
**Level 3 achieves 65.8% accuracy** - predicting the correct expert 5.3x better than random selection

---

## Slide 4: Key Findings & Implications

### Finding 1: Context Depth Matters
```
More expert history → Higher prediction accuracy
Level 1 (1 expert):  30.1%
Level 2 (2 experts): 44.0%
Level 3 (3 experts): 65.8%  ← Sweet spot
```

### Finding 2: Token Position > Problem Identity
- Tokens at **same position** across different problems share routing patterns
- p1_token_0 matches 18/20 Level 3 rules from p0 training data
- Suggests **positional routing consistency** in MoE

### Finding 3: Trade-off Between Coverage and Accuracy
| Training Size | Rules | Accuracy | Coverage |
|---------------|-------|----------|----------|
| 1 token | 29 | 87.5% | Low |
| 5 tokens | 139 | 65.8% | High |

### Practical Implication for Speculative Pre-loading
```
Recommended Hybrid Strategy:
1. Try Level 3 prediction first (65.8% hit rate)
2. Fallback to Level 2 if no match (44% hit rate)
3. Fallback to Level 1 if no match (30% hit rate)
4. Default to most frequent expert for layer
```

**Result: Predictable patterns exist in MoE routing that can enable speculative expert pre-loading**

---
