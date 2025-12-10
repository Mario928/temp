# Expert Prediction for Mixtral 8x7B - Speculative Pipeline Parallelism

## Goal
Predict which expert will be selected at Layer N+1 based on expert decisions at Layer N (or earlier), enabling speculative pre-loading of expert weights.

## Model Info
- **Model**: Mixtral 8x7B (Mixture of Experts)
- **Layers**: 32 MoE layers
- **Experts per layer**: 8
- **Top-K routing**: 2 experts selected per token per layer

---

## Files

### Data Files (20 token journeys)
| File | Description |
|------|-------------|
| `p0_token_0.jsonl` - `p0_token_9.jsonl` | 10 tokens from HumanEval problem 0 |
| `p1_token_0.jsonl` - `p1_token_9.jsonl` | 10 tokens from HumanEval problem 1 |

Each file contains 32 entries (one per layer) with expert routing decisions.

### Reusable Scripts
| Script | Description |
|--------|-------------|
| `create_rules.py` | Create rules from specified token journey files |
| `evaluate_rules.py` | Evaluate rules on specified test files |
| `extract_tokens.py` | Extract token journeys from raw routing data |

### Rule Files
| File | Description |
|------|-------------|
| `rules_from_1token.json` | Rules from p0_token_0 only (31 rules) |
| `rules_from_5tokens.json` | Rules from 5 tokens (majority vote) |

---

## 5 Prediction Levels

### Level 1: Simple (Layer, Expert) → Next Expert
```
Input:  Layer N, Expert E
Output: Predicted Expert at Layer N+1

Example: Layer 5, Expert 6 → Expert 3
```
**Accuracy: 44.9%** (31/69 correct)

### Level 2: Two Layers Back
```
Input:  Layer N, Previous Expert, Current Expert
Output: Predicted Expert at Layer N+1

Example: Layer 5, (Prev=0, Curr=6) → Expert 3
```
**Accuracy: 83.3%** (25/30 correct)

### Level 3: Three Layers Back (BEST)
```
Input:  Layer N, Expert at N-2, Expert at N-1, Expert at N
Output: Predicted Expert at Layer N+1

Example: Layer 5, (E-2=3, E-1=0, E=6) → Expert 3
```
**Accuracy: 87.5%** (21/24 correct) 🏆

### Level 4: Pair-Based (Primary + Secondary)
```
Input:  Layer N, Primary Expert, Secondary Expert
Output: Predicted Primary Expert at Layer N+1

Example: Layer 5, (Primary=6, Secondary=0) → Expert 3
```
**Accuracy: 65.7%** (23/35 correct)

### Level 5: Skip Layers
```
Input:  Layer N, Expert E
Output: Predicted Expert at Layer N+K (K=2,4,8)

Example: Layer 5, Expert 6 → Expert at Layer 7
```
**Accuracy: 48-55%**

---

## Results Summary

| Level | Method | Accuracy | vs Random (12.5%) |
|-------|--------|----------|-------------------|
| 1 | (Layer, Expert) | 44.9% | 3.6x better |
| 2 | (Layer, Prev, Curr) | 83.3% | 6.7x better |
| **3** | **(Layer, E-2, E-1, E)** | **87.5%** | **7x better** |
| 4 | (Layer, Prim, Sec) | 65.7% | 5.3x better |
| 5 | Skip layers | 48-55% | 4x better |

---

## How to Run (Reusable Scripts)

### Step 1: Create Rules from Training Tokens
```bash
# Create rules from 1 token (p0_token_0)
python create_rules.py p0_token_0.jsonl --output rules_from_1token.json

# Create rules from multiple tokens (majority vote)
python create_rules.py p0_token_0.jsonl p0_token_1.jsonl p0_token_2.jsonl --output rules_from_3tokens.json

# Create rules from all problem 0 tokens
python create_rules.py p0_token_*.jsonl --output rules_from_p0.json
```

### Step 2: Evaluate on Test Tokens
```bash
# Evaluate on specific test files
python evaluate_rules.py --rules rules_from_1token.json --test p0_token_1.jsonl p0_token_2.jsonl

# Evaluate on all tokens from problem 1
python evaluate_rules.py --rules rules_from_1token.json --test p1_token_*.jsonl

# Evaluate on ALL tokens
python evaluate_rules.py --rules rules_from_1token.json --test p0_token_*.jsonl p1_token_*.jsonl
```

### Example Output
```
EVALUATION RESULTS
Level 1 (Layer, E)             8          22         36.4%        2.9x
Level 2 (Layer, Prev, Curr)    20         29         69.0%        5.5x
Level 3 (Layer, E-2, E-1, E)   21         24         87.5%        7.0x
Level 4 (Layer, P, S)          10         22         45.5%        3.6x

🏆 BEST: level3 with 87.5% accuracy
```

---

## Methodology

### Step 1: Create Rules from Token 0
```
Token 0 journey: [6, 6, 7, 3, 0, 6, 3, 0, 7, 6, 0, 7, ...]

Level 1 Rule Creation:
  Layer 0: Expert 6 → Layer 1: Expert 6
  RULE: (Layer=0, Expert=6) → Next=6

  Layer 1: Expert 6 → Layer 2: Expert 7
  RULE: (Layer=1, Expert=6) → Next=7

  ... (31 rules total)
```

### Step 2: Validate on Tokens 1-9
```
Token 1 at Layer 5: Expert 1
  → Check rule for (Layer=5, Expert=1)
  → No rule exists (Token 0 had Expert 6 at Layer 5)
  → Skip

Token 1 at Layer X: Expert 6
  → Check rule for (Layer=X, Expert=6)
  → Rule exists! Predicts Expert Y
  → Compare with actual → Count correct/wrong
```

### Step 3: Calculate Accuracy
```
Accuracy = Correct Predictions / Total Tested
         = 31 / 69 = 44.9% (Level 1)
```

---

## Key Insights

1. **More context = Higher accuracy**
   - Level 1 (1 expert): 44.9%
   - Level 2 (2 experts): 83.3%
   - Level 3 (3 experts): 87.5%

2. **More context = Fewer matches**
   - Level 1: 69 transitions tested
   - Level 2: 30 transitions tested
   - Level 3: 24 transitions tested

3. **Recommended: Hybrid Approach**
   ```
   1. Try Level 3 rule first (87.5% accurate)
   2. If no rule, try Level 2 (83.3% accurate)
   3. If no rule, try Level 1 (44.9% accurate)
   4. If no rule, use most common expert for that layer
   ```

---

## Practical Application

For Speculative Pipeline Parallelism:
1. At Layer N, observe which expert was selected
2. Look up prediction rule for Layer N+1
3. Pre-load predicted expert weights BEFORE router computes
4. If prediction correct (87.5% with Level 3): Zero wait time!
5. If prediction wrong: Fall back to normal loading (no worse than baseline)

---

## Data Format

Each `.jsonl` file contains 32 entries (one per layer):
```json
{
  "layer": 5,
  "experts": [6, 0],      // [primary, secondary]
  "gating_probs": [0.85, 0.15],
  "token_idx": 0
}
```
