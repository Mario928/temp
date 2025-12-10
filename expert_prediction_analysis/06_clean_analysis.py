#!/usr/bin/env python3
"""
CLEAN ANALYSIS - Rules from Token 0, Evaluate on Tokens 1-9
Show ALL rules clearly
"""

import json

print("="*70)
print("CLEAN ANALYSIS: Rules from Token 0 → Evaluate on Tokens 1-9")
print("="*70)

# Load and filter data (1 entry per token per layer)
data = [json.loads(l) for l in open('../humaneval_2_routing.jsonl')]
seen = set()
clean = []
for d in data:
    key = (d['token_idx'], d['layer'])
    if key not in seen:
        seen.add(key)
        clean.append(d)

# Extract token journeys (first 10 tokens)
def get_journey(token_idx):
    j = [d for d in clean if d['token_idx'] == token_idx]
    j.sort(key=lambda x: x['layer'])
    return j

token0 = get_journey(0)
test_tokens = {i: get_journey(i) for i in range(1, 10)}

print(f"\nToken 0: {len(token0)} layers (for creating rules)")
print(f"Test tokens: 1-9 (for evaluation)")

# Show Token 0 journey
print("\n" + "="*70)
print("TOKEN 0 JOURNEY (32 layers)")
print("="*70)
print("\nLayer | Primary | Secondary | Gating Probs")
print("-"*50)
for d in token0:
    print(f"  {d['layer']:2d}  |    {d['experts'][0]}    |     {d['experts'][1]}     | {d['gating_probs'][0]:.2f}, {d['gating_probs'][1]:.2f}")


# ============================================================================
# LEVEL 1: (Layer, Expert) → Next Expert
# ============================================================================
print("\n" + "="*70)
print("LEVEL 1 RULES: (Layer, Expert) → Next Expert")
print("="*70)

level1_rules = {}
print("\nRule# | Layer | Expert | → Next Expert")
print("-"*45)
for i in range(len(token0) - 1):
    layer = token0[i]['layer']
    expert = token0[i]['experts'][0]
    next_expert = token0[i + 1]['experts'][0]
    level1_rules[(layer, expert)] = next_expert
    print(f"  {i+1:2d}  | L{layer:2d}  |   E{expert}   |  →  E{next_expert}")

print(f"\nTotal Level 1 rules: {len(level1_rules)}")


# ============================================================================
# LEVEL 2: (Layer, Prev, Current) → Next Expert
# ============================================================================
print("\n" + "="*70)
print("LEVEL 2 RULES: (Layer, Prev Expert, Current Expert) → Next Expert")
print("="*70)

level2_rules = {}
print("\nRule# | Layer | Prev | Curr | → Next")
print("-"*45)
for i in range(1, len(token0) - 1):
    layer = token0[i]['layer']
    prev_e = token0[i-1]['experts'][0]
    curr_e = token0[i]['experts'][0]
    next_e = token0[i+1]['experts'][0]
    level2_rules[(layer, prev_e, curr_e)] = next_e
    print(f"  {i:2d}  | L{layer:2d}  |  E{prev_e}  |  E{curr_e}  |  →  E{next_e}")

print(f"\nTotal Level 2 rules: {len(level2_rules)}")


# ============================================================================
# LEVEL 3: (Layer, E-2, E-1, E) → Next Expert
# ============================================================================
print("\n" + "="*70)
print("LEVEL 3 RULES: (Layer, E-2, E-1, Current) → Next Expert")
print("="*70)

level3_rules = {}
print("\nRule# | Layer | E-2 | E-1 | Curr | → Next")
print("-"*50)
for i in range(2, len(token0) - 1):
    layer = token0[i]['layer']
    e2 = token0[i-2]['experts'][0]
    e1 = token0[i-1]['experts'][0]
    e0 = token0[i]['experts'][0]
    next_e = token0[i+1]['experts'][0]
    level3_rules[(layer, e2, e1, e0)] = next_e
    print(f"  {i-1:2d}  | L{layer:2d}  |  E{e2}  |  E{e1}  |  E{e0}  |  →  E{next_e}")

print(f"\nTotal Level 3 rules: {len(level3_rules)}")


# ============================================================================
# LEVEL 4: (Layer, Primary, Secondary) → Next Primary
# ============================================================================
print("\n" + "="*70)
print("LEVEL 4 RULES: (Layer, Primary, Secondary) → Next Primary")
print("="*70)

level4_rules = {}
print("\nRule# | Layer | Prim | Sec | → Next Prim")
print("-"*45)
for i in range(len(token0) - 1):
    layer = token0[i]['layer']
    prim = token0[i]['experts'][0]
    sec = token0[i]['experts'][1]
    next_prim = token0[i+1]['experts'][0]
    level4_rules[(layer, prim, sec)] = next_prim
    print(f"  {i+1:2d}  | L{layer:2d}  |  E{prim}  |  E{sec}  |  →  E{next_prim}")

print(f"\nTotal Level 4 rules: {len(level4_rules)}")


# ============================================================================
# LEVEL 5: Layer-only prediction (most common expert at each layer)
# ============================================================================
print("\n" + "="*70)
print("LEVEL 5 RULES: Layer-only → Next Expert (simpler baseline)")
print("="*70)

level5_rules = {}
print("\nRule# | Layer | → Next Expert")
print("-"*35)
for i in range(len(token0) - 1):
    layer = token0[i]['layer']
    next_expert = token0[i + 1]['experts'][0]
    level5_rules[layer] = next_expert
    print(f"  {i+1:2d}  | L{layer:2d}  |  →  E{next_expert}")

print(f"\nTotal Level 5 rules: {len(level5_rules)}")


# ============================================================================
# LEVEL 6: Skip layer (Layer N → Layer N+2)
# ============================================================================
print("\n" + "="*70)
print("LEVEL 6 RULES: (Layer, Expert) → Expert at Layer+2 (skip 1)")
print("="*70)

level6_rules = {}
print("\nRule# | Layer | Expert | → Expert at L+2")
print("-"*45)
for i in range(len(token0) - 2):
    layer = token0[i]['layer']
    expert = token0[i]['experts'][0]
    future_expert = token0[i + 2]['experts'][0]
    level6_rules[(layer, expert)] = future_expert
    print(f"  {i+1:2d}  | L{layer:2d}  |   E{expert}   |  →  E{future_expert} (at L{layer+2})")

print(f"\nTotal Level 6 rules: {len(level6_rules)}")


# ============================================================================
# LEVEL 7: Gating probability based (High confidence only)
# ============================================================================
print("\n" + "="*70)
print("LEVEL 7 RULES: High-confidence transitions (gating > 0.7)")
print("="*70)

level7_rules = {}
print("\nRule# | Layer | Expert (prob) | → Next Expert")
print("-"*50)
rule_num = 0
for i in range(len(token0) - 1):
    layer = token0[i]['layer']
    expert = token0[i]['experts'][0]
    prob = token0[i]['gating_probs'][0]
    next_expert = token0[i + 1]['experts'][0]
    if prob >= 0.7:  # High confidence
        level7_rules[(layer, expert)] = next_expert
        rule_num += 1
        print(f"  {rule_num:2d}  | L{layer:2d}  |   E{expert} ({prob:.2f})  |  →  E{next_expert}")

print(f"\nTotal Level 7 rules (high-conf): {len(level7_rules)}")


# ============================================================================
# EVALUATION ON TOKENS 1-9
# ============================================================================
print("\n" + "="*70)
print("EVALUATION ON TOKENS 1-9")
print("="*70)

def evaluate(rules, get_key, start_idx=0):
    results = {}
    for t_idx, journey in test_tokens.items():
        correct = 0
        total = 0
        for i in range(start_idx, len(journey) - 1):
            key = get_key(journey, i)
            actual = journey[i + 1]['experts'][0]
            if key in rules:
                if rules[key] == actual:
                    correct += 1
                total += 1
        results[t_idx] = (correct, total)
    return results

# Level 1
print("\n[LEVEL 1] (Layer, Expert) → Next")
r1 = evaluate(level1_rules, lambda j, i: (j[i]['layer'], j[i]['experts'][0]))
for t, (c, tot) in r1.items():
    acc = c/tot*100 if tot > 0 else 0
    print(f"  Token {t}: {c}/{tot} = {acc:.1f}%")
total_c = sum(c for c, t in r1.values())
total_t = sum(t for c, t in r1.values())
print(f"  TOTAL: {total_c}/{total_t} = {total_c/total_t*100:.1f}%")

# Level 2
print("\n[LEVEL 2] (Layer, Prev, Curr) → Next")
r2 = evaluate(level2_rules, lambda j, i: (j[i]['layer'], j[i-1]['experts'][0], j[i]['experts'][0]), start_idx=1)
for t, (c, tot) in r2.items():
    acc = c/tot*100 if tot > 0 else 0
    print(f"  Token {t}: {c}/{tot} = {acc:.1f}%")
total_c = sum(c for c, t in r2.values())
total_t = sum(t for c, t in r2.values())
print(f"  TOTAL: {total_c}/{total_t} = {total_c/total_t*100:.1f}%" if total_t > 0 else "  TOTAL: 0/0")

# Level 3
print("\n[LEVEL 3] (Layer, E-2, E-1, Curr) → Next")
r3 = evaluate(level3_rules, lambda j, i: (j[i]['layer'], j[i-2]['experts'][0], j[i-1]['experts'][0], j[i]['experts'][0]), start_idx=2)
for t, (c, tot) in r3.items():
    acc = c/tot*100 if tot > 0 else 0
    print(f"  Token {t}: {c}/{tot} = {acc:.1f}%")
total_c = sum(c for c, t in r3.values())
total_t = sum(t for c, t in r3.values())
print(f"  TOTAL: {total_c}/{total_t} = {total_c/total_t*100:.1f}%" if total_t > 0 else "  TOTAL: 0/0")

# Level 4
print("\n[LEVEL 4] (Layer, Primary, Secondary) → Next Primary")
r4 = evaluate(level4_rules, lambda j, i: (j[i]['layer'], j[i]['experts'][0], j[i]['experts'][1]))
for t, (c, tot) in r4.items():
    acc = c/tot*100 if tot > 0 else 0
    print(f"  Token {t}: {c}/{tot} = {acc:.1f}%")
total_c = sum(c for c, t in r4.values())
total_t = sum(t for c, t in r4.values())
print(f"  TOTAL: {total_c}/{total_t} = {total_c/total_t*100:.1f}%" if total_t > 0 else "  TOTAL: 0/0")

# Level 5
print("\n[LEVEL 5] Layer-only → Next")
r5 = evaluate(level5_rules, lambda j, i: j[i]['layer'])
for t, (c, tot) in r5.items():
    acc = c/tot*100 if tot > 0 else 0
    print(f"  Token {t}: {c}/{tot} = {acc:.1f}%")
total_c = sum(c for c, t in r5.values())
total_t = sum(t for c, t in r5.values())
print(f"  TOTAL: {total_c}/{total_t} = {total_c/total_t*100:.1f}%")

# Level 6
print("\n[LEVEL 6] (Layer, Expert) → Expert at Layer+2")
def eval_skip(rules, skip=2):
    results = {}
    for t_idx, journey in test_tokens.items():
        correct = 0
        total = 0
        for i in range(len(journey) - skip):
            key = (journey[i]['layer'], journey[i]['experts'][0])
            actual = journey[i + skip]['experts'][0]
            if key in rules:
                if rules[key] == actual:
                    correct += 1
                total += 1
        results[t_idx] = (correct, total)
    return results

r6 = eval_skip(level6_rules, skip=2)
for t, (c, tot) in r6.items():
    acc = c/tot*100 if tot > 0 else 0
    print(f"  Token {t}: {c}/{tot} = {acc:.1f}%")
total_c = sum(c for c, t in r6.values())
total_t = sum(t for c, t in r6.values())
print(f"  TOTAL: {total_c}/{total_t} = {total_c/total_t*100:.1f}%" if total_t > 0 else "  TOTAL: 0/0")

# Level 7
print("\n[LEVEL 7] High-confidence (gating > 0.7) → Next")
r7 = evaluate(level7_rules, lambda j, i: (j[i]['layer'], j[i]['experts'][0]))
for t, (c, tot) in r7.items():
    acc = c/tot*100 if tot > 0 else 0
    print(f"  Token {t}: {c}/{tot} = {acc:.1f}%")
total_c = sum(c for c, t in r7.values())
total_t = sum(t for c, t in r7.values())
print(f"  TOTAL: {total_c}/{total_t} = {total_c/total_t*100:.1f}%" if total_t > 0 else "  TOTAL: 0/0")


# ============================================================================
# SUMMARY
# ============================================================================
print("\n" + "="*70)
print("SUMMARY")
print("="*70)

summaries = [
    ("Level 1: (Layer, Expert)", sum(c for c,t in r1.values()), sum(t for c,t in r1.values())),
    ("Level 2: (Layer, Prev, Curr)", sum(c for c,t in r2.values()), sum(t for c,t in r2.values())),
    ("Level 3: (Layer, E-2, E-1, E)", sum(c for c,t in r3.values()), sum(t for c,t in r3.values())),
    ("Level 4: (Layer, Prim, Sec)", sum(c for c,t in r4.values()), sum(t for c,t in r4.values())),
    ("Level 5: Layer-only", sum(c for c,t in r5.values()), sum(t for c,t in r5.values())),
    ("Level 6: Skip 1 layer", sum(c for c,t in r6.values()), sum(t for c,t in r6.values())),
    ("Level 7: High-conf only", sum(c for c,t in r7.values()), sum(t for c,t in r7.values())),
]

print(f"\n{'Rule Type':<30} {'Correct':<10} {'Total':<10} {'Accuracy'}")
print("-"*60)
for name, c, t in summaries:
    acc = c/t*100 if t > 0 else 0
    print(f"{name:<30} {c:<10} {t:<10} {acc:.1f}%")

print(f"\nRandom baseline: 12.5% (1/8 experts)")
print("="*70)
