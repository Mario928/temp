#!/usr/bin/env python3
"""
CORRECTED ANALYSIS - Properly handling 32 layers (removing duplicates)
"""

import json
from collections import defaultdict, Counter
import random

random.seed(42)

print("="*70)
print("CORRECTED ANALYSIS - 32 LAYERS (de-duplicated)")
print("="*70)

# Load data
routing_data = []
with open('../humaneval_2_routing.jsonl', 'r') as f:
    for line in f:
        routing_data.append(json.loads(line))

print(f"Total raw entries: {len(routing_data)}")

# De-duplicate: keep FIRST entry for each (token_idx, layer) pair
seen = set()
clean_data = []
for entry in routing_data:
    key = (entry['token_idx'], entry['layer'])
    if key not in seen:
        seen.add(key)
        clean_data.append(entry)

print(f"After de-duplication: {len(clean_data)} entries")

# Extract token journeys
def extract_journeys(data, token_list):
    journeys = {}
    for t in token_list:
        journey = [d for d in data if d['token_idx'] == t]
        journey.sort(key=lambda x: x['layer'])
        journeys[t] = journey
    return journeys

# Get all unique tokens
all_tokens = sorted(set(d['token_idx'] for d in clean_data))
print(f"Total tokens: {len(all_tokens)}")

# Check layers per token
sample_journey = [d for d in clean_data if d['token_idx'] == 0]
print(f"Layers per token (Token 0): {len(sample_journey)}")
print(f"Layer range: {min(d['layer'] for d in sample_journey)} to {max(d['layer'] for d in sample_journey)}")

# Split: 5 train, 5 test (matching original analysis)
TRAIN_TOKENS = [0, 1, 2, 3, 4]
TEST_TOKENS = [5, 6, 7, 8, 9]

train_journeys = extract_journeys(clean_data, TRAIN_TOKENS)
test_journeys = extract_journeys(clean_data, TEST_TOKENS)

print(f"\nTraining tokens: {TRAIN_TOKENS}")
print(f"Testing tokens: {TEST_TOKENS}")

# Show journeys
print("\n" + "-"*70)
print("TOKEN JOURNEYS (32 layers each):")
print("-"*70)
for t in TRAIN_TOKENS + TEST_TOKENS:
    journey = train_journeys.get(t) or test_journeys.get(t)
    experts = [j['experts'][0] for j in journey]
    label = "TRAIN" if t in TRAIN_TOKENS else "TEST"
    print(f"Token {t} [{label}]: {experts}")


# ============================================================================
# CREATE RULES FROM 5 TRAINING TOKENS
# ============================================================================
def create_rules(data_dict):
    rules = {}
    for key, predictions in data_dict.items():
        counter = Counter(predictions)
        most_common = counter.most_common()
        if len(most_common) == 1 or most_common[0][1] > most_common[1][1]:
            rule = most_common[0][0]
            conf = most_common[0][1] / len(predictions)
        else:
            rule = random.choice(predictions)
            conf = 1.0 / len(set(predictions))
        rules[key] = (rule, conf, len(predictions), dict(counter))
    return rules


print("\n" + "="*70)
print("CREATING RULES FROM 5 TRAINING TOKENS")
print("="*70)

# Level 1: (Layer, Expert) -> Next
level1_data = defaultdict(list)
for t, journey in train_journeys.items():
    for i in range(len(journey) - 1):
        layer = journey[i]['layer']
        expert = journey[i]['experts'][0]
        next_expert = journey[i + 1]['experts'][0]
        level1_data[(layer, expert)].append(next_expert)
level1_rules = create_rules(level1_data)
print(f"\nLevel 1 rules: {len(level1_rules)}")

# Level 2: (Layer, Prev, Curr) -> Next
level2_data = defaultdict(list)
for t, journey in train_journeys.items():
    for i in range(1, len(journey) - 1):
        layer = journey[i]['layer']
        prev_e = journey[i-1]['experts'][0]
        curr_e = journey[i]['experts'][0]
        next_e = journey[i+1]['experts'][0]
        level2_data[(layer, prev_e, curr_e)].append(next_e)
level2_rules = create_rules(level2_data)
print(f"Level 2 rules: {len(level2_rules)}")

# Level 3: (Layer, E-2, E-1, E) -> Next
level3_data = defaultdict(list)
for t, journey in train_journeys.items():
    for i in range(2, len(journey) - 1):
        layer = journey[i]['layer']
        e2 = journey[i-2]['experts'][0]
        e1 = journey[i-1]['experts'][0]
        e0 = journey[i]['experts'][0]
        next_e = journey[i+1]['experts'][0]
        level3_data[(layer, e2, e1, e0)].append(next_e)
level3_rules = create_rules(level3_data)
print(f"Level 3 rules: {len(level3_rules)}")

# Level 4: (Layer, Primary, Secondary) -> Next
level4_data = defaultdict(list)
for t, journey in train_journeys.items():
    for i in range(len(journey) - 1):
        layer = journey[i]['layer']
        prim = journey[i]['experts'][0]
        sec = journey[i]['experts'][1]
        next_e = journey[i+1]['experts'][0]
        level4_data[(layer, prim, sec)].append(next_e)
level4_rules = create_rules(level4_data)
print(f"Level 4 rules: {len(level4_rules)}")


# ============================================================================
# SHOW SAMPLE RULES
# ============================================================================
print("\n" + "-"*70)
print("SAMPLE LEVEL 1 RULES:")
print("-"*70)
for i, (key, val) in enumerate(list(level1_rules.items())[:15]):
    layer, expert = key
    pred, conf, n, votes = val
    print(f"  Layer {layer:2d}, Expert {expert} -> Expert {pred} (conf={conf:.0%}, n={n}, votes={votes})")


# ============================================================================
# EVALUATE ON 5 TEST TOKENS
# ============================================================================
print("\n" + "="*70)
print("EVALUATING ON 5 TEST TOKENS")
print("="*70)

def evaluate(test_journeys, rules, get_key_func, start_idx=0):
    correct = 0
    matched = 0
    total = 0
    no_rule = 0

    for t, journey in test_journeys.items():
        for i in range(start_idx, len(journey) - 1):
            key = get_key_func(journey, i)
            actual = journey[i + 1]['experts'][0]

            if key in rules:
                matched += 1
                if rules[key][0] == actual:
                    correct += 1
            else:
                no_rule += 1
            total += 1

    return correct, matched, total, no_rule

# Level 1
c, m, t, nr = evaluate(test_journeys, level1_rules,
                       lambda j, i: (j[i]['layer'], j[i]['experts'][0]))
acc1 = c/m*100 if m > 0 else 0
cov1 = m/t*100 if t > 0 else 0
print(f"\nLevel 1: {c}/{m} = {acc1:.1f}% accuracy, {cov1:.1f}% coverage ({nr} no-rule)")

# Level 2
c, m, t, nr = evaluate(test_journeys, level2_rules,
                       lambda j, i: (j[i]['layer'], j[i-1]['experts'][0], j[i]['experts'][0]),
                       start_idx=1)
acc2 = c/m*100 if m > 0 else 0
cov2 = m/t*100 if t > 0 else 0
print(f"Level 2: {c}/{m} = {acc2:.1f}% accuracy, {cov2:.1f}% coverage ({nr} no-rule)")

# Level 3
c, m, t, nr = evaluate(test_journeys, level3_rules,
                       lambda j, i: (j[i]['layer'], j[i-2]['experts'][0], j[i-1]['experts'][0], j[i]['experts'][0]),
                       start_idx=2)
acc3 = c/m*100 if m > 0 else 0
cov3 = m/t*100 if t > 0 else 0
print(f"Level 3: {c}/{m} = {acc3:.1f}% accuracy, {cov3:.1f}% coverage ({nr} no-rule)")

# Level 4
c, m, t, nr = evaluate(test_journeys, level4_rules,
                       lambda j, i: (j[i]['layer'], j[i]['experts'][0], j[i]['experts'][1]))
acc4 = c/m*100 if m > 0 else 0
cov4 = m/t*100 if t > 0 else 0
print(f"Level 4: {c}/{m} = {acc4:.1f}% accuracy, {cov4:.1f}% coverage ({nr} no-rule)")


# ============================================================================
# SUMMARY
# ============================================================================
print("\n" + "="*70)
print("SUMMARY TABLE")
print("="*70)
print(f"""
┌─────────────────────────────────────────────────────────────┐
│ Level │ Method                      │ Accuracy │ Coverage  │
├─────────────────────────────────────────────────────────────┤
│   1   │ (Layer, Expert)             │  {acc1:>5.1f}%  │  {cov1:>5.1f}%   │
│   2   │ (Layer, Prev, Curr)         │  {acc2:>5.1f}%  │  {cov2:>5.1f}%   │
│   3   │ (Layer, E-2, E-1, E)        │  {acc3:>5.1f}%  │  {cov3:>5.1f}%   │
│   4   │ (Layer, Primary, Secondary) │  {acc4:>5.1f}%  │  {cov4:>5.1f}%   │
├─────────────────────────────────────────────────────────────┤
│       │ Random Baseline (1/8)       │  12.5%   │    -      │
└─────────────────────────────────────────────────────────────┘
""")

# Compare with earlier test_all_5_levels.py results
print("COMPARISON WITH EARLIER ANALYSIS (test_all_5_levels.py):")
print("-"*70)
print("""
Earlier (Token 0 only -> Tokens 1-9, 32 layers):
  Level 1: 44.9% (69 tested)
  Level 2: 83.3% (30 tested)
  Level 3: 87.5% (24 tested)
  Level 4: 65.7% (35 tested)

This analysis (Tokens 0-4 -> Tokens 5-9, 32 layers):
  Level 1: {:.1f}%
  Level 2: {:.1f}%
  Level 3: {:.1f}%
  Level 4: {:.1f}%
""".format(acc1, acc2, acc3, acc4))

print("="*70)
print("DONE!")
print("="*70)
