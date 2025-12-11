#!/usr/bin/env python3
"""
Step 4: Scale up - Use 50 training tokens, evaluate on 50 test tokens
"""

import json
from collections import defaultdict, Counter
import random

random.seed(42)

print("="*60)
print("SCALED UP ANALYSIS: 50 Train / 50 Test Tokens")
print("="*60)

# Load all routing data
routing_data = []
with open('../humaneval_2_routing.jsonl', 'r') as f:
    for line in f:
        routing_data.append(json.loads(line))

# Find how many unique tokens we have
all_tokens = set(entry['token_idx'] for entry in routing_data)
print(f"Total unique tokens in dataset: {len(all_tokens)}")

# Extract token journeys
def extract_journeys(data, token_list):
    journeys = defaultdict(list)
    for entry in data:
        if entry['token_idx'] in token_list:
            journeys[entry['token_idx']].append(entry)
    for t in journeys:
        journeys[t].sort(key=lambda x: x['layer'])
    return journeys

# Split tokens
max_token = max(all_tokens)
TRAIN_TOKENS = list(range(0, min(50, max_token//2)))
TEST_TOKENS = list(range(50, min(100, max_token)))

if len(TEST_TOKENS) < 10:
    # Not enough tokens, use smaller split
    TRAIN_TOKENS = list(range(0, len(all_tokens)//2))
    TEST_TOKENS = list(range(len(all_tokens)//2, len(all_tokens)))

print(f"Training tokens: {len(TRAIN_TOKENS)} (tokens {TRAIN_TOKENS[0]}-{TRAIN_TOKENS[-1]})")
print(f"Testing tokens: {len(TEST_TOKENS)} (tokens {TEST_TOKENS[0]}-{TEST_TOKENS[-1]})")

train_journeys = extract_journeys(routing_data, TRAIN_TOKENS)
test_journeys = extract_journeys(routing_data, TEST_TOKENS)

print(f"Loaded {len(train_journeys)} training journeys")
print(f"Loaded {len(test_journeys)} test journeys")


def create_rules(data_dict):
    """Create rules with majority vote"""
    rules = {}
    for key, predictions in data_dict.items():
        counter = Counter(predictions)
        most_common = counter.most_common()
        if len(most_common) == 1 or most_common[0][1] > most_common[1][1]:
            rule = most_common[0][0]
            confidence = most_common[0][1] / len(predictions)
        else:
            rule = random.choice(predictions)
            confidence = 1.0 / len(set(predictions))
        rules[key] = (rule, confidence, len(predictions))
    return rules


# ============================================================================
# BUILD RULES FROM TRAINING DATA
# ============================================================================
print("\n" + "-"*60)
print("Building rules from training data...")
print("-"*60)

# Level 1
level1_data = defaultdict(list)
for t, journey in train_journeys.items():
    for i in range(len(journey) - 1):
        layer = journey[i]['layer']
        expert = journey[i]['experts'][0]
        next_expert = journey[i + 1]['experts'][0]
        level1_data[(layer, expert)].append(next_expert)
level1_rules = create_rules(level1_data)
print(f"Level 1 rules: {len(level1_rules)}")

# Level 2
level2_data = defaultdict(list)
for t, journey in train_journeys.items():
    for i in range(1, len(journey) - 1):
        layer = journey[i]['layer']
        prev = journey[i-1]['experts'][0]
        curr = journey[i]['experts'][0]
        next_e = journey[i + 1]['experts'][0]
        level2_data[(layer, prev, curr)].append(next_e)
level2_rules = create_rules(level2_data)
print(f"Level 2 rules: {len(level2_rules)}")

# Level 3
level3_data = defaultdict(list)
for t, journey in train_journeys.items():
    for i in range(2, len(journey) - 1):
        layer = journey[i]['layer']
        e2 = journey[i-2]['experts'][0]
        e1 = journey[i-1]['experts'][0]
        e0 = journey[i]['experts'][0]
        next_e = journey[i + 1]['experts'][0]
        level3_data[(layer, e2, e1, e0)].append(next_e)
level3_rules = create_rules(level3_data)
print(f"Level 3 rules: {len(level3_rules)}")

# Level 4
level4_data = defaultdict(list)
for t, journey in train_journeys.items():
    for i in range(len(journey) - 1):
        layer = journey[i]['layer']
        prim = journey[i]['experts'][0]
        sec = journey[i]['experts'][1]
        next_e = journey[i + 1]['experts'][0]
        level4_data[(layer, prim, sec)].append(next_e)
level4_rules = create_rules(level4_data)
print(f"Level 4 rules: {len(level4_rules)}")


# ============================================================================
# EVALUATE ON TEST DATA
# ============================================================================
print("\n" + "-"*60)
print("Evaluating on test data...")
print("-"*60)

def evaluate(test_journeys, rules, get_key_func, start_idx=0):
    correct = 0
    matched = 0
    total = 0

    for t, journey in test_journeys.items():
        for i in range(start_idx, len(journey) - 1):
            key = get_key_func(journey, i)
            actual = journey[i + 1]['experts'][0]

            if key in rules:
                matched += 1
                if rules[key][0] == actual:
                    correct += 1
            total += 1

    return correct, matched, total

# Level 1
c, m, t = evaluate(test_journeys, level1_rules,
                   lambda j, i: (j[i]['layer'], j[i]['experts'][0]))
acc1 = c/m*100 if m > 0 else 0
cov1 = m/t*100 if t > 0 else 0
print(f"\nLevel 1: Acc={acc1:.1f}% ({c}/{m}), Coverage={cov1:.1f}% ({m}/{t})")

# Level 2
c, m, t = evaluate(test_journeys, level2_rules,
                   lambda j, i: (j[i]['layer'], j[i-1]['experts'][0], j[i]['experts'][0]),
                   start_idx=1)
acc2 = c/m*100 if m > 0 else 0
cov2 = m/t*100 if t > 0 else 0
print(f"Level 2: Acc={acc2:.1f}% ({c}/{m}), Coverage={cov2:.1f}% ({m}/{t})")

# Level 3
c, m, t = evaluate(test_journeys, level3_rules,
                   lambda j, i: (j[i]['layer'], j[i-2]['experts'][0], j[i-1]['experts'][0], j[i]['experts'][0]),
                   start_idx=2)
acc3 = c/m*100 if m > 0 else 0
cov3 = m/t*100 if t > 0 else 0
print(f"Level 3: Acc={acc3:.1f}% ({c}/{m}), Coverage={cov3:.1f}% ({m}/{t})")

# Level 4
c, m, t = evaluate(test_journeys, level4_rules,
                   lambda j, i: (j[i]['layer'], j[i]['experts'][0], j[i]['experts'][1]))
acc4 = c/m*100 if m > 0 else 0
cov4 = m/t*100 if t > 0 else 0
print(f"Level 4: Acc={acc4:.1f}% ({c}/{m}), Coverage={cov4:.1f}% ({m}/{t})")


# ============================================================================
# ANALYZE HIGH-CONFIDENCE RULES
# ============================================================================
print("\n" + "-"*60)
print("Analyzing high-confidence rules (samples >= 5, confidence >= 60%)...")
print("-"*60)

high_conf_rules = {k: v for k, v in level1_rules.items() if v[1] >= 0.6 and v[2] >= 5}
print(f"\nLevel 1 high-confidence rules: {len(high_conf_rules)}")

# Test high-confidence rules only
def evaluate_high_conf(test_journeys, rules, high_conf_keys, get_key_func, start_idx=0):
    correct = 0
    matched = 0

    for t, journey in test_journeys.items():
        for i in range(start_idx, len(journey) - 1):
            key = get_key_func(journey, i)
            actual = journey[i + 1]['experts'][0]

            if key in high_conf_keys:
                matched += 1
                if rules[key][0] == actual:
                    correct += 1

    return correct, matched

c, m = evaluate_high_conf(test_journeys, level1_rules, high_conf_rules,
                          lambda j, i: (j[i]['layer'], j[i]['experts'][0]))
if m > 0:
    print(f"High-conf Level 1: Acc={c/m*100:.1f}% ({c}/{m})")

# Show some high-confidence rules
print("\nSample high-confidence rules:")
for i, (key, val) in enumerate(list(high_conf_rules.items())[:15]):
    layer, expert = key
    pred, conf, samples = val
    print(f"  Layer {layer:2d}, Expert {expert} -> Expert {pred} (conf={conf:.0%}, n={samples})")


# ============================================================================
# RANDOM BASELINE
# ============================================================================
print("\n" + "-"*60)
print("Random baseline (pick random expert 0-7):")
print("-"*60)
print(f"Expected accuracy: 12.5% (1/8 experts)")


# ============================================================================
# SUMMARY
# ============================================================================
print("\n" + "="*60)
print("SUMMARY")
print("="*60)
print(f"""
Training tokens: {len(TRAIN_TOKENS)}
Testing tokens: {len(TEST_TOKENS)}

Results:
  Level 1: {acc1:.1f}% accuracy, {cov1:.1f}% coverage
  Level 2: {acc2:.1f}% accuracy, {cov2:.1f}% coverage
  Level 3: {acc3:.1f}% accuracy, {cov3:.1f}% coverage
  Level 4: {acc4:.1f}% accuracy, {cov4:.1f}% coverage

Random baseline: 12.5%

Key Insight:
  - More training data improves COVERAGE (more patterns seen)
  - But ACCURACY still limited by content-dependence
  - High-confidence rules perform better
""")
