#!/usr/bin/env python3
"""
Comprehensive 5-Level Expert Prediction Analysis

Strategy:
1. Create rules from FIRST 5 token journeys
2. SHOW all the rules created
3. Evaluate on REMAINING 5 token journeys
4. Test all 5 levels
5. For conflicts: majority vote, if all unique: random choice
"""

import json
from collections import defaultdict, Counter
import random

random.seed(42)  # For reproducibility

# Load all routing data
print("="*70)
print("LOADING DATA...")
print("="*70)

routing_data = []
with open('humaneval_2_routing.jsonl', 'r') as f:
    for line in f:
        routing_data.append(json.loads(line))

# Extract token journeys (first 10 tokens)
def extract_token_journeys(data, num_tokens=10):
    """Extract journeys for first N tokens"""
    journeys = defaultdict(list)

    for entry in data:
        token_idx = entry['token_idx']
        if token_idx < num_tokens:
            journeys[token_idx].append(entry)

    # Sort each journey by layer
    for token_idx in journeys:
        journeys[token_idx].sort(key=lambda x: x['layer'])

    return journeys

journeys = extract_token_journeys(routing_data, num_tokens=10)
print(f"Extracted {len(journeys)} token journeys")
print(f"Each journey has {len(journeys[0])} layers")

# Split: first 5 for training, last 5 for validation
TRAIN_TOKENS = [0, 1, 2, 3, 4]
TEST_TOKENS = [5, 6, 7, 8, 9]

print(f"\nTRAINING tokens: {TRAIN_TOKENS}")
print(f"TESTING tokens: {TEST_TOKENS}")

# Show token journeys
print("\n" + "="*70)
print("TOKEN JOURNEYS (Primary Expert per Layer)")
print("="*70)
for t in range(10):
    experts = [journeys[t][layer]['experts'][0] for layer in range(32)]
    label = "TRAIN" if t in TRAIN_TOKENS else "TEST"
    print(f"Token {t} [{label}]: {experts}")


# ============================================================================
# LEVEL 1: (Layer, Expert) -> Next Expert
# ============================================================================
print("\n" + "="*70)
print("LEVEL 1: (Layer, Expert) -> Next Expert")
print("="*70)

level1_data = defaultdict(list)  # key: (layer, expert) -> list of next experts

# Collect from training tokens
for t in TRAIN_TOKENS:
    journey = journeys[t]
    for i in range(len(journey) - 1):
        layer = journey[i]['layer']
        expert = journey[i]['experts'][0]
        next_expert = journey[i + 1]['experts'][0]
        key = (layer, expert)
        level1_data[key].append(next_expert)

# Create rules with majority vote
level1_rules = {}
print("\nLevel 1 Rules (from 5 training tokens):")
print("-" * 50)

for key in sorted(level1_data.keys()):
    predictions = level1_data[key]
    counter = Counter(predictions)
    most_common = counter.most_common()

    if len(most_common) == 1 or most_common[0][1] > most_common[1][1]:
        # Clear winner
        rule = most_common[0][0]
        confidence = most_common[0][1] / len(predictions)
        method = "majority"
    else:
        # Tie or all unique - pick random
        rule = random.choice(predictions)
        confidence = 1.0 / len(set(predictions))
        method = "random"

    level1_rules[key] = (rule, confidence, method)

    layer, expert = key
    print(f"  Layer {layer:2d}, Expert {expert} -> Expert {rule} "
          f"(conf={confidence:.0%}, {method}, votes={dict(counter)})")

print(f"\nTotal Level 1 rules: {len(level1_rules)}")


# ============================================================================
# LEVEL 2: (Layer, Prev Expert, Current Expert) -> Next Expert
# ============================================================================
print("\n" + "="*70)
print("LEVEL 2: (Layer, Prev Expert, Current Expert) -> Next Expert")
print("="*70)

level2_data = defaultdict(list)

for t in TRAIN_TOKENS:
    journey = journeys[t]
    for i in range(1, len(journey) - 1):
        layer = journey[i]['layer']
        prev_expert = journey[i - 1]['experts'][0]
        curr_expert = journey[i]['experts'][0]
        next_expert = journey[i + 1]['experts'][0]
        key = (layer, prev_expert, curr_expert)
        level2_data[key].append(next_expert)

level2_rules = {}
print("\nLevel 2 Rules (showing first 30):")
print("-" * 60)

for idx, key in enumerate(sorted(level2_data.keys())):
    predictions = level2_data[key]
    counter = Counter(predictions)
    most_common = counter.most_common()

    if len(most_common) == 1 or most_common[0][1] > most_common[1][1]:
        rule = most_common[0][0]
        confidence = most_common[0][1] / len(predictions)
        method = "majority"
    else:
        rule = random.choice(predictions)
        confidence = 1.0 / len(set(predictions))
        method = "random"

    level2_rules[key] = (rule, confidence, method)

    if idx < 30:
        layer, prev_e, curr_e = key
        print(f"  Layer {layer:2d}, E[{prev_e}]->{curr_e} -> Next: {rule} "
              f"(conf={confidence:.0%}, {method})")

print(f"\n... and {len(level2_rules) - 30} more rules")
print(f"Total Level 2 rules: {len(level2_rules)}")


# ============================================================================
# LEVEL 3: (Layer, E-2, E-1, E) -> Next Expert
# ============================================================================
print("\n" + "="*70)
print("LEVEL 3: (Layer, Expert-2, Expert-1, Expert) -> Next Expert")
print("="*70)

level3_data = defaultdict(list)

for t in TRAIN_TOKENS:
    journey = journeys[t]
    for i in range(2, len(journey) - 1):
        layer = journey[i]['layer']
        e_minus2 = journey[i - 2]['experts'][0]
        e_minus1 = journey[i - 1]['experts'][0]
        e_curr = journey[i]['experts'][0]
        next_expert = journey[i + 1]['experts'][0]
        key = (layer, e_minus2, e_minus1, e_curr)
        level3_data[key].append(next_expert)

level3_rules = {}
print("\nLevel 3 Rules (showing first 30):")
print("-" * 70)

for idx, key in enumerate(sorted(level3_data.keys())):
    predictions = level3_data[key]
    counter = Counter(predictions)
    most_common = counter.most_common()

    if len(most_common) == 1 or most_common[0][1] > most_common[1][1]:
        rule = most_common[0][0]
        confidence = most_common[0][1] / len(predictions)
        method = "majority"
    else:
        rule = random.choice(predictions)
        confidence = 1.0 / len(set(predictions))
        method = "random"

    level3_rules[key] = (rule, confidence, method)

    if idx < 30:
        layer, e2, e1, e0 = key
        print(f"  Layer {layer:2d}, [{e2}]->[{e1}]->[{e0}] -> Next: {rule} "
              f"(conf={confidence:.0%}, {method})")

print(f"\n... and {len(level3_rules) - 30} more rules")
print(f"Total Level 3 rules: {len(level3_rules)}")


# ============================================================================
# LEVEL 4: (Layer, Primary, Secondary) -> Next Primary
# ============================================================================
print("\n" + "="*70)
print("LEVEL 4: (Layer, Primary, Secondary) -> Next Primary")
print("="*70)

level4_data = defaultdict(list)

for t in TRAIN_TOKENS:
    journey = journeys[t]
    for i in range(len(journey) - 1):
        layer = journey[i]['layer']
        primary = journey[i]['experts'][0]
        secondary = journey[i]['experts'][1]
        next_primary = journey[i + 1]['experts'][0]
        key = (layer, primary, secondary)
        level4_data[key].append(next_primary)

level4_rules = {}
print("\nLevel 4 Rules (showing first 30):")
print("-" * 60)

for idx, key in enumerate(sorted(level4_data.keys())):
    predictions = level4_data[key]
    counter = Counter(predictions)
    most_common = counter.most_common()

    if len(most_common) == 1 or most_common[0][1] > most_common[1][1]:
        rule = most_common[0][0]
        confidence = most_common[0][1] / len(predictions)
        method = "majority"
    else:
        rule = random.choice(predictions)
        confidence = 1.0 / len(set(predictions))
        method = "random"

    level4_rules[key] = (rule, confidence, method)

    if idx < 30:
        layer, prim, sec = key
        print(f"  Layer {layer:2d}, Primary={prim}, Secondary={sec} -> Next Primary: {rule} "
              f"(conf={confidence:.0%}, {method})")

print(f"\n... and {len(level4_rules) - 30} more rules")
print(f"Total Level 4 rules: {len(level4_rules)}")


# ============================================================================
# LEVEL 5: Skip Layers (N -> N+2, N+3, N+4)
# ============================================================================
print("\n" + "="*70)
print("LEVEL 5: Skip Layer Prediction (Layer N -> Layer N+K)")
print("="*70)

level5_data = {2: defaultdict(list), 3: defaultdict(list), 4: defaultdict(list)}

for t in TRAIN_TOKENS:
    journey = journeys[t]
    for i in range(len(journey)):
        layer = journey[i]['layer']
        expert = journey[i]['experts'][0]

        for skip in [2, 3, 4]:
            if i + skip < len(journey):
                future_expert = journey[i + skip]['experts'][0]
                key = (layer, expert)
                level5_data[skip][key].append(future_expert)

level5_rules = {2: {}, 3: {}, 4: {}}

for skip in [2, 3, 4]:
    print(f"\nLevel 5 (Skip {skip}) Rules (showing first 15):")
    print("-" * 50)

    for idx, key in enumerate(sorted(level5_data[skip].keys())):
        predictions = level5_data[skip][key]
        counter = Counter(predictions)
        most_common = counter.most_common()

        if len(most_common) == 1 or most_common[0][1] > most_common[1][1]:
            rule = most_common[0][0]
            confidence = most_common[0][1] / len(predictions)
            method = "majority"
        else:
            rule = random.choice(predictions)
            confidence = 1.0 / len(set(predictions))
            method = "random"

        level5_rules[skip][key] = (rule, confidence, method)

        if idx < 15:
            layer, expert = key
            print(f"  Layer {layer:2d}, Expert {expert} -> Layer {layer+skip} Expert: {rule} "
                  f"(conf={confidence:.0%}, {method})")

    print(f"  Total Skip-{skip} rules: {len(level5_rules[skip])}")


# ============================================================================
# EVALUATION ON TEST TOKENS
# ============================================================================
print("\n" + "="*70)
print("EVALUATION ON TEST TOKENS (5, 6, 7, 8, 9)")
print("="*70)

def evaluate_level1(test_tokens, rules, journeys):
    correct = 0
    total = 0
    matched = 0

    for t in test_tokens:
        journey = journeys[t]
        for i in range(len(journey) - 1):
            layer = journey[i]['layer']
            expert = journey[i]['experts'][0]
            actual_next = journey[i + 1]['experts'][0]

            key = (layer, expert)
            if key in rules:
                matched += 1
                predicted, conf, method = rules[key]
                if predicted == actual_next:
                    correct += 1
            total += 1

    return correct, matched, total

def evaluate_level2(test_tokens, rules, journeys):
    correct = 0
    total = 0
    matched = 0

    for t in test_tokens:
        journey = journeys[t]
        for i in range(1, len(journey) - 1):
            layer = journey[i]['layer']
            prev_expert = journey[i - 1]['experts'][0]
            curr_expert = journey[i]['experts'][0]
            actual_next = journey[i + 1]['experts'][0]

            key = (layer, prev_expert, curr_expert)
            if key in rules:
                matched += 1
                predicted, conf, method = rules[key]
                if predicted == actual_next:
                    correct += 1
            total += 1

    return correct, matched, total

def evaluate_level3(test_tokens, rules, journeys):
    correct = 0
    total = 0
    matched = 0

    for t in test_tokens:
        journey = journeys[t]
        for i in range(2, len(journey) - 1):
            layer = journey[i]['layer']
            e_minus2 = journey[i - 2]['experts'][0]
            e_minus1 = journey[i - 1]['experts'][0]
            e_curr = journey[i]['experts'][0]
            actual_next = journey[i + 1]['experts'][0]

            key = (layer, e_minus2, e_minus1, e_curr)
            if key in rules:
                matched += 1
                predicted, conf, method = rules[key]
                if predicted == actual_next:
                    correct += 1
            total += 1

    return correct, matched, total

def evaluate_level4(test_tokens, rules, journeys):
    correct = 0
    total = 0
    matched = 0

    for t in test_tokens:
        journey = journeys[t]
        for i in range(len(journey) - 1):
            layer = journey[i]['layer']
            primary = journey[i]['experts'][0]
            secondary = journey[i]['experts'][1]
            actual_next = journey[i + 1]['experts'][0]

            key = (layer, primary, secondary)
            if key in rules:
                matched += 1
                predicted, conf, method = rules[key]
                if predicted == actual_next:
                    correct += 1
            total += 1

    return correct, matched, total

def evaluate_level5(test_tokens, rules, journeys, skip):
    correct = 0
    total = 0
    matched = 0

    for t in test_tokens:
        journey = journeys[t]
        for i in range(len(journey) - skip):
            layer = journey[i]['layer']
            expert = journey[i]['experts'][0]
            actual_future = journey[i + skip]['experts'][0]

            key = (layer, expert)
            if key in rules:
                matched += 1
                predicted, conf, method = rules[key]
                if predicted == actual_future:
                    correct += 1
            total += 1

    return correct, matched, total

# Run all evaluations
print("\n" + "-"*70)
print("RESULTS:")
print("-"*70)

results = []

# Level 1
c, m, t = evaluate_level1(TEST_TOKENS, level1_rules, journeys)
acc = c/m*100 if m > 0 else 0
cov = m/t*100 if t > 0 else 0
results.append(("Level 1: (Layer, Expert)", acc, cov, c, m, t))
print(f"\nLevel 1: (Layer, Expert) -> Next")
print(f"  Accuracy: {c}/{m} = {acc:.1f}% (when rule exists)")
print(f"  Coverage: {m}/{t} = {cov:.1f}% (rules that matched)")

# Level 2
c, m, t = evaluate_level2(TEST_TOKENS, level2_rules, journeys)
acc = c/m*100 if m > 0 else 0
cov = m/t*100 if t > 0 else 0
results.append(("Level 2: (Layer, Prev, Curr)", acc, cov, c, m, t))
print(f"\nLevel 2: (Layer, Prev Expert, Current Expert) -> Next")
print(f"  Accuracy: {c}/{m} = {acc:.1f}% (when rule exists)")
print(f"  Coverage: {m}/{t} = {cov:.1f}% (rules that matched)")

# Level 3
c, m, t = evaluate_level3(TEST_TOKENS, level3_rules, journeys)
acc = c/m*100 if m > 0 else 0
cov = m/t*100 if t > 0 else 0
results.append(("Level 3: (Layer, E-2, E-1, E)", acc, cov, c, m, t))
print(f"\nLevel 3: (Layer, E-2, E-1, E) -> Next")
print(f"  Accuracy: {c}/{m} = {acc:.1f}% (when rule exists)")
print(f"  Coverage: {m}/{t} = {cov:.1f}% (rules that matched)")

# Level 4
c, m, t = evaluate_level4(TEST_TOKENS, level4_rules, journeys)
acc = c/m*100 if m > 0 else 0
cov = m/t*100 if t > 0 else 0
results.append(("Level 4: (Layer, Primary, Secondary)", acc, cov, c, m, t))
print(f"\nLevel 4: (Layer, Primary, Secondary) -> Next Primary")
print(f"  Accuracy: {c}/{m} = {acc:.1f}% (when rule exists)")
print(f"  Coverage: {m}/{t} = {cov:.1f}% (rules that matched)")

# Level 5
for skip in [2, 3, 4]:
    c, m, t = evaluate_level5(TEST_TOKENS, level5_rules[skip], journeys, skip)
    acc = c/m*100 if m > 0 else 0
    cov = m/t*100 if t > 0 else 0
    results.append((f"Level 5: Skip {skip} layers", acc, cov, c, m, t))
    print(f"\nLevel 5: Skip {skip} layers (Layer N -> Layer N+{skip})")
    print(f"  Accuracy: {c}/{m} = {acc:.1f}% (when rule exists)")
    print(f"  Coverage: {m}/{t} = {cov:.1f}% (rules that matched)")


# ============================================================================
# SUMMARY TABLE
# ============================================================================
print("\n" + "="*70)
print("SUMMARY TABLE")
print("="*70)
print(f"\n{'Level':<35} {'Accuracy':<12} {'Coverage':<12} {'Correct/Match/Total'}")
print("-"*70)

for name, acc, cov, c, m, t in results:
    print(f"{name:<35} {acc:>6.1f}%      {cov:>6.1f}%      {c}/{m}/{t}")

# Find best
best_acc = max(results, key=lambda x: x[1])
best_cov = max(results, key=lambda x: x[2])

print("\n" + "="*70)
print("ANALYSIS")
print("="*70)
print(f"\nBest ACCURACY:  {best_acc[0]} ({best_acc[1]:.1f}%)")
print(f"Best COVERAGE:  {best_cov[0]} ({best_cov[2]:.1f}%)")

# Trade-off analysis
print("\n" + "-"*70)
print("TRADE-OFF ANALYSIS:")
print("-"*70)
print("""
- Higher context (Level 2, 3) = Higher accuracy but lower coverage
- Level 1 has highest coverage but lower accuracy
- Level 4 (using both experts) provides different information

RECOMMENDATION:
Use HIERARCHICAL approach:
1. Try Level 3 first (highest accuracy)
2. If no rule, fallback to Level 2
3. If no rule, fallback to Level 1
4. If no rule, use most common expert for that layer
""")

# ============================================================================
# HIERARCHICAL EVALUATION
# ============================================================================
print("\n" + "="*70)
print("HIERARCHICAL PREDICTION (Level 3 -> 2 -> 1 -> fallback)")
print("="*70)

# Get most common expert per layer (fallback)
layer_fallback = {}
for layer in range(32):
    experts_at_layer = []
    for t in TRAIN_TOKENS:
        experts_at_layer.append(journeys[t][layer]['experts'][0])
    counter = Counter(experts_at_layer)
    layer_fallback[layer] = counter.most_common(1)[0][0]

print("\nFallback (most common expert per layer from training):")
for layer in range(32):
    print(f"  Layer {layer}: Expert {layer_fallback[layer]}", end="")
    if (layer + 1) % 8 == 0:
        print()

# Hierarchical evaluation
hier_correct = 0
hier_total = 0
level_used = {1: 0, 2: 0, 3: 0, 'fallback': 0}

for t in TEST_TOKENS:
    journey = journeys[t]
    for i in range(2, len(journey) - 1):  # Start from 2 for Level 3
        layer = journey[i]['layer']
        actual_next = journey[i + 1]['experts'][0]

        # Try Level 3
        e_minus2 = journey[i - 2]['experts'][0]
        e_minus1 = journey[i - 1]['experts'][0]
        e_curr = journey[i]['experts'][0]
        key3 = (layer, e_minus2, e_minus1, e_curr)

        if key3 in level3_rules:
            predicted, _, _ = level3_rules[key3]
            level_used[3] += 1
        else:
            # Try Level 2
            key2 = (layer, e_minus1, e_curr)
            if key2 in level2_rules:
                predicted, _, _ = level2_rules[key2]
                level_used[2] += 1
            else:
                # Try Level 1
                key1 = (layer, e_curr)
                if key1 in level1_rules:
                    predicted, _, _ = level1_rules[key1]
                    level_used[1] += 1
                else:
                    # Fallback
                    predicted = layer_fallback[layer + 1] if layer + 1 < 32 else layer_fallback[31]
                    level_used['fallback'] += 1

        if predicted == actual_next:
            hier_correct += 1
        hier_total += 1

print(f"\n\nHierarchical Results:")
print(f"  Accuracy: {hier_correct}/{hier_total} = {hier_correct/hier_total*100:.1f}%")
print(f"  Level usage: {dict(level_used)}")

print("\n" + "="*70)
print("DONE!")
print("="*70)
