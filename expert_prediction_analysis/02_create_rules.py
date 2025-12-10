#!/usr/bin/env python3
"""
Step 2: Create prediction rules from 5 training token journeys
Reads: token_journey_0.json through token_journey_4.json
Creates: rules_level1.json through rules_level5.json
"""

import json
from collections import defaultdict, Counter
import random

random.seed(42)

print("="*60)
print("STEP 2: Creating Rules from 5 Training Tokens")
print("="*60)

# Load 5 training token journeys
TRAIN_TOKENS = [0, 1, 2, 3, 4]
journeys = {}

print("\nLoading training token journeys...")
for t in TRAIN_TOKENS:
    with open(f'token_journey_{t}.json', 'r') as f:
        journeys[t] = json.load(f)
    print(f"  Token {t}: {journeys[t]['num_layers']} layers")
    print(f"    Primary: {journeys[t]['primary_experts']}")


def create_rules_with_vote(data_dict):
    """Create rules using majority vote, random if tie"""
    rules = {}
    for key, predictions in data_dict.items():
        counter = Counter(predictions)
        most_common = counter.most_common()

        if len(most_common) == 1 or most_common[0][1] > most_common[1][1]:
            # Clear winner
            rule = most_common[0][0]
            confidence = most_common[0][1] / len(predictions)
            method = "majority"
        else:
            # Tie - pick random
            rule = random.choice(predictions)
            confidence = 1.0 / len(set(predictions))
            method = "random"

        # Convert key to string for JSON
        key_str = str(key)
        rules[key_str] = {
            "prediction": rule,
            "confidence": confidence,
            "method": method,
            "votes": dict(counter),
            "total_samples": len(predictions)
        }
    return rules


# ============================================================================
# LEVEL 1: (Layer, Expert) -> Next Expert
# ============================================================================
print("\n" + "-"*60)
print("Creating LEVEL 1 rules: (Layer, Expert) -> Next Expert")
print("-"*60)

level1_data = defaultdict(list)

for t in TRAIN_TOKENS:
    primary = journeys[t]['primary_experts']
    for i in range(len(primary) - 1):
        layer = i
        expert = primary[i]
        next_expert = primary[i + 1]
        key = (layer, expert)
        level1_data[key].append(next_expert)

level1_rules = create_rules_with_vote(level1_data)

print(f"Total Level 1 rules: {len(level1_rules)}")
print("\nSample rules:")
for i, (key, rule) in enumerate(list(level1_rules.items())[:10]):
    print(f"  {key} -> {rule['prediction']} (conf={rule['confidence']:.0%}, {rule['method']}, votes={rule['votes']})")

with open('rules_level1.json', 'w') as f:
    json.dump(level1_rules, f, indent=2)
print("\nSaved to: rules_level1.json")


# ============================================================================
# LEVEL 2: (Layer, Prev, Curr) -> Next Expert
# ============================================================================
print("\n" + "-"*60)
print("Creating LEVEL 2 rules: (Layer, Prev, Curr) -> Next Expert")
print("-"*60)

level2_data = defaultdict(list)

for t in TRAIN_TOKENS:
    primary = journeys[t]['primary_experts']
    for i in range(1, len(primary) - 1):
        layer = i
        prev_expert = primary[i - 1]
        curr_expert = primary[i]
        next_expert = primary[i + 1]
        key = (layer, prev_expert, curr_expert)
        level2_data[key].append(next_expert)

level2_rules = create_rules_with_vote(level2_data)

print(f"Total Level 2 rules: {len(level2_rules)}")
print("\nSample rules:")
for i, (key, rule) in enumerate(list(level2_rules.items())[:10]):
    print(f"  {key} -> {rule['prediction']} (conf={rule['confidence']:.0%}, {rule['method']})")

with open('rules_level2.json', 'w') as f:
    json.dump(level2_rules, f, indent=2)
print("\nSaved to: rules_level2.json")


# ============================================================================
# LEVEL 3: (Layer, E-2, E-1, E) -> Next Expert
# ============================================================================
print("\n" + "-"*60)
print("Creating LEVEL 3 rules: (Layer, E-2, E-1, E) -> Next Expert")
print("-"*60)

level3_data = defaultdict(list)

for t in TRAIN_TOKENS:
    primary = journeys[t]['primary_experts']
    for i in range(2, len(primary) - 1):
        layer = i
        e_minus2 = primary[i - 2]
        e_minus1 = primary[i - 1]
        e_curr = primary[i]
        next_expert = primary[i + 1]
        key = (layer, e_minus2, e_minus1, e_curr)
        level3_data[key].append(next_expert)

level3_rules = create_rules_with_vote(level3_data)

print(f"Total Level 3 rules: {len(level3_rules)}")
print("\nSample rules:")
for i, (key, rule) in enumerate(list(level3_rules.items())[:10]):
    print(f"  {key} -> {rule['prediction']} (conf={rule['confidence']:.0%}, {rule['method']})")

with open('rules_level3.json', 'w') as f:
    json.dump(level3_rules, f, indent=2)
print("\nSaved to: rules_level3.json")


# ============================================================================
# LEVEL 4: (Layer, Primary, Secondary) -> Next Primary
# ============================================================================
print("\n" + "-"*60)
print("Creating LEVEL 4 rules: (Layer, Primary, Secondary) -> Next Primary")
print("-"*60)

level4_data = defaultdict(list)

for t in TRAIN_TOKENS:
    primary = journeys[t]['primary_experts']
    secondary = journeys[t]['secondary_experts']
    for i in range(len(primary) - 1):
        layer = i
        prim = primary[i]
        sec = secondary[i]
        next_prim = primary[i + 1]
        key = (layer, prim, sec)
        level4_data[key].append(next_prim)

level4_rules = create_rules_with_vote(level4_data)

print(f"Total Level 4 rules: {len(level4_rules)}")
print("\nSample rules:")
for i, (key, rule) in enumerate(list(level4_rules.items())[:10]):
    print(f"  {key} -> {rule['prediction']} (conf={rule['confidence']:.0%}, {rule['method']})")

with open('rules_level4.json', 'w') as f:
    json.dump(level4_rules, f, indent=2)
print("\nSaved to: rules_level4.json")


# ============================================================================
# LEVEL 5: Skip Layers (N -> N+2, N+3, N+4)
# ============================================================================
print("\n" + "-"*60)
print("Creating LEVEL 5 rules: Skip Layer Prediction")
print("-"*60)

level5_rules = {}

for skip in [2, 3, 4]:
    level5_data = defaultdict(list)

    for t in TRAIN_TOKENS:
        primary = journeys[t]['primary_experts']
        for i in range(len(primary) - skip):
            layer = i
            expert = primary[i]
            future_expert = primary[i + skip]
            key = (layer, expert)
            level5_data[key].append(future_expert)

    level5_rules[f"skip_{skip}"] = create_rules_with_vote(level5_data)
    print(f"  Skip-{skip} rules: {len(level5_rules[f'skip_{skip}'])}")

with open('rules_level5.json', 'w') as f:
    json.dump(level5_rules, f, indent=2)
print("\nSaved to: rules_level5.json")


# ============================================================================
# SUMMARY
# ============================================================================
print("\n" + "="*60)
print("SUMMARY: Rules Created")
print("="*60)

summary = {
    "training_tokens": TRAIN_TOKENS,
    "rules_count": {
        "level1": len(level1_rules),
        "level2": len(level2_rules),
        "level3": len(level3_rules),
        "level4": len(level4_rules),
        "level5_skip2": len(level5_rules["skip_2"]),
        "level5_skip3": len(level5_rules["skip_3"]),
        "level5_skip4": len(level5_rules["skip_4"]),
    }
}

print(f"\nLevel 1 (Layer, Expert):           {summary['rules_count']['level1']} rules")
print(f"Level 2 (Layer, Prev, Curr):       {summary['rules_count']['level2']} rules")
print(f"Level 3 (Layer, E-2, E-1, E):      {summary['rules_count']['level3']} rules")
print(f"Level 4 (Layer, Prim, Sec):        {summary['rules_count']['level4']} rules")
print(f"Level 5 Skip-2:                    {summary['rules_count']['level5_skip2']} rules")
print(f"Level 5 Skip-3:                    {summary['rules_count']['level5_skip3']} rules")
print(f"Level 5 Skip-4:                    {summary['rules_count']['level5_skip4']} rules")

with open('rules_summary.json', 'w') as f:
    json.dump(summary, f, indent=2)

print("\n" + "="*60)
print("DONE! All rules saved to JSON files")
print("="*60)
