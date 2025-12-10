#!/usr/bin/env python3
"""
Step 3: Evaluate rules on 5 test token journeys
Reads: token_journey_5.json through token_journey_9.json
Reads: rules_level1.json through rules_level5.json
"""

import json

print("="*60)
print("STEP 3: Evaluating Rules on 5 Test Tokens")
print("="*60)

# Load test token journeys
TEST_TOKENS = [5, 6, 7, 8, 9]
journeys = {}

print("\nLoading test token journeys...")
for t in TEST_TOKENS:
    with open(f'token_journey_{t}.json', 'r') as f:
        journeys[t] = json.load(f)
    print(f"  Token {t}: Primary experts = {journeys[t]['primary_experts']}")

# Load all rules
print("\nLoading rules...")
with open('rules_level1.json', 'r') as f:
    level1_rules = json.load(f)
print(f"  Level 1: {len(level1_rules)} rules")

with open('rules_level2.json', 'r') as f:
    level2_rules = json.load(f)
print(f"  Level 2: {len(level2_rules)} rules")

with open('rules_level3.json', 'r') as f:
    level3_rules = json.load(f)
print(f"  Level 3: {len(level3_rules)} rules")

with open('rules_level4.json', 'r') as f:
    level4_rules = json.load(f)
print(f"  Level 4: {len(level4_rules)} rules")

with open('rules_level5.json', 'r') as f:
    level5_rules = json.load(f)
print(f"  Level 5: Skip-2={len(level5_rules['skip_2'])}, Skip-3={len(level5_rules['skip_3'])}, Skip-4={len(level5_rules['skip_4'])}")


# ============================================================================
# EVALUATION FUNCTIONS
# ============================================================================

def evaluate_level1(test_tokens, rules, journeys):
    correct = 0
    matched = 0
    total = 0

    for t in test_tokens:
        primary = journeys[t]['primary_experts']
        for i in range(len(primary) - 1):
            layer = i
            expert = primary[i]
            actual_next = primary[i + 1]
            key = str((layer, expert))

            if key in rules:
                matched += 1
                if rules[key]['prediction'] == actual_next:
                    correct += 1
            total += 1

    return correct, matched, total


def evaluate_level2(test_tokens, rules, journeys):
    correct = 0
    matched = 0
    total = 0

    for t in test_tokens:
        primary = journeys[t]['primary_experts']
        for i in range(1, len(primary) - 1):
            layer = i
            prev_expert = primary[i - 1]
            curr_expert = primary[i]
            actual_next = primary[i + 1]
            key = str((layer, prev_expert, curr_expert))

            if key in rules:
                matched += 1
                if rules[key]['prediction'] == actual_next:
                    correct += 1
            total += 1

    return correct, matched, total


def evaluate_level3(test_tokens, rules, journeys):
    correct = 0
    matched = 0
    total = 0

    for t in test_tokens:
        primary = journeys[t]['primary_experts']
        for i in range(2, len(primary) - 1):
            layer = i
            e2 = primary[i - 2]
            e1 = primary[i - 1]
            e0 = primary[i]
            actual_next = primary[i + 1]
            key = str((layer, e2, e1, e0))

            if key in rules:
                matched += 1
                if rules[key]['prediction'] == actual_next:
                    correct += 1
            total += 1

    return correct, matched, total


def evaluate_level4(test_tokens, rules, journeys):
    correct = 0
    matched = 0
    total = 0

    for t in test_tokens:
        primary = journeys[t]['primary_experts']
        secondary = journeys[t]['secondary_experts']
        for i in range(len(primary) - 1):
            layer = i
            prim = primary[i]
            sec = secondary[i]
            actual_next = primary[i + 1]
            key = str((layer, prim, sec))

            if key in rules:
                matched += 1
                if rules[key]['prediction'] == actual_next:
                    correct += 1
            total += 1

    return correct, matched, total


def evaluate_level5(test_tokens, rules, journeys, skip):
    correct = 0
    matched = 0
    total = 0

    for t in test_tokens:
        primary = journeys[t]['primary_experts']
        for i in range(len(primary) - skip):
            layer = i
            expert = primary[i]
            actual_future = primary[i + skip]
            key = str((layer, expert))

            if key in rules:
                matched += 1
                if rules[key]['prediction'] == actual_future:
                    correct += 1
            total += 1

    return correct, matched, total


# ============================================================================
# RUN EVALUATIONS
# ============================================================================
print("\n" + "="*60)
print("EVALUATION RESULTS")
print("="*60)

results = []

# Level 1
c, m, t = evaluate_level1(TEST_TOKENS, level1_rules, journeys)
acc = c/m*100 if m > 0 else 0
cov = m/t*100 if t > 0 else 0
results.append(("Level 1: (Layer, Expert)", acc, cov, c, m, t))
print(f"\n[LEVEL 1] (Layer, Expert) -> Next Expert")
print(f"  Correct: {c}/{m} = {acc:.1f}% accuracy")
print(f"  Coverage: {m}/{t} = {cov:.1f}%")

# Level 2
c, m, t = evaluate_level2(TEST_TOKENS, level2_rules, journeys)
acc = c/m*100 if m > 0 else 0
cov = m/t*100 if t > 0 else 0
results.append(("Level 2: (Layer, Prev, Curr)", acc, cov, c, m, t))
print(f"\n[LEVEL 2] (Layer, Prev, Curr) -> Next Expert")
print(f"  Correct: {c}/{m} = {acc:.1f}% accuracy")
print(f"  Coverage: {m}/{t} = {cov:.1f}%")

# Level 3
c, m, t = evaluate_level3(TEST_TOKENS, level3_rules, journeys)
acc = c/m*100 if m > 0 else 0
cov = m/t*100 if t > 0 else 0
results.append(("Level 3: (Layer, E-2, E-1, E)", acc, cov, c, m, t))
print(f"\n[LEVEL 3] (Layer, E-2, E-1, E) -> Next Expert")
print(f"  Correct: {c}/{m} = {acc:.1f}% accuracy")
print(f"  Coverage: {m}/{t} = {cov:.1f}%")

# Level 4
c, m, t = evaluate_level4(TEST_TOKENS, level4_rules, journeys)
acc = c/m*100 if m > 0 else 0
cov = m/t*100 if t > 0 else 0
results.append(("Level 4: (Layer, Prim, Sec)", acc, cov, c, m, t))
print(f"\n[LEVEL 4] (Layer, Primary, Secondary) -> Next Primary")
print(f"  Correct: {c}/{m} = {acc:.1f}% accuracy")
print(f"  Coverage: {m}/{t} = {cov:.1f}%")

# Level 5
for skip in [2, 3, 4]:
    c, m, t = evaluate_level5(TEST_TOKENS, level5_rules[f'skip_{skip}'], journeys, skip)
    acc = c/m*100 if m > 0 else 0
    cov = m/t*100 if t > 0 else 0
    results.append((f"Level 5: Skip-{skip}", acc, cov, c, m, t))
    print(f"\n[LEVEL 5] Skip {skip} layers (Layer N -> Layer N+{skip})")
    print(f"  Correct: {c}/{m} = {acc:.1f}% accuracy")
    print(f"  Coverage: {m}/{t} = {cov:.1f}%")


# ============================================================================
# SUMMARY TABLE
# ============================================================================
print("\n" + "="*60)
print("SUMMARY TABLE")
print("="*60)
print(f"\n{'Level':<30} {'Accuracy':<12} {'Coverage':<12} {'Correct/Match/Total'}")
print("-"*70)

for name, acc, cov, c, m, t in results:
    print(f"{name:<30} {acc:>6.1f}%      {cov:>6.1f}%      {c}/{m}/{t}")

# Find best
best_acc = max(results, key=lambda x: x[1])
best_cov = max(results, key=lambda x: x[2])

print("\n" + "-"*70)
print(f"BEST ACCURACY:  {best_acc[0]} ({best_acc[1]:.1f}%)")
print(f"BEST COVERAGE:  {best_cov[0]} ({best_cov[2]:.1f}%)")

# Save results
eval_results = {
    "test_tokens": TEST_TOKENS,
    "results": [
        {"level": r[0], "accuracy": r[1], "coverage": r[2],
         "correct": r[3], "matched": r[4], "total": r[5]}
        for r in results
    ]
}

with open('evaluation_results.json', 'w') as f:
    json.dump(eval_results, f, indent=2)

print("\n" + "="*60)
print("Results saved to: evaluation_results.json")
print("="*60)
