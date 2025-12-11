#!/usr/bin/env python3
"""
RE-ANALYSIS: Find ACTUALLY USEFUL patterns
Not "UP/DOWN" but SPECIFIC expert predictions

Question: Can we predict WHICH expert at Layer N+1?
"""

import json

print("=" * 70)
print("RE-ANALYSIS: Finding ACTUALLY USEFUL Patterns")
print("(Not UP/DOWN, but SPECIFIC expert predictions)")
print("=" * 70)

# Load Token 0
journey = []
with open('/home/user/temp/token_journey_0_p0_t0.jsonl', 'r') as f:
    for line in f:
        if line.strip():
            journey.append(json.loads(line))
journey.sort(key=lambda x: x['layer'])

print("\n📊 TOKEN 0 JOURNEY:")
print("-" * 50)

expert_sequence = []
for r in journey:
    expert_sequence.append(r['experts'][0])

print(f"Expert sequence: {expert_sequence}")
print(f"Total layers: {len(expert_sequence)}")

# USEFUL PATTERN 1: Specific Expert Transitions
print("\n" + "=" * 70)
print("USEFUL PATTERN 1: Specific Expert → Expert Transitions")
print("=" * 70)

print("\nLooking at EACH transition:")
transitions = {}
for i in range(len(expert_sequence) - 1):
    from_e = expert_sequence[i]
    to_e = expert_sequence[i + 1]
    layer = i

    key = f"Expert {from_e}"
    if key not in transitions:
        transitions[key] = []
    transitions[key].append(to_e)

    print(f"  Layer {i}→{i+1}: Expert {from_e} → Expert {to_e}")

print("\n📌 PATTERN: What does each expert transition to?")
for from_expert, to_list in sorted(transitions.items()):
    print(f"  {from_expert} goes to: {to_list}")
    # Find most common
    if to_list:
        from collections import Counter
        counts = Counter(to_list)
        most_common = counts.most_common(1)[0]
        print(f"    → Most common: Expert {most_common[0]} ({most_common[1]}/{len(to_list)} = {most_common[1]/len(to_list)*100:.0f}%)")

# USEFUL PATTERN 2: Layer-Specific Prediction
print("\n" + "=" * 70)
print("USEFUL PATTERN 2: Layer-Specific Expert Prediction")
print("=" * 70)

print("\nFor EACH layer, which expert is chosen?")
layer_expert = {}
for i, e in enumerate(expert_sequence):
    layer_expert[i] = e
    print(f"  Layer {i}: Expert {e}")

print("\n📌 Can we memorize: 'Layer X always uses Expert Y'?")
print("   (This would be the simplest prediction)")

# USEFUL PATTERN 3: Secondary Expert becomes Primary?
print("\n" + "=" * 70)
print("USEFUL PATTERN 3: Does Secondary become Primary?")
print("=" * 70)

secondary_to_primary = 0
total = 0

for i in range(len(journey) - 1):
    secondary_now = journey[i]['experts'][1]
    primary_next = journey[i + 1]['experts'][0]

    total += 1
    if secondary_now == primary_next:
        secondary_to_primary += 1
        print(f"  Layer {i}→{i+1}: Secondary {secondary_now} became Primary {primary_next} ✅")

rate = secondary_to_primary / total * 100
print(f"\n📌 RESULT: Secondary→Primary happens {secondary_to_primary}/{total} = {rate:.1f}%")

# USEFUL PATTERN 4: Pair of experts predicts next primary?
print("\n" + "=" * 70)
print("USEFUL PATTERN 4: Current (Primary,Secondary) → Next Primary")
print("=" * 70)

pair_predictions = {}
for i in range(len(journey) - 1):
    pair = (journey[i]['experts'][0], journey[i]['experts'][1])
    next_primary = journey[i + 1]['experts'][0]

    if pair not in pair_predictions:
        pair_predictions[pair] = []
    pair_predictions[pair].append(next_primary)

    print(f"  Layer {i}: Pair {pair} → Next Primary {next_primary}")

print("\n📌 PATTERN: Each pair predicts which next primary?")
for pair, nexts in pair_predictions.items():
    if len(nexts) >= 1:
        from collections import Counter
        counts = Counter(nexts)
        most_common = counts.most_common(1)[0]
        print(f"  Pair {pair} → usually Expert {most_common[0]} ({most_common[1]}/{len(nexts)})")

# SUMMARY
print("\n" + "=" * 70)
print("HONEST ASSESSMENT: What is ACTUALLY Useful?")
print("=" * 70)

print("""
❌ Anti-Momentum (UP/DOWN): USELESS for prediction
   - Knowing "UP" doesn't tell which expert (could be any of 7)

Potentially Useful Patterns (need to validate):

1. SPECIFIC TRANSITION: "Expert 3 → Expert X"
   - But need more data to see if consistent

2. LAYER-SPECIFIC: "Layer 5 always uses Expert 6"
   - Need to check across multiple tokens

3. SECONDARY→PRIMARY: Only ~10% of time
   - Not strong enough

4. PAIR PREDICTION: "(Primary, Secondary) → Next Primary"
   - Most specific, but need to validate
""")

print("\n🔴 CONCLUSION FROM TOKEN 0 ALONE:")
print("   With only 1 token (32 transitions), we don't have")
print("   enough data to find reliable SPECIFIC predictions.")
print("")
print("   We need to aggregate across multiple tokens to see")
print("   if there are consistent patterns like:")
print("   'Expert 3 at Layer 5 → usually Expert 7 at Layer 6'")
