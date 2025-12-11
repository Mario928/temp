#!/usr/bin/env python3
"""
FINDING ACTUALLY USEFUL PATTERNS
Goal: "At Layer X, if Expert A, then Layer X+1 will be Expert B"
This is SPECIFIC and ACTIONABLE for pre-loading!
"""

import json
from collections import defaultdict, Counter

print("=" * 70)
print("FINDING ACTUALLY USEFUL PATTERNS FOR PRE-LOADING")
print("=" * 70)

# Load ALL 10 token journeys
token_files = [
    'token_journey_0_p0_t0.jsonl',
    'token_journey_1_p0_t1.jsonl',
    'token_journey_2_p0_t2.jsonl',
    'token_journey_3_p0_t3.jsonl',
    'token_journey_4_p0_t4.jsonl',
    'token_journey_5_p1_t0.jsonl',
    'token_journey_6_p1_t1.jsonl',
    'token_journey_7_p1_t2.jsonl',
    'token_journey_8_p1_t3.jsonl',
    'token_journey_9_p1_t4.jsonl',
]

all_journeys = []
for filename in token_files:
    journey = []
    with open(f'/home/user/temp/{filename}', 'r') as f:
        for line in f:
            if line.strip():
                journey.append(json.loads(line))
    journey.sort(key=lambda x: x['layer'])
    all_journeys.append(journey)

print(f"\nLoaded {len(all_journeys)} token journeys")

# ============================================================
# USEFUL PATTERN 1: Layer-Specific Most Common Expert
# "At Layer X, most tokens use Expert Y"
# ============================================================
print("\n" + "=" * 70)
print("PATTERN 1: Most Common Expert per Layer")
print("(Can we just always pre-load the most common expert?)")
print("=" * 70)

layer_experts = defaultdict(list)
for journey in all_journeys:
    for record in journey:
        layer_experts[record['layer']].append(record['experts'][0])

print(f"\n{'Layer':<8} {'Most Common':<15} {'Frequency':<12} {'Useful?':<10}")
print("-" * 50)

layer_predictions = {}
useful_layers = 0

for layer in range(32):
    experts = layer_experts[layer]
    counts = Counter(experts)
    most_common_expert, count = counts.most_common(1)[0]
    frequency = count / len(experts) * 100

    layer_predictions[layer] = most_common_expert

    useful = "✅ YES" if frequency >= 40 else "❌ NO"
    if frequency >= 40:
        useful_layers += 1

    print(f"{layer:<8} Expert {most_common_expert:<8} {frequency:>5.1f}%       {useful}")

print(f"\n📌 RESULT: {useful_layers}/32 layers have a dominant expert (≥40%)")
print("   This means: For most layers, you CANNOT just pre-load one expert")

# ============================================================
# USEFUL PATTERN 2: (Layer, Current Expert) → Next Expert
# "At Layer X, if Expert A, then Layer X+1 will be Expert B"
# ============================================================
print("\n" + "=" * 70)
print("PATTERN 2: Layer-Specific Transitions")
print("(At Layer X, Expert A → Expert B at Layer X+1)")
print("=" * 70)

# Build: (layer, from_expert) → list of next experts
layer_transitions = defaultdict(lambda: defaultdict(list))

for journey in all_journeys:
    for i in range(len(journey) - 1):
        if journey[i+1]['layer'] == journey[i]['layer'] + 1:
            layer = journey[i]['layer']
            from_e = journey[i]['experts'][0]
            to_e = journey[i+1]['experts'][0]
            layer_transitions[layer][from_e].append(to_e)

print("\nLooking for STRONG patterns (≥60% confidence):")
print("-" * 60)

strong_patterns = []
all_patterns = []

for layer in range(31):
    for from_e in range(8):
        next_experts = layer_transitions[layer][from_e]
        if len(next_experts) >= 2:  # Need at least 2 samples
            counts = Counter(next_experts)
            most_common, count = counts.most_common(1)[0]
            confidence = count / len(next_experts) * 100

            pattern = {
                'layer': layer,
                'from': from_e,
                'to': most_common,
                'confidence': confidence,
                'samples': len(next_experts)
            }
            all_patterns.append(pattern)

            if confidence >= 60:
                strong_patterns.append(pattern)
                print(f"  Layer {layer}: Expert {from_e} → Expert {most_common} ({confidence:.0f}%, n={len(next_experts)})")

print(f"\n📌 RESULT: Found {len(strong_patterns)} strong patterns (≥60%)")
print(f"   Out of {len(all_patterns)} total patterns")

if strong_patterns:
    avg_conf = sum(p['confidence'] for p in strong_patterns) / len(strong_patterns)
    print(f"   Average confidence of strong patterns: {avg_conf:.1f}%")

# ============================================================
# USEFUL PATTERN 3: (Layer, Pair) → Next Expert
# "At Layer X, if Pair (A,B), then Layer X+1 will be Expert C"
# ============================================================
print("\n" + "=" * 70)
print("PATTERN 3: Pair-Based Prediction")
print("(At Layer X, if Pair (A,B), then Layer X+1 will be Expert C)")
print("=" * 70)

pair_transitions = defaultdict(lambda: defaultdict(list))

for journey in all_journeys:
    for i in range(len(journey) - 1):
        if journey[i+1]['layer'] == journey[i]['layer'] + 1:
            layer = journey[i]['layer']
            pair = tuple(journey[i]['experts'])  # (primary, secondary)
            next_e = journey[i+1]['experts'][0]
            pair_transitions[layer][pair].append(next_e)

print("\nLooking for STRONG pair patterns (≥70% confidence, n≥2):")
print("-" * 60)

strong_pair_patterns = []

for layer in range(31):
    for pair, next_list in pair_transitions[layer].items():
        if len(next_list) >= 2:
            counts = Counter(next_list)
            most_common, count = counts.most_common(1)[0]
            confidence = count / len(next_list) * 100

            if confidence >= 70:
                strong_pair_patterns.append({
                    'layer': layer,
                    'pair': pair,
                    'to': most_common,
                    'confidence': confidence,
                    'samples': len(next_list)
                })
                print(f"  Layer {layer}: Pair {pair} → Expert {most_common} ({confidence:.0f}%, n={len(next_list)})")

print(f"\n📌 RESULT: Found {len(strong_pair_patterns)} strong pair patterns (≥70%)")

# ============================================================
# SUMMARY: What is ACTUALLY Useful?
# ============================================================
print("\n" + "=" * 70)
print("🎯 HONEST SUMMARY: What Can We Actually Use?")
print("=" * 70)

print(f"""
FROM 10 TOKEN JOURNEYS (320 layer transitions):

1. LAYER-SPECIFIC EXPERT (Pre-load most common):
   - Only {useful_layers}/32 layers have dominant expert (≥40%)
   - NOT reliable for most layers

2. LAYER-SPECIFIC TRANSITIONS (Layer X, Expert A → Expert B):
   - Found {len(strong_patterns)} patterns with ≥60% confidence
   - These ARE useful for specific (layer, expert) combinations

3. PAIR-BASED PREDICTION:
   - Found {len(strong_pair_patterns)} patterns with ≥70% confidence
   - Most specific, but need more data for reliability

🔴 HONEST CONCLUSION:
""")

if len(strong_patterns) > 20:
    print("   ✅ We found USEFUL patterns in the transitions!")
    print(f"   ✅ {len(strong_patterns)} specific (Layer, Expert) → Next Expert rules")
    print("   ✅ These can be used for pre-loading with 60%+ confidence")
else:
    print("   ⚠️ With only 10 tokens, we don't have enough data")
    print("   ⚠️ Each (layer, expert) combination only has ~1-3 samples")
    print("   ⚠️ Need MORE tokens to find reliable patterns")

print("""
📊 TO GET RELIABLE PATTERNS, WE NEED:
   - More token journeys (100+)
   - Then we can build reliable prediction tables
   - Each (layer, from_expert) needs 10+ samples for confidence
""")
