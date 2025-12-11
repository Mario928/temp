#!/usr/bin/env python3
"""
STEP 2: Validate hypotheses from Token 0 on Token 1
"""

import json

print("=" * 70)
print("STEP 2: VALIDATE HYPOTHESES ON TOKEN 1")
print("=" * 70)

# Hypotheses from Token 0
H1_EXPECTED = 60.0  # Anti-momentum reversal rate
H2_EXPECTED = 19.4  # Pair overlap rate
H3_EXPECTED = 9.7   # Persistence rate
H4_L18_EXPERT = 3   # Layer 18 expert
H4_L26_EXPERT = 5   # Layer 26 expert

print(f"\n📋 HYPOTHESES FROM TOKEN 0:")
print(f"   H1: Anti-Momentum = {H1_EXPECTED}%")
print(f"   H2: Pair Overlap  = {H2_EXPECTED}%")
print(f"   H3: Persistence   = {H3_EXPECTED}%")
print(f"   H4: L18={H4_L18_EXPERT} → L26={H4_L26_EXPERT}")

# Load Token 1 journey
journey = []
with open('/home/user/temp/token_journey_1_p0_t1.jsonl', 'r') as f:
    for line in f:
        if line.strip():
            journey.append(json.loads(line))

journey.sort(key=lambda x: x['layer'])

print("\n" + "-" * 70)
print("TOKEN 1 DATA:")
print("-" * 70)

expert_sequence = [r['experts'][0] for r in journey]
print(f"Expert sequence: {expert_sequence}")

# Validate H1: Anti-Momentum
print("\n" + "=" * 70)
print("VALIDATING H1: Anti-Momentum")
print("=" * 70)

directions = []
for i in range(len(expert_sequence) - 1):
    change = expert_sequence[i + 1] - expert_sequence[i]
    if change > 0:
        directions.append("UP")
    elif change < 0:
        directions.append("DOWN")
    else:
        directions.append("SAME")

reversals = 0
continuations = 0
for i in range(len(directions) - 1):
    if directions[i] == "SAME" or directions[i + 1] == "SAME":
        continue
    if directions[i] != directions[i + 1]:
        reversals += 1
    else:
        continuations += 1

total = reversals + continuations
h1_observed = reversals / total * 100 if total > 0 else 0

print(f"Reversals: {reversals}, Continuations: {continuations}")
print(f"Expected: {H1_EXPECTED}%")
print(f"Observed: {h1_observed:.1f}%")
print(f"Difference: {abs(h1_observed - H1_EXPECTED):.1f}%")

if abs(h1_observed - H1_EXPECTED) < 20:
    print("✅ H1 VALIDATED!")
else:
    print("❌ H1 NOT VALIDATED")

# Validate H2: Pair Overlap
print("\n" + "=" * 70)
print("VALIDATING H2: Expert Pair Overlap")
print("=" * 70)

pair_hits = 0
pair_total = 0
for i in range(len(journey) - 1):
    current_pair = {journey[i]['experts'][0], journey[i]['experts'][1]}
    next_primary = journey[i + 1]['experts'][0]
    pair_total += 1
    if next_primary in current_pair:
        pair_hits += 1

h2_observed = pair_hits / pair_total * 100

print(f"Pair Overlap Hits: {pair_hits}/{pair_total}")
print(f"Expected: {H2_EXPECTED}%")
print(f"Observed: {h2_observed:.1f}%")
print(f"Difference: {abs(h2_observed - H2_EXPECTED):.1f}%")

if abs(h2_observed - H2_EXPECTED) < 15:
    print("✅ H2 VALIDATED!")
else:
    print("❌ H2 NOT VALIDATED (but check if in same ballpark)")

# Validate H3: Persistence
print("\n" + "=" * 70)
print("VALIDATING H3: Expert Persistence")
print("=" * 70)

persist_hits = 0
for i in range(len(expert_sequence) - 1):
    if expert_sequence[i] == expert_sequence[i + 1]:
        persist_hits += 1

h3_observed = persist_hits / (len(expert_sequence) - 1) * 100

print(f"Persistence Hits: {persist_hits}/{len(expert_sequence) - 1}")
print(f"Expected: {H3_EXPECTED}%")
print(f"Observed: {h3_observed:.1f}%")
print(f"Difference: {abs(h3_observed - H3_EXPECTED):.1f}%")

if abs(h3_observed - H3_EXPECTED) < 10:
    print("✅ H3 VALIDATED!")
else:
    print("❌ H3 NOT VALIDATED")

# Validate H4: Skip Layer Prediction
print("\n" + "=" * 70)
print("VALIDATING H4: Skip Layer (L18 → L26)")
print("=" * 70)

layer_to_expert = {r['layer']: r['experts'][0] for r in journey}
l18_expert = layer_to_expert.get(18, None)
l26_expert = layer_to_expert.get(26, None)

print(f"Token 0: L18={H4_L18_EXPERT} → L26={H4_L26_EXPERT}")
print(f"Token 1: L18={l18_expert} → L26={l26_expert}")

if l18_expert == H4_L18_EXPERT and l26_expert == H4_L26_EXPERT:
    print("✅ H4 EXACTLY MATCHED!")
elif l18_expert == H4_L18_EXPERT:
    print(f"📌 Same L18 expert ({l18_expert}), but L26 differs ({l26_expert} vs {H4_L26_EXPERT})")
else:
    print(f"📌 Different L18 expert ({l18_expert} vs {H4_L18_EXPERT})")

# Summary
print("\n" + "=" * 70)
print("VALIDATION SUMMARY: TOKEN 0 → TOKEN 1")
print("=" * 70)

print(f"""
┌────────────────────────────────────────────────────────────────────┐
│ Hypothesis         │ Token 0   │ Token 1   │ Match?               │
├────────────────────────────────────────────────────────────────────┤
│ H1: Anti-Momentum  │ {H1_EXPECTED:>6.1f}%   │ {h1_observed:>6.1f}%   │ {'✅ YES' if abs(h1_observed - H1_EXPECTED) < 20 else '❌ NO'}                  │
│ H2: Pair Overlap   │ {H2_EXPECTED:>6.1f}%   │ {h2_observed:>6.1f}%   │ {'✅ YES' if abs(h2_observed - H2_EXPECTED) < 15 else '⚠️ CLOSE'}                  │
│ H3: Persistence    │ {H3_EXPECTED:>6.1f}%   │ {h3_observed:>6.1f}%   │ {'✅ YES' if abs(h3_observed - H3_EXPECTED) < 10 else '❌ NO'}                  │
│ H4: L18→L26        │ {H4_L18_EXPERT}→{H4_L26_EXPERT}     │ {l18_expert}→{l26_expert}     │ (see above)          │
└────────────────────────────────────────────────────────────────────┘
""")
