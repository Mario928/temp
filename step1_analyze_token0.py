#!/usr/bin/env python3
"""
PROPER ANALYSIS: Start from ONE token journey only
Token 0, Problem 0 - 32 layers
"""

import json

print("=" * 70)
print("STEP 1: ANALYZING ONE TOKEN JOURNEY ONLY")
print("Token 0 from Problem 0 (32 layers)")
print("=" * 70)

# Load Token 0 journey
journey = []
with open('/home/user/temp/token_journey_0_p0_t0.jsonl', 'r') as f:
    for line in f:
        if line.strip():
            journey.append(json.loads(line))

journey.sort(key=lambda x: x['layer'])

print("\n📊 RAW DATA - Token 0 Journey through 32 layers:")
print("-" * 70)
print(f"{'Layer':<8} {'Primary':<10} {'Secondary':<10} {'Gating Prob':<12}")
print("-" * 70)

expert_sequence = []
for r in journey:
    expert_sequence.append(r['experts'][0])
    print(f"{r['layer']:<8} {r['experts'][0]:<10} {r['experts'][1]:<10} {r['gating_probs'][0]:.3f}")

print("\n" + "=" * 70)
print("STEP 2: VISUAL PATTERN ANALYSIS")
print("=" * 70)

print("\n🔢 Expert sequence (Primary only):")
print(expert_sequence)

# Compute transitions
print("\n📈 TRANSITION ANALYSIS (Layer by Layer):")
print("-" * 70)
print(f"{'Transition':<15} {'From→To':<12} {'Direction':<12} {'Change':<10}")
print("-" * 70)

directions = []  # 'UP', 'DOWN', 'SAME'
for i in range(len(expert_sequence) - 1):
    from_e = expert_sequence[i]
    to_e = expert_sequence[i + 1]
    change = to_e - from_e

    if change > 0:
        direction = "UP ↑"
        directions.append("UP")
    elif change < 0:
        direction = "DOWN ↓"
        directions.append("DOWN")
    else:
        direction = "SAME ="
        directions.append("SAME")

    print(f"L{i}→L{i+1}         {from_e}→{to_e}          {direction:<12} {change:+d}")

# Now let's look for patterns
print("\n" + "=" * 70)
print("STEP 3: HYPOTHESIS FORMATION FROM THIS ONE TOKEN")
print("=" * 70)

# Pattern 1: Does direction reverse?
print("\n🔬 HYPOTHESIS 1: Anti-Momentum (Direction Reversal)")
print("-" * 50)

reversals = 0
continuations = 0
reversal_details = []

for i in range(len(directions) - 1):
    prev_dir = directions[i]
    curr_dir = directions[i + 1]

    if prev_dir == "SAME" or curr_dir == "SAME":
        continue

    if prev_dir != curr_dir:
        reversals += 1
        reversal_details.append(f"L{i}→L{i+1}→L{i+2}: {prev_dir}→{curr_dir} (REVERSED)")
    else:
        continuations += 1
        reversal_details.append(f"L{i}→L{i+1}→L{i+2}: {prev_dir}→{curr_dir} (continued)")

print(f"Reversals: {reversals}")
print(f"Continuations: {continuations}")
total_momentum = reversals + continuations
if total_momentum > 0:
    reversal_rate = reversals / total_momentum * 100
    print(f"Reversal Rate: {reversal_rate:.1f}%")
    print(f"\n📌 HYPOTHESIS: {reversal_rate:.0f}% of transitions REVERSE direction")

print("\nDetails:")
for d in reversal_details[:10]:
    print(f"  {d}")
if len(reversal_details) > 10:
    print(f"  ... and {len(reversal_details) - 10} more")

# Pattern 2: Expert Pair Overlap
print("\n🔬 HYPOTHESIS 2: Expert Pair Overlap")
print("-" * 50)

pair_overlap_hits = 0
pair_overlap_total = 0

for i in range(len(journey) - 1):
    current_pair = {journey[i]['experts'][0], journey[i]['experts'][1]}
    next_primary = journey[i + 1]['experts'][0]

    pair_overlap_total += 1
    if next_primary in current_pair:
        pair_overlap_hits += 1
        print(f"  L{i}→L{i+1}: Pair {current_pair} → Primary {next_primary} ✅ IN PAIR")
    else:
        print(f"  L{i}→L{i+1}: Pair {current_pair} → Primary {next_primary} ❌ NOT in pair")

pair_overlap_rate = pair_overlap_hits / pair_overlap_total * 100
print(f"\nPair Overlap: {pair_overlap_hits}/{pair_overlap_total} = {pair_overlap_rate:.1f}%")
print(f"📌 HYPOTHESIS: {pair_overlap_rate:.0f}% of next primaries are in current pair")

# Pattern 3: Expert Persistence
print("\n🔬 HYPOTHESIS 3: Expert Persistence (Same Expert)")
print("-" * 50)

persistence_hits = 0
for i in range(len(expert_sequence) - 1):
    if expert_sequence[i] == expert_sequence[i + 1]:
        persistence_hits += 1
        print(f"  L{i}→L{i+1}: Expert {expert_sequence[i]} stayed SAME ✅")

persistence_rate = persistence_hits / (len(expert_sequence) - 1) * 100
print(f"\nPersistence: {persistence_hits}/{len(expert_sequence)-1} = {persistence_rate:.1f}%")
print(f"📌 HYPOTHESIS: {persistence_rate:.0f}% of experts stay the same")

# Pattern 4: Look for skip correlations
print("\n🔬 HYPOTHESIS 4: Skip Layer Correlation")
print("-" * 50)

# Check if Layer 0 correlates with later layers
print(f"Layer 0 Expert: {expert_sequence[0]}")
print(f"Layer 8 Expert: {expert_sequence[8]} (same? {expert_sequence[0] == expert_sequence[8]})")
print(f"Layer 16 Expert: {expert_sequence[16]} (same? {expert_sequence[0] == expert_sequence[16]})")
print(f"Layer 24 Expert: {expert_sequence[24]} (same? {expert_sequence[0] == expert_sequence[24]})")

# Check Layer 18 -> Layer 26
print(f"\nLayer 18 Expert: {expert_sequence[18]}")
print(f"Layer 26 Expert: {expert_sequence[26]}")
print(f"📌 OBSERVATION: L18={expert_sequence[18]} → L26={expert_sequence[26]}")

# Pattern 5: Gating probability analysis
print("\n🔬 HYPOTHESIS 5: High Gating Confidence = More Predictable?")
print("-" * 50)

high_conf_persist = 0
high_conf_total = 0
low_conf_persist = 0
low_conf_total = 0

for i in range(len(journey) - 1):
    prob = journey[i]['gating_probs'][0]
    same = journey[i]['experts'][0] == journey[i + 1]['experts'][0]

    if prob >= 0.8:
        high_conf_total += 1
        if same:
            high_conf_persist += 1
    else:
        low_conf_total += 1
        if same:
            low_conf_persist += 1

print(f"High confidence (≥0.8): {high_conf_persist}/{high_conf_total} persist = {high_conf_persist/high_conf_total*100 if high_conf_total > 0 else 0:.1f}%")
print(f"Low confidence (<0.8): {low_conf_persist}/{low_conf_total} persist = {low_conf_persist/low_conf_total*100 if low_conf_total > 0 else 0:.1f}%")

# Summary
print("\n" + "=" * 70)
print("SUMMARY: HYPOTHESES FROM TOKEN 0 JOURNEY")
print("=" * 70)

print(f"""
┌─────────────────────────────────────────────────────────────────┐
│ HYPOTHESES TO VALIDATE ON TOKEN 1:                              │
├─────────────────────────────────────────────────────────────────┤
│ H1: Anti-Momentum Rate        = {reversal_rate:>5.1f}%                        │
│ H2: Expert Pair Overlap Rate  = {pair_overlap_rate:>5.1f}%                        │
│ H3: Expert Persistence Rate   = {persistence_rate:>5.1f}%                        │
│ H4: L18→L26 mapping           = {expert_sequence[18]}→{expert_sequence[26]}                            │
└─────────────────────────────────────────────────────────────────┘
""")

print("Next step: Validate these on Token 1...")
