#!/usr/bin/env python3
"""
Extract 10 tokens from Problem 0 and 10 tokens from Problem 1
Properly de-duplicate to get exactly 32 layers per token
"""

import json
from collections import defaultdict

print("="*60)
print("EXTRACTING TOKEN JOURNEYS")
print("="*60)

# Load raw data
data = [json.loads(l) for l in open('../humaneval_2_routing.jsonl')]
print(f"Total raw entries: {len(data)}")

# Group by (problem_id, token_idx, layer) - keep first entry only (de-duplicate)
seen = set()
clean = []
for d in data:
    key = (d['problem_id'], d['token_idx'], d['layer'])
    if key not in seen:
        seen.add(key)
        clean.append(d)

print(f"After de-duplication: {len(clean)} entries")

# Extract journeys
def extract_journey(problem_id, token_idx):
    """Extract 32-layer journey for specific problem and token"""
    journey = [d for d in clean if d['problem_id'] == problem_id and d['token_idx'] == token_idx]
    journey.sort(key=lambda x: x['layer'])
    return journey

# Extract 10 tokens from Problem 0
print("\n" + "-"*60)
print("PROBLEM 0 (p0): Extracting tokens 0-9")
print("-"*60)

p0_journeys = {}
for t in range(10):
    journey = extract_journey(problem_id=0, token_idx=t)
    if len(journey) == 32:
        p0_journeys[t] = journey
        experts = [j['experts'][0] for j in journey]

        # Save to file
        fname = f"p0_token_{t}.jsonl"
        with open(fname, 'w') as f:
            for entry in journey:
                f.write(json.dumps(entry) + '\n')

        print(f"  Token {t}: 32 layers ✓ -> {fname}")
        print(f"    Experts: {experts}")
    else:
        print(f"  Token {t}: {len(journey)} layers (SKIPPED)")

# Extract 10 tokens from Problem 1
print("\n" + "-"*60)
print("PROBLEM 1 (p1): Extracting tokens 0-9")
print("-"*60)

p1_journeys = {}
for t in range(10):
    journey = extract_journey(problem_id=1, token_idx=t)
    if len(journey) == 32:
        p1_journeys[t] = journey
        experts = [j['experts'][0] for j in journey]

        # Save to file
        fname = f"p1_token_{t}.jsonl"
        with open(fname, 'w') as f:
            for entry in journey:
                f.write(json.dumps(entry) + '\n')

        print(f"  Token {t}: 32 layers ✓ -> {fname}")
        print(f"    Experts: {experts}")
    else:
        print(f"  Token {t}: {len(journey)} layers (SKIPPED)")

print("\n" + "="*60)
print("SUMMARY")
print("="*60)
print(f"Problem 0 tokens extracted: {len(p0_journeys)}")
print(f"Problem 1 tokens extracted: {len(p1_journeys)}")
print(f"Total: {len(p0_journeys) + len(p1_journeys)} token journey files")
print("\nFiles created:")
print("  p0_token_0.jsonl through p0_token_9.jsonl")
print("  p1_token_0.jsonl through p1_token_9.jsonl")
