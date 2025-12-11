#!/usr/bin/env python3
"""
Step 1: Extract individual token journeys into separate files
Creates: token_journey_0.json through token_journey_9.json
"""

import json
from collections import defaultdict

print("="*60)
print("STEP 1: Extracting Token Journeys")
print("="*60)

# Load routing data
routing_data = []
with open('../humaneval_2_routing.jsonl', 'r') as f:
    for line in f:
        routing_data.append(json.loads(line))

print(f"Loaded {len(routing_data)} routing entries")

# Group by token
journeys = defaultdict(list)
for entry in routing_data:
    token_idx = entry['token_idx']
    if token_idx < 10:  # First 10 tokens
        journeys[token_idx].append(entry)

# Sort each journey by layer
for token_idx in journeys:
    journeys[token_idx].sort(key=lambda x: x['layer'])

# Save each journey to separate file
for token_idx in range(10):
    journey = journeys[token_idx]

    # Extract key info
    journey_data = {
        "token_idx": token_idx,
        "num_layers": len(journey),
        "primary_experts": [j['experts'][0] for j in journey],
        "secondary_experts": [j['experts'][1] for j in journey],
        "full_data": journey
    }

    filename = f"token_journey_{token_idx}.json"
    with open(filename, 'w') as f:
        json.dump(journey_data, f, indent=2)

    print(f"\nToken {token_idx}: {filename}")
    print(f"  Layers: {len(journey)}")
    print(f"  Primary experts:   {journey_data['primary_experts']}")
    print(f"  Secondary experts: {journey_data['secondary_experts']}")

print("\n" + "="*60)
print("DONE! Created 10 token journey files")
print("="*60)
print("\nFiles created:")
print("  token_journey_0.json through token_journey_9.json")
print("\nTokens 0-4: TRAINING (for creating rules)")
print("Tokens 5-9: TESTING (for validation)")
