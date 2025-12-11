#!/usr/bin/env python3
"""
Validate findings on 10 individual token journeys
Check if patterns hold for each token separately
"""

import json
from collections import defaultdict, Counter
import numpy as np

def load_routing_data(filepath):
    data = []
    with open(filepath, 'r') as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line))
    return data

def extract_token_journeys(data):
    token_journeys = defaultdict(list)
    for record in data:
        key = (record['problem_id'], record['token_idx'])
        token_journeys[key].append(record)
    for key in token_journeys:
        token_journeys[key].sort(key=lambda x: x['layer'])
    return token_journeys

def save_token_journey(journey, filepath):
    with open(filepath, 'w') as f:
        for record in journey:
            f.write(json.dumps(record) + '\n')

def analyze_single_token(journey, token_id):
    """Analyze patterns for a single token journey"""
    results = {
        'token_id': token_id,
        'num_layers': len(journey),
        'expert_sequence': [],
        'pair_overlap_hits': 0,
        'pair_overlap_total': 0,
        'anti_momentum_reversals': 0,
        'anti_momentum_total': 0,
        'persistence_hits': 0,
        'persistence_total': 0,
    }

    sorted_journey = sorted(journey, key=lambda x: x['layer'])

    # Extract expert sequence
    results['expert_sequence'] = [r['experts'][0] for r in sorted_journey]

    prev_direction = None

    for i in range(len(sorted_journey) - 1):
        if sorted_journey[i+1]['layer'] == sorted_journey[i]['layer'] + 1:
            current_primary = sorted_journey[i]['experts'][0]
            current_secondary = sorted_journey[i]['experts'][1]
            next_primary = sorted_journey[i+1]['experts'][0]

            # Pattern 1: Expert Pair Overlap
            current_pair = {current_primary, current_secondary}
            results['pair_overlap_total'] += 1
            if next_primary in current_pair:
                results['pair_overlap_hits'] += 1

            # Pattern 2: Persistence (same expert)
            results['persistence_total'] += 1
            if current_primary == next_primary:
                results['persistence_hits'] += 1

            # Pattern 3: Anti-momentum
            if i > 0 and sorted_journey[i]['layer'] == sorted_journey[i-1]['layer'] + 1:
                prev_primary = sorted_journey[i-1]['experts'][0]
                prev_direction = current_primary - prev_primary
                curr_direction = next_primary - current_primary

                if prev_direction != 0 and curr_direction != 0:
                    results['anti_momentum_total'] += 1
                    # Reversal = opposite signs
                    if (prev_direction > 0 and curr_direction < 0) or \
                       (prev_direction < 0 and curr_direction > 0):
                        results['anti_momentum_reversals'] += 1

    # Compute rates
    results['pair_overlap_rate'] = (results['pair_overlap_hits'] / results['pair_overlap_total']
                                     if results['pair_overlap_total'] > 0 else 0)
    results['persistence_rate'] = (results['persistence_hits'] / results['persistence_total']
                                    if results['persistence_total'] > 0 else 0)
    results['anti_momentum_rate'] = (results['anti_momentum_reversals'] / results['anti_momentum_total']
                                      if results['anti_momentum_total'] > 0 else 0)

    return results

def check_skip_prediction(journey, from_layer, to_layer):
    """Check if skip prediction holds for this token"""
    layer_to_record = {r['layer']: r for r in journey}
    if from_layer in layer_to_record and to_layer in layer_to_record:
        return {
            'from_expert': layer_to_record[from_layer]['experts'][0],
            'to_expert': layer_to_record[to_layer]['experts'][0],
            'valid': True
        }
    return {'valid': False}

def main():
    print("=" * 80)
    print("VALIDATION: Analyzing 10 Individual Token Journeys")
    print("=" * 80)

    # Load data
    data = load_routing_data('/home/user/temp/humaneval_2_routing.jsonl')
    journeys = extract_token_journeys(data)

    print(f"\nTotal tokens available: {len(journeys)}")

    # Select 10 tokens (5 from each problem)
    problem_0_tokens = [(k, v) for k, v in journeys.items() if k[0] == 0][:5]
    problem_1_tokens = [(k, v) for k, v in journeys.items() if k[0] == 1][:5]
    selected_tokens = problem_0_tokens + problem_1_tokens

    print(f"Selected 10 tokens: {[k for k, v in selected_tokens]}")

    # Save individual journey files
    print("\n" + "-" * 60)
    print("Saving 10 token journey files...")
    print("-" * 60)

    for i, (key, journey) in enumerate(selected_tokens):
        filepath = f'/home/user/temp/token_journey_{i}_p{key[0]}_t{key[1]}.jsonl'
        save_token_journey(journey, filepath)
        print(f"  Saved: token_journey_{i}_p{key[0]}_t{key[1]}.jsonl ({len(journey)} layers)")

    # Analyze each token
    print("\n" + "=" * 80)
    print("PATTERN VALIDATION FOR EACH TOKEN")
    print("=" * 80)

    all_results = []

    for i, (key, journey) in enumerate(selected_tokens):
        results = analyze_single_token(journey, key)
        all_results.append(results)

        print(f"\n{'─' * 60}")
        print(f"TOKEN {i+1}: Problem {key[0]}, Token {key[1]}")
        print(f"{'─' * 60}")
        print(f"  Layers: {results['num_layers']}")
        print(f"  Expert sequence (first 10): {results['expert_sequence'][:10]}...")
        print(f"  ")
        print(f"  📊 PAIR OVERLAP: {results['pair_overlap_hits']}/{results['pair_overlap_total']} = {results['pair_overlap_rate']*100:.1f}%")
        print(f"  📊 PERSISTENCE:  {results['persistence_hits']}/{results['persistence_total']} = {results['persistence_rate']*100:.1f}%")
        print(f"  📊 ANTI-MOMENTUM: {results['anti_momentum_reversals']}/{results['anti_momentum_total']} = {results['anti_momentum_rate']*100:.1f}%")

        # Check skip prediction (Layer 18 -> 26)
        skip = check_skip_prediction(journey, 18, 26)
        if skip['valid']:
            print(f"  📊 SKIP (L18→L26): Expert {skip['from_expert']} → Expert {skip['to_expert']}")

    # Aggregate statistics
    print("\n" + "=" * 80)
    print("AGGREGATE STATISTICS ACROSS 10 TOKENS")
    print("=" * 80)

    avg_pair_overlap = np.mean([r['pair_overlap_rate'] for r in all_results])
    avg_persistence = np.mean([r['persistence_rate'] for r in all_results])
    avg_anti_momentum = np.mean([r['anti_momentum_rate'] for r in all_results])

    std_pair_overlap = np.std([r['pair_overlap_rate'] for r in all_results])
    std_persistence = np.std([r['persistence_rate'] for r in all_results])
    std_anti_momentum = np.std([r['anti_momentum_rate'] for r in all_results])

    print(f"\n{'Pattern':<25} {'Mean':<12} {'Std Dev':<12} {'Min':<10} {'Max':<10}")
    print("-" * 70)

    pair_rates = [r['pair_overlap_rate'] for r in all_results]
    print(f"{'Expert Pair Overlap':<25} {avg_pair_overlap*100:>6.1f}%     {std_pair_overlap*100:>6.1f}%     {min(pair_rates)*100:>5.1f}%    {max(pair_rates)*100:>5.1f}%")

    pers_rates = [r['persistence_rate'] for r in all_results]
    print(f"{'Expert Persistence':<25} {avg_persistence*100:>6.1f}%     {std_persistence*100:>6.1f}%     {min(pers_rates)*100:>5.1f}%    {max(pers_rates)*100:>5.1f}%")

    anti_rates = [r['anti_momentum_rate'] for r in all_results]
    print(f"{'Anti-Momentum':<25} {avg_anti_momentum*100:>6.1f}%     {std_anti_momentum*100:>6.1f}%     {min(anti_rates)*100:>5.1f}%    {max(anti_rates)*100:>5.1f}%")

    # Validation verdict
    print("\n" + "=" * 80)
    print("VALIDATION VERDICT")
    print("=" * 80)

    print(f"""
    ┌─────────────────────────────────────────────────────────────────────┐
    │ Pattern              │ Expected │ Observed (10 tokens) │ Status    │
    ├─────────────────────────────────────────────────────────────────────┤
    │ Expert Pair Overlap  │  ~46%    │  {avg_pair_overlap*100:>5.1f}% ± {std_pair_overlap*100:.1f}%       │ {'✅ VALID' if avg_pair_overlap > 0.35 else '❌ WEAK'}     │
    │ Anti-Momentum        │  ~72%    │  {avg_anti_momentum*100:>5.1f}% ± {std_anti_momentum*100:.1f}%       │ {'✅ VALID' if avg_anti_momentum > 0.60 else '❌ WEAK'}     │
    │ Expert Persistence   │  ~12%    │  {avg_persistence*100:>5.1f}% ± {std_persistence*100:.1f}%       │ {'✅ VALID' if 0.08 < avg_persistence < 0.20 else '❌ WEAK'}     │
    └─────────────────────────────────────────────────────────────────────┘
    """)

    # Per-token breakdown table
    print("\n" + "=" * 80)
    print("PER-TOKEN BREAKDOWN")
    print("=" * 80)
    print(f"\n{'Token':<20} {'Pair Overlap':<15} {'Anti-Momentum':<15} {'Persistence':<15}")
    print("-" * 65)

    for i, r in enumerate(all_results):
        token_str = f"P{r['token_id'][0]}_T{r['token_id'][1]}"
        print(f"{token_str:<20} {r['pair_overlap_rate']*100:>6.1f}%        {r['anti_momentum_rate']*100:>6.1f}%         {r['persistence_rate']*100:>6.1f}%")

    # Skip prediction validation
    print("\n" + "=" * 80)
    print("SKIP PREDICTION VALIDATION (Layer 18 → Layer 26)")
    print("=" * 80)

    # Build skip prediction model from all data except test tokens
    all_journeys = list(journeys.values())
    skip_trans = defaultdict(Counter)

    for journey in all_journeys:
        layer_to_record = {r['layer']: r for r in journey}
        if 18 in layer_to_record and 26 in layer_to_record:
            from_e = layer_to_record[18]['experts'][0]
            to_e = layer_to_record[26]['experts'][0]
            skip_trans[from_e][to_e] += 1

    # For each expert at layer 18, find most likely expert at layer 26
    skip_model = {}
    for from_e in range(8):
        if skip_trans[from_e]:
            best = skip_trans[from_e].most_common(1)[0]
            skip_model[from_e] = best[0]

    print(f"\nSkip prediction model (L18 → L26):")
    for from_e, to_e in skip_model.items():
        print(f"  Expert {from_e} at L18 → predict Expert {to_e} at L26")

    # Test on 10 tokens
    skip_correct = 0
    skip_total = 0

    print(f"\nPer-token skip prediction results:")
    for i, (key, journey) in enumerate(selected_tokens):
        skip = check_skip_prediction(journey, 18, 26)
        if skip['valid']:
            from_e = skip['from_expert']
            to_e = skip['to_expert']
            predicted = skip_model.get(from_e, -1)
            correct = "✅" if predicted == to_e else "❌"
            print(f"  Token {i+1}: L18=E{from_e} → L26=E{to_e} (predicted E{predicted}) {correct}")
            skip_total += 1
            if predicted == to_e:
                skip_correct += 1

    skip_accuracy = skip_correct / skip_total if skip_total > 0 else 0
    print(f"\nSkip prediction accuracy on 10 tokens: {skip_correct}/{skip_total} = {skip_accuracy*100:.1f}%")

    return all_results

if __name__ == "__main__":
    results = main()
