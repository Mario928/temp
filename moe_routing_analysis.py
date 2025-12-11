#!/usr/bin/env python3
"""
MoE Routing Pattern Analysis for Speculative Pipeline Parallelism
Goal: Find novel patterns in expert routing that can be exploited for optimization
"""

import json
import numpy as np
from collections import defaultdict, Counter
from itertools import combinations
import os

# ============================================================================
# STEP 1: Load and Split Data
# ============================================================================

def load_routing_data(filepath):
    """Load routing data from JSONL file"""
    data = []
    with open(filepath, 'r') as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line))
    return data

def split_by_problem(data):
    """Split data by problem_id"""
    problem_data = defaultdict(list)
    for record in data:
        problem_data[record['problem_id']].append(record)
    return problem_data

def extract_token_journeys(problem_data):
    """Extract the journey of each token through all 32 layers"""
    token_journeys = defaultdict(list)
    for record in problem_data:
        token_idx = record['token_idx']
        token_journeys[token_idx].append(record)

    # Sort each journey by layer
    for token_idx in token_journeys:
        token_journeys[token_idx].sort(key=lambda x: x['layer'])

    return token_journeys

def save_jsonl(data, filepath):
    """Save data to JSONL file"""
    with open(filepath, 'w') as f:
        for record in data:
            f.write(json.dumps(record) + '\n')

# ============================================================================
# STEP 2: Analysis Functions
# ============================================================================

def analyze_layer_transition_matrix(token_journeys, num_experts=8, num_layers=32):
    """
    Build transition matrices: P(expert at layer N+1 | expert at layer N)
    This is a key analysis for speculative pipeline parallelism
    """
    # For primary expert transitions
    primary_transitions = np.zeros((num_layers-1, num_experts, num_experts))
    # Count transitions
    primary_counts = np.zeros((num_layers-1, num_experts))

    for token_idx, journey in token_journeys.items():
        if len(journey) < 2:
            continue
        for i in range(len(journey) - 1):
            layer_n = journey[i]['layer']
            layer_n1 = journey[i+1]['layer']
            if layer_n1 != layer_n + 1:
                continue  # Skip non-consecutive layers

            expert_n = journey[i]['experts'][0]  # Primary expert at layer N
            expert_n1 = journey[i+1]['experts'][0]  # Primary expert at layer N+1

            primary_transitions[layer_n, expert_n, expert_n1] += 1
            primary_counts[layer_n, expert_n] += 1

    # Normalize to get probabilities
    transition_probs = np.zeros_like(primary_transitions)
    for l in range(num_layers-1):
        for e in range(num_experts):
            if primary_counts[l, e] > 0:
                transition_probs[l, e, :] = primary_transitions[l, e, :] / primary_counts[l, e]

    return transition_probs, primary_transitions, primary_counts

def find_high_confidence_transitions(transition_probs, threshold=0.5):
    """Find transitions with high predictability (above threshold)"""
    high_conf = []
    num_layers, num_experts, _ = transition_probs.shape

    for layer in range(num_layers):
        for expert_from in range(num_experts):
            for expert_to in range(num_experts):
                prob = transition_probs[layer, expert_from, expert_to]
                if prob >= threshold:
                    high_conf.append({
                        'layer': layer,
                        'from_expert': expert_from,
                        'to_expert': expert_to,
                        'probability': prob
                    })

    return sorted(high_conf, key=lambda x: -x['probability'])

def analyze_expert_pairs(token_journeys, num_experts=8):
    """
    Analyze if the pair of experts (primary, secondary) at layer N predicts layer N+1
    This could reveal hidden structure in routing
    """
    # Map: (layer, pair_at_N) -> Counter of pair_at_N+1
    pair_transitions = defaultdict(Counter)

    for token_idx, journey in token_journeys.items():
        if len(journey) < 2:
            continue
        for i in range(len(journey) - 1):
            layer_n = journey[i]['layer']
            layer_n1 = journey[i+1]['layer']
            if layer_n1 != layer_n + 1:
                continue

            pair_n = tuple(sorted(journey[i]['experts']))
            pair_n1 = tuple(sorted(journey[i+1]['experts']))

            pair_transitions[(layer_n, pair_n)][pair_n1] += 1

    # Find high confidence pair transitions
    high_conf_pairs = []
    for (layer, pair_from), counter in pair_transitions.items():
        total = sum(counter.values())
        if total >= 3:  # Minimum sample requirement
            for pair_to, count in counter.most_common(1):
                prob = count / total
                if prob >= 0.4:
                    high_conf_pairs.append({
                        'layer': layer,
                        'from_pair': pair_from,
                        'to_pair': pair_to,
                        'probability': prob,
                        'count': count,
                        'total': total
                    })

    return sorted(high_conf_pairs, key=lambda x: -x['probability'])

def analyze_expert_frequency_by_layer(token_journeys, num_experts=8, num_layers=32):
    """Analyze which experts dominate at each layer"""
    layer_expert_counts = defaultdict(Counter)

    for token_idx, journey in token_journeys.items():
        for record in journey:
            layer = record['layer']
            primary_expert = record['experts'][0]
            layer_expert_counts[layer][primary_expert] += 1

    # Compute distributions
    layer_distributions = {}
    for layer in range(num_layers):
        total = sum(layer_expert_counts[layer].values())
        if total > 0:
            dist = {e: layer_expert_counts[layer][e]/total for e in range(num_experts)}
            layer_distributions[layer] = dist

    return layer_distributions

def analyze_routing_graph_structure(token_journeys, num_layers=32):
    """
    Look for a "routing graph" - are there common paths through the network?
    This could reveal structural patterns exploitable for pipeline parallelism
    """
    # Build a graph of layer->expert nodes
    # Count edge frequencies
    edge_counts = defaultdict(int)

    for token_idx, journey in token_journeys.items():
        if len(journey) < 2:
            continue
        for i in range(len(journey) - 1):
            layer_n = journey[i]['layer']
            layer_n1 = journey[i+1]['layer']
            if layer_n1 != layer_n + 1:
                continue

            expert_n = journey[i]['experts'][0]
            expert_n1 = journey[i+1]['experts'][0]

            edge = (f"L{layer_n}_E{expert_n}", f"L{layer_n1}_E{expert_n1}")
            edge_counts[edge] += 1

    return dict(sorted(edge_counts.items(), key=lambda x: -x[1])[:100])

def analyze_gating_entropy(token_journeys, num_layers=32):
    """
    Analyze the entropy of gating probabilities at each layer
    Low entropy = high confidence, could be predictable
    """
    layer_entropies = defaultdict(list)

    for token_idx, journey in token_journeys.items():
        for record in journey:
            layer = record['layer']
            probs = record['gating_probs']
            # Binary entropy for top-2
            p = probs[0]
            if p > 0 and p < 1:
                entropy = -p * np.log2(p) - (1-p) * np.log2(1-p)
            else:
                entropy = 0
            layer_entropies[layer].append(entropy)

    avg_entropies = {layer: np.mean(ents) for layer, ents in layer_entropies.items()}
    return avg_entropies

def analyze_expert_co_occurrence_patterns(token_journeys, window_size=3):
    """
    Look for co-occurrence patterns within a window of layers
    This could reveal "expert motifs"
    """
    motif_counts = Counter()

    for token_idx, journey in token_journeys.items():
        if len(journey) < window_size:
            continue
        for i in range(len(journey) - window_size + 1):
            window = journey[i:i+window_size]
            # Check if layers are consecutive
            layers = [w['layer'] for w in window]
            if layers != list(range(layers[0], layers[0] + window_size)):
                continue

            motif = tuple(w['experts'][0] for w in window)
            motif_counts[motif] += 1

    return motif_counts.most_common(50)

def analyze_layer_skip_prediction(token_journeys, skip=2, num_experts=8, num_layers=32):
    """
    Can we predict layer N+skip from layer N?
    Useful for aggressive speculation
    """
    skip_transitions = np.zeros((num_layers-skip, num_experts, num_experts))
    skip_counts = np.zeros((num_layers-skip, num_experts))

    for token_idx, journey in token_journeys.items():
        layer_to_record = {r['layer']: r for r in journey}

        for layer_n in range(num_layers - skip):
            layer_n_skip = layer_n + skip
            if layer_n in layer_to_record and layer_n_skip in layer_to_record:
                expert_n = layer_to_record[layer_n]['experts'][0]
                expert_n_skip = layer_to_record[layer_n_skip]['experts'][0]

                skip_transitions[layer_n, expert_n, expert_n_skip] += 1
                skip_counts[layer_n, expert_n] += 1

    # Normalize
    skip_probs = np.zeros_like(skip_transitions)
    for l in range(num_layers-skip):
        for e in range(num_experts):
            if skip_counts[l, e] > 0:
                skip_probs[l, e, :] = skip_transitions[l, e, :] / skip_counts[l, e]

    return skip_probs

def analyze_expert_persistence(token_journeys, num_layers=32):
    """
    How often does the same expert get selected in consecutive layers?
    This is "expert persistence" - if high, we can speculate same expert
    """
    persistence_by_layer = defaultdict(lambda: {'same': 0, 'different': 0})

    for token_idx, journey in token_journeys.items():
        if len(journey) < 2:
            continue
        for i in range(len(journey) - 1):
            layer_n = journey[i]['layer']
            layer_n1 = journey[i+1]['layer']
            if layer_n1 != layer_n + 1:
                continue

            expert_n = journey[i]['experts'][0]
            expert_n1 = journey[i+1]['experts'][0]

            if expert_n == expert_n1:
                persistence_by_layer[layer_n]['same'] += 1
            else:
                persistence_by_layer[layer_n]['different'] += 1

    # Compute persistence rate
    persistence_rates = {}
    for layer in range(num_layers - 1):
        total = persistence_by_layer[layer]['same'] + persistence_by_layer[layer]['different']
        if total > 0:
            persistence_rates[layer] = persistence_by_layer[layer]['same'] / total

    return persistence_rates

def analyze_layer_phase_patterns(token_journeys, num_layers=32, phase_size=8):
    """
    Divide layers into phases and look for patterns within each phase
    Hypothesis: Different phases of the network behave differently
    """
    num_phases = num_layers // phase_size
    phase_patterns = defaultdict(lambda: defaultdict(Counter))

    for token_idx, journey in token_journeys.items():
        for record in journey:
            layer = record['layer']
            phase = layer // phase_size
            primary = record['experts'][0]
            phase_patterns[phase][layer % phase_size][primary] += 1

    return phase_patterns

def analyze_secondary_expert_prediction(token_journeys, num_layers=32, num_experts=8):
    """
    HYPOTHESIS: Secondary expert at layer N predicts primary at layer N+1
    This could be a key insight for speculation!
    """
    secondary_to_primary = np.zeros((num_layers-1, num_experts, num_experts))
    secondary_counts = np.zeros((num_layers-1, num_experts))

    for token_idx, journey in token_journeys.items():
        if len(journey) < 2:
            continue
        for i in range(len(journey) - 1):
            layer_n = journey[i]['layer']
            layer_n1 = journey[i+1]['layer']
            if layer_n1 != layer_n + 1:
                continue

            secondary_n = journey[i]['experts'][1]  # Secondary at layer N
            primary_n1 = journey[i+1]['experts'][0]  # Primary at layer N+1

            secondary_to_primary[layer_n, secondary_n, primary_n1] += 1
            secondary_counts[layer_n, secondary_n] += 1

    # Normalize
    probs = np.zeros_like(secondary_to_primary)
    for l in range(num_layers-1):
        for e in range(num_experts):
            if secondary_counts[l, e] > 0:
                probs[l, e, :] = secondary_to_primary[l, e, :] / secondary_counts[l, e]

    return probs

def analyze_combined_expert_prediction(token_journeys, num_layers=32, num_experts=8):
    """
    Use BOTH primary and secondary at layer N to predict primary at layer N+1
    This is a more powerful predictor
    """
    combined_transitions = defaultdict(Counter)

    for token_idx, journey in token_journeys.items():
        if len(journey) < 2:
            continue
        for i in range(len(journey) - 1):
            layer_n = journey[i]['layer']
            layer_n1 = journey[i+1]['layer']
            if layer_n1 != layer_n + 1:
                continue

            pair_n = tuple(journey[i]['experts'])  # (primary, secondary) at N
            primary_n1 = journey[i+1]['experts'][0]  # Primary at N+1

            combined_transitions[(layer_n, pair_n)][primary_n1] += 1

    # Find best predictions
    predictions = []
    for (layer, pair), counter in combined_transitions.items():
        total = sum(counter.values())
        best_pred, best_count = counter.most_common(1)[0]
        prob = best_count / total
        predictions.append({
            'layer': layer,
            'input_pair': pair,
            'predicted_expert': best_pred,
            'probability': prob,
            'count': total
        })

    return sorted(predictions, key=lambda x: -x['probability'])

def analyze_gating_prob_threshold(token_journeys, thresholds=[0.6, 0.7, 0.8, 0.9]):
    """
    When gating probability is above threshold, how predictable is the next layer?
    High confidence routing might be more predictable
    """
    results = {}

    for threshold in thresholds:
        same_primary = 0
        total = 0

        for token_idx, journey in token_journeys.items():
            if len(journey) < 2:
                continue
            for i in range(len(journey) - 1):
                layer_n = journey[i]['layer']
                layer_n1 = journey[i+1]['layer']
                if layer_n1 != layer_n + 1:
                    continue

                if journey[i]['gating_probs'][0] >= threshold:
                    total += 1
                    if journey[i]['experts'][0] == journey[i+1]['experts'][0]:
                        same_primary += 1

        results[threshold] = {'same_rate': same_primary / total if total > 0 else 0, 'samples': total}

    return results

def analyze_early_layer_predictor(token_journeys, num_layers=32, num_experts=8):
    """
    Can early layers (0-3) predict expert choices in later layers?
    If so, we can start pre-loading experts very early
    """
    # Use first 4 layers to predict layer 8, 16, 24
    target_layers = [8, 16, 24, 31]

    results = {}
    for target in target_layers:
        early_to_target = defaultdict(Counter)

        for token_idx, journey in token_journeys.items():
            layer_to_record = {r['layer']: r for r in journey}

            if 0 not in layer_to_record or target not in layer_to_record:
                continue

            # Use layer 0 primary expert
            early_expert = layer_to_record[0]['experts'][0]
            target_expert = layer_to_record[target]['experts'][0]

            early_to_target[early_expert][target_expert] += 1

        # Compute predictability
        layer_results = []
        for early_e, counter in early_to_target.items():
            total = sum(counter.values())
            best_pred, best_count = counter.most_common(1)[0]
            prob = best_count / total
            layer_results.append({
                'from_expert': early_e,
                'to_expert': best_pred,
                'probability': prob,
                'samples': total
            })

        results[target] = layer_results

    return results

def analyze_expert_sequence_fingerprints(token_journeys, fingerprint_length=4):
    """
    Create "fingerprints" from first N layers and see if they predict patterns
    This is like building a routing taxonomy
    """
    fingerprint_to_rest = defaultdict(list)

    for token_idx, journey in token_journeys.items():
        if len(journey) < fingerprint_length + 1:
            continue

        # Sort by layer
        sorted_journey = sorted(journey, key=lambda x: x['layer'])

        # Get fingerprint from first layers
        fingerprint = tuple(r['experts'][0] for r in sorted_journey[:fingerprint_length])

        # Get rest of journey
        rest = tuple(r['experts'][0] for r in sorted_journey[fingerprint_length:])

        fingerprint_to_rest[fingerprint].append(rest)

    # Analyze pattern diversity for each fingerprint
    fingerprint_analysis = {}
    for fp, rest_patterns in fingerprint_to_rest.items():
        pattern_counter = Counter(rest_patterns)
        total = len(rest_patterns)
        most_common = pattern_counter.most_common(1)[0] if pattern_counter else (None, 0)

        fingerprint_analysis[fp] = {
            'total_samples': total,
            'unique_patterns': len(pattern_counter),
            'most_common_pattern': most_common[0],
            'most_common_count': most_common[1],
            'predictability': most_common[1] / total if total > 0 else 0
        }

    return fingerprint_analysis

# ============================================================================
# MAIN ANALYSIS LOOP
# ============================================================================

def run_analysis():
    print("=" * 80)
    print("MOE ROUTING PATTERN ANALYSIS FOR SPECULATIVE PIPELINE PARALLELISM")
    print("=" * 80)

    # Load data
    print("\n[STEP 1] Loading routing data...")
    routing_data = load_routing_data('/home/user/temp/humaneval_2_routing.jsonl')
    print(f"Loaded {len(routing_data)} routing records")

    # Split by problem
    print("\n[STEP 2] Splitting data by problem...")
    problem_data = split_by_problem(routing_data)
    for pid, data in problem_data.items():
        print(f"  Problem {pid}: {len(data)} records")

    # Save split files
    print("\n[STEP 3] Saving split files...")
    for pid in [0, 1]:
        if pid in problem_data:
            save_jsonl(problem_data[pid], f'/home/user/temp/problem_{pid}_routing.jsonl')
            print(f"  Saved problem_{pid}_routing.jsonl")

    # Extract token journeys for each problem
    print("\n[STEP 4] Extracting token journeys...")
    journeys_by_problem = {}
    for pid in [0, 1]:
        if pid in problem_data:
            journeys = extract_token_journeys(problem_data[pid])
            journeys_by_problem[pid] = journeys
            print(f"  Problem {pid}: {len(journeys)} tokens")

    # Save first two token journeys for problem 0
    if 0 in journeys_by_problem:
        journeys = journeys_by_problem[0]
        if 0 in journeys:
            save_jsonl(journeys[0], '/home/user/temp/problem_0_token_0_journey.jsonl')
            print("  Saved problem_0_token_0_journey.jsonl (first token)")
        if 1 in journeys:
            save_jsonl(journeys[1], '/home/user/temp/problem_0_token_1_journey.jsonl')
            print("  Saved problem_0_token_1_journey.jsonl (second token)")

    # ========================================================================
    # ITERATIVE RESEARCH LOOP
    # ========================================================================

    findings = []

    # Focus on problem 0 first token journey for initial POC
    primary_journeys = journeys_by_problem.get(0, {})
    validation_journeys = journeys_by_problem.get(1, {})

    print("\n" + "=" * 80)
    print("ITERATIVE RESEARCH LOOP - Finding Novel Patterns")
    print("=" * 80)

    # ========================================================================
    # HYPOTHESIS 1: Layer-to-Layer Transition Predictability
    # ========================================================================
    print("\n" + "-" * 60)
    print("HYPOTHESIS 1: Layer N predicts Layer N+1 expert choice")
    print("-" * 60)

    trans_probs, trans_counts, counts = analyze_layer_transition_matrix(primary_journeys)
    high_conf = find_high_confidence_transitions(trans_probs, threshold=0.4)

    print(f"\nFound {len(high_conf)} high-confidence transitions (>40%)")
    print("\nTop 20 transitions:")
    for i, t in enumerate(high_conf[:20]):
        print(f"  {i+1}. Layer {t['layer']}: Expert {t['from_expert']} -> Expert {t['to_expert']} (prob={t['probability']:.3f})")

    # Compute average predictability across all layers
    avg_best_prob = []
    for layer in range(31):
        for expert in range(8):
            if counts[layer, expert] > 0:
                best_prob = np.max(trans_probs[layer, expert, :])
                avg_best_prob.append(best_prob)

    overall_predictability = np.mean(avg_best_prob)
    print(f"\nOverall average best transition probability: {overall_predictability:.3f}")

    if overall_predictability > 0.3:
        findings.append({
            'hypothesis': 'Layer N -> Layer N+1 prediction',
            'result': 'POSITIVE',
            'metric': overall_predictability,
            'details': f'Average best prediction: {overall_predictability:.3f}'
        })

    # Validate on problem 1
    print("\n[VALIDATION on Problem 1]")
    val_trans_probs, _, val_counts = analyze_layer_transition_matrix(validation_journeys)
    val_avg_best_prob = []
    for layer in range(31):
        for expert in range(8):
            if val_counts[layer, expert] > 0:
                best_prob = np.max(val_trans_probs[layer, expert, :])
                val_avg_best_prob.append(best_prob)

    val_predictability = np.mean(val_avg_best_prob) if val_avg_best_prob else 0
    print(f"Validation average best transition probability: {val_predictability:.3f}")

    # ========================================================================
    # HYPOTHESIS 2: Expert Persistence Analysis
    # ========================================================================
    print("\n" + "-" * 60)
    print("HYPOTHESIS 2: Expert Persistence (same expert in consecutive layers)")
    print("-" * 60)

    persistence = analyze_expert_persistence(primary_journeys)
    avg_persistence = np.mean(list(persistence.values()))

    print(f"\nAverage persistence rate: {avg_persistence:.3f}")
    print("\nPersistence by layer:")
    for layer in range(0, 31, 4):
        if layer in persistence:
            print(f"  Layer {layer}->{layer+1}: {persistence[layer]:.3f}")

    # Find layers with high persistence
    high_persist_layers = [l for l, p in persistence.items() if p > 0.3]
    print(f"\nLayers with >30% persistence: {high_persist_layers}")

    if avg_persistence > 0.2:
        findings.append({
            'hypothesis': 'Expert persistence (same expert consecutive layers)',
            'result': 'POSITIVE' if avg_persistence > 0.25 else 'WEAK',
            'metric': avg_persistence,
            'details': f'Avg persistence: {avg_persistence:.3f}, High persistence layers: {high_persist_layers}'
        })

    # ========================================================================
    # HYPOTHESIS 3: Secondary Expert Predicts Next Primary
    # ========================================================================
    print("\n" + "-" * 60)
    print("HYPOTHESIS 3: Secondary expert at N predicts primary at N+1")
    print("-" * 60)

    sec_to_pri = analyze_secondary_expert_prediction(primary_journeys)

    # Find best predictions
    sec_predictions = []
    for layer in range(31):
        for expert in range(8):
            if np.sum(sec_to_pri[layer, expert, :]) > 0:
                best_next = np.argmax(sec_to_pri[layer, expert, :])
                best_prob = sec_to_pri[layer, expert, best_next]
                sec_predictions.append((layer, expert, best_next, best_prob))

    avg_sec_pred = np.mean([p[3] for p in sec_predictions]) if sec_predictions else 0
    print(f"\nAverage secondary->primary prediction: {avg_sec_pred:.3f}")

    # Compare with primary->primary
    print(f"Compare with primary->primary prediction: {overall_predictability:.3f}")

    improvement = avg_sec_pred / overall_predictability if overall_predictability > 0 else 0
    print(f"Secondary predictor is {improvement:.2f}x vs primary predictor")

    if avg_sec_pred > overall_predictability:
        findings.append({
            'hypothesis': 'Secondary expert predicts next primary better',
            'result': 'NOVEL FINDING',
            'metric': avg_sec_pred,
            'details': f'Secondary pred: {avg_sec_pred:.3f} vs Primary pred: {overall_predictability:.3f}'
        })

    # ========================================================================
    # HYPOTHESIS 4: Combined (Primary, Secondary) Prediction
    # ========================================================================
    print("\n" + "-" * 60)
    print("HYPOTHESIS 4: Combined (primary, secondary) pair predicts next layer")
    print("-" * 60)

    combined_pred = analyze_combined_expert_prediction(primary_journeys)

    # Get high confidence combined predictions
    high_conf_combined = [p for p in combined_pred if p['probability'] > 0.5 and p['count'] >= 3]
    print(f"\nFound {len(high_conf_combined)} high-confidence combined predictions (>50%, n>=3)")

    if high_conf_combined:
        avg_combined = np.mean([p['probability'] for p in high_conf_combined])
        print(f"Average probability for high-confidence combined: {avg_combined:.3f}")

        print("\nTop 10 combined predictions:")
        for i, p in enumerate(high_conf_combined[:10]):
            print(f"  {i+1}. Layer {p['layer']}: {p['input_pair']} -> Expert {p['predicted_expert']} (prob={p['probability']:.3f}, n={p['count']})")

        findings.append({
            'hypothesis': 'Combined pair prediction',
            'result': 'POSITIVE',
            'metric': avg_combined,
            'details': f'{len(high_conf_combined)} high-conf predictions, avg prob: {avg_combined:.3f}'
        })

    # ========================================================================
    # HYPOTHESIS 5: Expert Sequence Motifs
    # ========================================================================
    print("\n" + "-" * 60)
    print("HYPOTHESIS 5: Common expert sequence motifs (3-layer patterns)")
    print("-" * 60)

    motifs = analyze_expert_co_occurrence_patterns(primary_journeys, window_size=3)

    print(f"\nTop 20 most common 3-layer motifs:")
    total_motifs = sum(c for _, c in motifs)
    for motif, count in motifs[:20]:
        pct = count / total_motifs * 100
        print(f"  {motif}: {count} occurrences ({pct:.1f}%)")

    # Check if some motifs dominate
    top_5_pct = sum(c for _, c in motifs[:5]) / total_motifs * 100
    print(f"\nTop 5 motifs cover {top_5_pct:.1f}% of all occurrences")

    if top_5_pct > 20:
        findings.append({
            'hypothesis': 'Common routing motifs',
            'result': 'POSITIVE',
            'metric': top_5_pct,
            'details': f'Top 5 motifs: {top_5_pct:.1f}% coverage'
        })

    # ========================================================================
    # HYPOTHESIS 6: Early Layer Predicts Deep Layers
    # ========================================================================
    print("\n" + "-" * 60)
    print("HYPOTHESIS 6: Layer 0 predicts experts at layers 8, 16, 24, 31")
    print("-" * 60)

    early_pred = analyze_early_layer_predictor(primary_journeys)

    for target_layer, preds in early_pred.items():
        if preds:
            avg_pred = np.mean([p['probability'] for p in preds])
            print(f"\n  Layer 0 -> Layer {target_layer}:")
            print(f"    Average prediction probability: {avg_pred:.3f}")
            for p in sorted(preds, key=lambda x: -x['probability'])[:3]:
                print(f"    Expert {p['from_expert']} -> Expert {p['to_expert']}: {p['probability']:.3f} (n={p['samples']})")

    # ========================================================================
    # HYPOTHESIS 7: High Gating Confidence = More Predictable
    # ========================================================================
    print("\n" + "-" * 60)
    print("HYPOTHESIS 7: High gating confidence leads to more predictable routing")
    print("-" * 60)

    gating_results = analyze_gating_prob_threshold(primary_journeys)

    print("\nPersistence rate by gating confidence threshold:")
    for threshold, result in sorted(gating_results.items()):
        print(f"  Threshold >= {threshold}: persistence = {result['same_rate']:.3f} (n={result['samples']})")

    if gating_results.get(0.9, {}).get('same_rate', 0) > gating_results.get(0.6, {}).get('same_rate', 0):
        findings.append({
            'hypothesis': 'High gating confidence = more predictable',
            'result': 'POSITIVE',
            'metric': gating_results[0.9]['same_rate'],
            'details': f"0.9 threshold: {gating_results[0.9]['same_rate']:.3f} vs 0.6: {gating_results[0.6]['same_rate']:.3f}"
        })

    # ========================================================================
    # HYPOTHESIS 8: Gating Entropy by Layer
    # ========================================================================
    print("\n" + "-" * 60)
    print("HYPOTHESIS 8: Entropy analysis (low entropy = confident routing)")
    print("-" * 60)

    entropies = analyze_gating_entropy(primary_journeys)

    print("\nAverage gating entropy by layer:")
    low_entropy_layers = []
    for layer in range(0, 32, 4):
        if layer in entropies:
            ent = entropies[layer]
            marker = " ** LOW **" if ent < 0.6 else ""
            print(f"  Layer {layer}: {ent:.3f}{marker}")
            if ent < 0.6:
                low_entropy_layers.append(layer)

    print(f"\nLow entropy layers (<0.6): {low_entropy_layers}")

    # ========================================================================
    # HYPOTHESIS 9: Layer Phase Patterns
    # ========================================================================
    print("\n" + "-" * 60)
    print("HYPOTHESIS 9: Phase-based patterns (layers 0-7, 8-15, 16-23, 24-31)")
    print("-" * 60)

    phase_patterns = analyze_layer_phase_patterns(primary_journeys)

    for phase in range(4):
        print(f"\n  Phase {phase} (layers {phase*8}-{phase*8+7}):")
        dominant_experts = Counter()
        for rel_layer in range(8):
            expert_dist = phase_patterns[phase][rel_layer]
            if expert_dist:
                top_expert = expert_dist.most_common(1)[0][0]
                dominant_experts[top_expert] += 1
        print(f"    Dominant experts: {dominant_experts.most_common(3)}")

    # ========================================================================
    # HYPOTHESIS 10: Fingerprint-based Prediction
    # ========================================================================
    print("\n" + "-" * 60)
    print("HYPOTHESIS 10: First 4 layers fingerprint predicts rest of routing")
    print("-" * 60)

    fingerprints = analyze_expert_sequence_fingerprints(primary_journeys, fingerprint_length=4)

    # Find fingerprints with high predictability
    high_pred_fps = [(fp, info) for fp, info in fingerprints.items()
                     if info['predictability'] > 0.3 and info['total_samples'] >= 3]

    print(f"\nFound {len(high_pred_fps)} fingerprints with >30% predictability (n>=3)")

    if high_pred_fps:
        avg_fp_pred = np.mean([info['predictability'] for _, info in high_pred_fps])
        print(f"Average predictability: {avg_fp_pred:.3f}")

        print("\nTop fingerprints:")
        for fp, info in sorted(high_pred_fps, key=lambda x: -x[1]['predictability'])[:10]:
            print(f"  {fp}: pred={info['predictability']:.3f}, unique={info['unique_patterns']}, n={info['total_samples']}")

        if avg_fp_pred > 0.4:
            findings.append({
                'hypothesis': 'Fingerprint-based routing prediction',
                'result': 'NOVEL FINDING',
                'metric': avg_fp_pred,
                'details': f'{len(high_pred_fps)} fingerprints with avg {avg_fp_pred:.3f} predictability'
            })

    # ========================================================================
    # HYPOTHESIS 11: Skip Layer Prediction (Layer N -> Layer N+2)
    # ========================================================================
    print("\n" + "-" * 60)
    print("HYPOTHESIS 11: Skip prediction (Layer N -> Layer N+2)")
    print("-" * 60)

    skip_probs = analyze_layer_skip_prediction(primary_journeys, skip=2)

    skip_predictions = []
    for layer in range(30):
        for expert in range(8):
            if np.sum(skip_probs[layer, expert, :]) > 0:
                best_skip = np.argmax(skip_probs[layer, expert, :])
                best_prob = skip_probs[layer, expert, best_skip]
                skip_predictions.append(best_prob)

    avg_skip_pred = np.mean(skip_predictions) if skip_predictions else 0
    print(f"\nAverage Layer N -> Layer N+2 prediction: {avg_skip_pred:.3f}")
    print(f"Compare with N -> N+1: {overall_predictability:.3f}")

    # ========================================================================
    # HYPOTHESIS 12: Expert Pair Co-occurrence Across Layers
    # ========================================================================
    print("\n" + "-" * 60)
    print("HYPOTHESIS 12: Expert pair transitions (both experts predict both next)")
    print("-" * 60)

    pair_transitions = analyze_expert_pairs(primary_journeys)

    print(f"\nFound {len(pair_transitions)} high-confidence pair transitions")
    print("\nTop 10 pair transitions:")
    for i, p in enumerate(pair_transitions[:10]):
        print(f"  {i+1}. Layer {p['layer']}: {p['from_pair']} -> {p['to_pair']} (prob={p['probability']:.3f}, n={p['total']})")

    if pair_transitions:
        avg_pair_pred = np.mean([p['probability'] for p in pair_transitions])
        print(f"\nAverage pair transition probability: {avg_pair_pred:.3f}")

        if avg_pair_pred > 0.5:
            findings.append({
                'hypothesis': 'Expert pair transitions',
                'result': 'NOVEL FINDING',
                'metric': avg_pair_pred,
                'details': f'{len(pair_transitions)} pair transitions with avg {avg_pair_pred:.3f}'
            })

    # ========================================================================
    # HYPOTHESIS 13: Layer-Specific Expert Dominance
    # ========================================================================
    print("\n" + "-" * 60)
    print("HYPOTHESIS 13: Expert dominance varies by layer")
    print("-" * 60)

    layer_dist = analyze_expert_frequency_by_layer(primary_journeys)

    print("\nDominant expert per layer (showing every 4th layer):")
    for layer in range(0, 32, 4):
        if layer in layer_dist:
            dist = layer_dist[layer]
            top_expert = max(dist.items(), key=lambda x: x[1])
            print(f"  Layer {layer}: Expert {top_expert[0]} ({top_expert[1]*100:.1f}%)")

    # Check if some experts dominate certain layers
    high_dominance = []
    for layer, dist in layer_dist.items():
        max_share = max(dist.values())
        if max_share > 0.25:
            high_dominance.append((layer, max_share))

    print(f"\nLayers with >25% expert dominance: {len(high_dominance)}")

    # ========================================================================
    # SUMMARY OF FINDINGS
    # ========================================================================
    print("\n" + "=" * 80)
    print("SUMMARY OF FINDINGS")
    print("=" * 80)

    novel_findings = [f for f in findings if 'NOVEL' in f['result']]
    positive_findings = [f for f in findings if f['result'] == 'POSITIVE']

    print(f"\n🔬 Total findings: {len(findings)}")
    print(f"🌟 Novel findings: {len(novel_findings)}")
    print(f"✓ Positive findings: {len(positive_findings)}")

    print("\n--- ALL FINDINGS ---")
    for i, f in enumerate(findings, 1):
        status = "🌟" if 'NOVEL' in f['result'] else "✓"
        print(f"\n{i}. {status} {f['hypothesis']}")
        print(f"   Result: {f['result']}")
        print(f"   Metric: {f['metric']:.3f}")
        print(f"   Details: {f['details']}")

    return findings

if __name__ == "__main__":
    findings = run_analysis()
