#!/usr/bin/env python3
"""
Deep Analysis - Iterations 2-10: Finding Novel Patterns
Focus on optimization opportunities for Speculative Pipeline Parallelism
"""

import json
import numpy as np
from collections import defaultdict, Counter
from itertools import combinations, product
import os

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
        token_idx = record['token_idx']
        token_journeys[token_idx].append(record)
    for token_idx in token_journeys:
        token_journeys[token_idx].sort(key=lambda x: x['layer'])
    return token_journeys

# ============================================================================
# ITERATION 2: Look-ahead with multiple layers
# ============================================================================

def analyze_multi_layer_lookahead(journeys, lookahead_steps=[1, 2, 3, 4, 5]):
    """
    How does prediction accuracy decay with lookahead distance?
    Key for determining how far ahead we can speculate
    """
    print("\n" + "=" * 70)
    print("ITERATION 2: Multi-Layer Lookahead Prediction Accuracy")
    print("=" * 70)

    results = {}

    for lookahead in lookahead_steps:
        correct = 0
        total = 0

        for token_idx, journey in journeys.items():
            layer_to_record = {r['layer']: r for r in journey}

            for layer_n in range(32 - lookahead):
                layer_target = layer_n + lookahead
                if layer_n in layer_to_record and layer_target in layer_to_record:
                    # Prediction: use same expert
                    pred = layer_to_record[layer_n]['experts'][0]
                    actual = layer_to_record[layer_target]['experts'][0]

                    if pred == actual:
                        correct += 1
                    total += 1

        accuracy = correct / total if total > 0 else 0
        results[lookahead] = accuracy
        print(f"  Lookahead {lookahead} layers: accuracy = {accuracy:.3f}")

    return results

# ============================================================================
# ITERATION 3: Layer-Specific Static Expert Assignment
# ============================================================================

def analyze_static_expert_assignment(journeys):
    """
    NOVEL IDEA: What if we statically assign the MOST COMMON expert per layer?
    This is the simplest speculation strategy - no dynamic prediction needed
    """
    print("\n" + "=" * 70)
    print("ITERATION 3: Static Expert Assignment (Most Common Per Layer)")
    print("=" * 70)

    layer_expert_counts = defaultdict(Counter)

    for token_idx, journey in journeys.items():
        for record in journey:
            layer = record['layer']
            primary = record['experts'][0]
            layer_expert_counts[layer][primary] += 1

    # Find most common expert per layer
    static_assignment = {}
    for layer in range(32):
        if layer_expert_counts[layer]:
            most_common = layer_expert_counts[layer].most_common(1)[0]
            static_assignment[layer] = most_common[0]
            total = sum(layer_expert_counts[layer].values())
            share = most_common[1] / total
            print(f"  Layer {layer}: Expert {most_common[0]} ({share*100:.1f}%)")

    # Compute overall accuracy of static assignment
    correct = 0
    total = 0
    for token_idx, journey in journeys.items():
        for record in journey:
            layer = record['layer']
            if layer in static_assignment:
                if record['experts'][0] == static_assignment[layer]:
                    correct += 1
                total += 1

    accuracy = correct / total if total > 0 else 0
    print(f"\n  Overall static assignment accuracy: {accuracy:.3f}")
    print(f"  Random baseline (1/8): 0.125")
    print(f"  Improvement over random: {accuracy/0.125:.2f}x")

    return static_assignment, accuracy

# ============================================================================
# ITERATION 4: Top-K Static Assignment
# ============================================================================

def analyze_topk_static_assignment(journeys, k_values=[2, 3, 4]):
    """
    What if we preload top-K most common experts per layer?
    Trade-off: more memory vs higher hit rate
    """
    print("\n" + "=" * 70)
    print("ITERATION 4: Top-K Static Expert Assignment")
    print("=" * 70)

    layer_expert_counts = defaultdict(Counter)

    for token_idx, journey in journeys.items():
        for record in journey:
            layer = record['layer']
            primary = record['experts'][0]
            layer_expert_counts[layer][primary] += 1

    results = {}

    for k in k_values:
        # Find top-k experts per layer
        topk_assignment = {}
        for layer in range(32):
            if layer_expert_counts[layer]:
                topk = [e for e, _ in layer_expert_counts[layer].most_common(k)]
                topk_assignment[layer] = topk

        # Compute hit rate
        hits = 0
        total = 0
        for token_idx, journey in journeys.items():
            for record in journey:
                layer = record['layer']
                if layer in topk_assignment:
                    if record['experts'][0] in topk_assignment[layer]:
                        hits += 1
                    total += 1

        hit_rate = hits / total if total > 0 else 0
        results[k] = hit_rate
        print(f"  Top-{k} static assignment hit rate: {hit_rate:.3f}")

    return results

# ============================================================================
# ITERATION 5: Markov Chain Model
# ============================================================================

def analyze_markov_chain(journeys):
    """
    Build a global Markov chain: P(next expert | current expert)
    Ignoring layer - just overall transition probabilities
    """
    print("\n" + "=" * 70)
    print("ITERATION 5: Global Markov Chain (Layer-Agnostic)")
    print("=" * 70)

    transitions = np.zeros((8, 8))

    for token_idx, journey in journeys.items():
        sorted_journey = sorted(journey, key=lambda x: x['layer'])
        for i in range(len(sorted_journey) - 1):
            if sorted_journey[i+1]['layer'] == sorted_journey[i]['layer'] + 1:
                from_e = sorted_journey[i]['experts'][0]
                to_e = sorted_journey[i+1]['experts'][0]
                transitions[from_e, to_e] += 1

    # Normalize
    row_sums = transitions.sum(axis=1, keepdims=True)
    probs = np.divide(transitions, row_sums, where=row_sums > 0)

    print("\nTransition probability matrix:")
    print("From\\To ", end="")
    for e in range(8):
        print(f"   E{e}", end="")
    print()

    for from_e in range(8):
        print(f"  E{from_e}:  ", end="")
        for to_e in range(8):
            print(f" {probs[from_e, to_e]:.2f}", end="")
        print()

    # Find strongest transitions
    print("\nStrongest transitions (>30%):")
    for from_e in range(8):
        for to_e in range(8):
            if probs[from_e, to_e] > 0.3:
                print(f"  Expert {from_e} -> Expert {to_e}: {probs[from_e, to_e]:.3f}")

    # Compute prediction accuracy using Markov chain
    correct = 0
    total = 0
    for token_idx, journey in journeys.items():
        sorted_journey = sorted(journey, key=lambda x: x['layer'])
        for i in range(len(sorted_journey) - 1):
            if sorted_journey[i+1]['layer'] == sorted_journey[i]['layer'] + 1:
                from_e = sorted_journey[i]['experts'][0]
                pred = np.argmax(probs[from_e, :])
                actual = sorted_journey[i+1]['experts'][0]
                if pred == actual:
                    correct += 1
                total += 1

    accuracy = correct / total if total > 0 else 0
    print(f"\nMarkov chain prediction accuracy: {accuracy:.3f}")

    return probs, accuracy

# ============================================================================
# ITERATION 6: Layer-Specific Markov Chain
# ============================================================================

def analyze_layer_specific_markov(journeys):
    """
    Build layer-specific Markov chains: P(next expert | current expert, layer)
    More accurate but requires more storage
    """
    print("\n" + "=" * 70)
    print("ITERATION 6: Layer-Specific Markov Chains")
    print("=" * 70)

    layer_transitions = defaultdict(lambda: np.zeros((8, 8)))

    for token_idx, journey in journeys.items():
        sorted_journey = sorted(journey, key=lambda x: x['layer'])
        for i in range(len(sorted_journey) - 1):
            layer = sorted_journey[i]['layer']
            if sorted_journey[i+1]['layer'] == layer + 1:
                from_e = sorted_journey[i]['experts'][0]
                to_e = sorted_journey[i+1]['experts'][0]
                layer_transitions[layer][from_e, to_e] += 1

    # Normalize and compute per-layer accuracy
    layer_probs = {}
    layer_accuracy = {}

    for layer in range(31):
        transitions = layer_transitions[layer]
        row_sums = transitions.sum(axis=1, keepdims=True)
        probs = np.divide(transitions, row_sums, where=row_sums > 0)
        layer_probs[layer] = probs

        # Compute accuracy for this layer
        correct = 0
        total = 0
        for token_idx, journey in journeys.items():
            layer_to_record = {r['layer']: r for r in journey}
            if layer in layer_to_record and layer + 1 in layer_to_record:
                from_e = layer_to_record[layer]['experts'][0]
                if row_sums[from_e] > 0:
                    pred = np.argmax(probs[from_e, :])
                    actual = layer_to_record[layer + 1]['experts'][0]
                    if pred == actual:
                        correct += 1
                    total += 1

        accuracy = correct / total if total > 0 else 0
        layer_accuracy[layer] = accuracy

    # Show per-layer accuracy
    print("\nPer-layer prediction accuracy:")
    for layer in range(0, 31, 4):
        print(f"  Layer {layer}->{layer+1}: {layer_accuracy[layer]:.3f}")

    avg_accuracy = np.mean(list(layer_accuracy.values()))
    print(f"\nAverage layer-specific Markov accuracy: {avg_accuracy:.3f}")

    # Find best and worst layers
    best_layer = max(layer_accuracy, key=layer_accuracy.get)
    worst_layer = min(layer_accuracy, key=layer_accuracy.get)
    print(f"Best layer: {best_layer} ({layer_accuracy[best_layer]:.3f})")
    print(f"Worst layer: {worst_layer} ({layer_accuracy[worst_layer]:.3f})")

    return layer_probs, layer_accuracy

# ============================================================================
# ITERATION 7: Expert Cluster Analysis
# ============================================================================

def analyze_expert_clusters(journeys):
    """
    NOVEL IDEA: Do experts form clusters that tend to activate together?
    If yes, we can pre-load entire clusters
    """
    print("\n" + "=" * 70)
    print("ITERATION 7: Expert Clustering Analysis")
    print("=" * 70)

    # Build co-occurrence matrix (experts appearing together in a pair)
    cooccurrence = np.zeros((8, 8))

    for token_idx, journey in journeys.items():
        for record in journey:
            e1, e2 = record['experts']
            cooccurrence[e1, e2] += 1
            cooccurrence[e2, e1] += 1

    print("\nExpert Co-occurrence Matrix:")
    print("     ", end="")
    for e in range(8):
        print(f"  E{e}", end="")
    print()

    for e1 in range(8):
        print(f"  E{e1}:", end="")
        for e2 in range(8):
            print(f" {int(cooccurrence[e1, e2]):3d}", end="")
        print()

    # Find strongest co-occurrences
    print("\nStrongest expert pairs (co-occurrence):")
    pairs = []
    for e1 in range(8):
        for e2 in range(e1+1, 8):
            pairs.append((e1, e2, cooccurrence[e1, e2]))

    pairs.sort(key=lambda x: -x[2])
    for e1, e2, count in pairs[:10]:
        print(f"  ({e1}, {e2}): {int(count)} times")

    # Sequential co-occurrence (expert at layer N, then layer N+1)
    sequential_cooc = np.zeros((8, 8))

    for token_idx, journey in journeys.items():
        sorted_journey = sorted(journey, key=lambda x: x['layer'])
        for i in range(len(sorted_journey) - 1):
            if sorted_journey[i+1]['layer'] == sorted_journey[i]['layer'] + 1:
                e1 = sorted_journey[i]['experts'][0]
                e2 = sorted_journey[i+1]['experts'][0]
                sequential_cooc[e1, e2] += 1

    print("\nSequential Expert Transitions (N -> N+1):")
    print("     ", end="")
    for e in range(8):
        print(f"  E{e}", end="")
    print()

    for e1 in range(8):
        print(f"  E{e1}:", end="")
        for e2 in range(8):
            print(f" {int(sequential_cooc[e1, e2]):3d}", end="")
        print()

    return cooccurrence, sequential_cooc

# ============================================================================
# ITERATION 8: Routing Path Entropy
# ============================================================================

def analyze_routing_path_entropy(journeys):
    """
    NOVEL IDEA: Analyze entropy of routing paths
    Low entropy = highly predictable routing = good for speculation
    """
    print("\n" + "=" * 70)
    print("ITERATION 8: Routing Path Entropy Analysis")
    print("=" * 70)

    # Get all unique routing paths
    paths = []
    for token_idx, journey in journeys.items():
        sorted_journey = sorted(journey, key=lambda x: x['layer'])
        path = tuple(r['experts'][0] for r in sorted_journey)
        paths.append(path)

    # Count unique paths
    path_counts = Counter(paths)
    total_paths = len(paths)
    unique_paths = len(path_counts)

    print(f"\nTotal token journeys: {total_paths}")
    print(f"Unique routing paths: {unique_paths}")
    print(f"Path reuse ratio: {total_paths / unique_paths:.2f}")

    # Compute path entropy
    path_probs = [count / total_paths for count in path_counts.values()]
    entropy = -sum(p * np.log2(p) for p in path_probs if p > 0)
    max_entropy = np.log2(unique_paths) if unique_paths > 0 else 0
    normalized_entropy = entropy / max_entropy if max_entropy > 0 else 0

    print(f"\nPath entropy: {entropy:.2f} bits")
    print(f"Max possible entropy: {max_entropy:.2f} bits")
    print(f"Normalized entropy: {normalized_entropy:.3f}")

    # Most common paths
    print("\nTop 10 most common paths:")
    for path, count in path_counts.most_common(10):
        pct = count / total_paths * 100
        # Show abbreviated path
        path_str = "->".join(str(e) for e in path[:5]) + "->..." + "->".join(str(e) for e in path[-3:])
        print(f"  {path_str}: {count} ({pct:.1f}%)")

    return path_counts, entropy

# ============================================================================
# ITERATION 9: Conditional Speculation Strategy
# ============================================================================

def analyze_conditional_speculation(journeys):
    """
    NOVEL IDEA: Conditional speculation based on gating confidence
    High confidence at layer N -> speculate, Low confidence -> wait
    """
    print("\n" + "=" * 70)
    print("ITERATION 9: Conditional Speculation Strategy")
    print("=" * 70)

    # Analyze prediction accuracy by gating probability buckets
    buckets = [(0.5, 0.6), (0.6, 0.7), (0.7, 0.8), (0.8, 0.9), (0.9, 1.0)]
    bucket_stats = {b: {'correct': 0, 'total': 0} for b in buckets}

    for token_idx, journey in journeys.items():
        sorted_journey = sorted(journey, key=lambda x: x['layer'])
        for i in range(len(sorted_journey) - 1):
            if sorted_journey[i+1]['layer'] == sorted_journey[i]['layer'] + 1:
                gating_prob = sorted_journey[i]['gating_probs'][0]
                current_expert = sorted_journey[i]['experts'][0]
                next_expert = sorted_journey[i+1]['experts'][0]

                for low, high in buckets:
                    if low <= gating_prob < high:
                        bucket_stats[(low, high)]['total'] += 1
                        if current_expert == next_expert:
                            bucket_stats[(low, high)]['correct'] += 1
                        break

    print("\nPersistence rate by gating confidence:")
    for (low, high), stats in bucket_stats.items():
        if stats['total'] > 0:
            rate = stats['correct'] / stats['total']
            print(f"  [{low:.1f}-{high:.1f}): {rate:.3f} (n={stats['total']})")

    # Now analyze if high confidence predicts transition better
    print("\nBest-expert prediction accuracy by confidence:")
    bucket_best_pred = {b: {'correct': 0, 'total': 0} for b in buckets}

    # Build global transition matrix for prediction
    global_trans = np.zeros((8, 8))
    for token_idx, journey in journeys.items():
        sorted_journey = sorted(journey, key=lambda x: x['layer'])
        for i in range(len(sorted_journey) - 1):
            if sorted_journey[i+1]['layer'] == sorted_journey[i]['layer'] + 1:
                from_e = sorted_journey[i]['experts'][0]
                to_e = sorted_journey[i+1]['experts'][0]
                global_trans[from_e, to_e] += 1

    row_sums = global_trans.sum(axis=1, keepdims=True)
    global_probs = np.divide(global_trans, row_sums, where=row_sums > 0)

    for token_idx, journey in journeys.items():
        sorted_journey = sorted(journey, key=lambda x: x['layer'])
        for i in range(len(sorted_journey) - 1):
            if sorted_journey[i+1]['layer'] == sorted_journey[i]['layer'] + 1:
                gating_prob = sorted_journey[i]['gating_probs'][0]
                current_expert = sorted_journey[i]['experts'][0]
                next_expert = sorted_journey[i+1]['experts'][0]
                pred = np.argmax(global_probs[current_expert, :])

                for low, high in buckets:
                    if low <= gating_prob < high:
                        bucket_best_pred[(low, high)]['total'] += 1
                        if pred == next_expert:
                            bucket_best_pred[(low, high)]['correct'] += 1
                        break

    for (low, high), stats in bucket_best_pred.items():
        if stats['total'] > 0:
            rate = stats['correct'] / stats['total']
            print(f"  [{low:.1f}-{high:.1f}): {rate:.3f} (n={stats['total']})")

    return bucket_stats, bucket_best_pred

# ============================================================================
# ITERATION 10: Expert Set Prediction
# ============================================================================

def analyze_expert_set_prediction(journeys, set_sizes=[2, 3, 4]):
    """
    NOVEL IDEA: Predict a SET of likely experts, not just one
    If any expert in set is correct, it's a hit
    """
    print("\n" + "=" * 70)
    print("ITERATION 10: Expert Set Prediction (Top-K Prediction)")
    print("=" * 70)

    # Build layer-specific transition probabilities
    layer_trans = defaultdict(lambda: np.zeros((8, 8)))

    for token_idx, journey in journeys.items():
        sorted_journey = sorted(journey, key=lambda x: x['layer'])
        for i in range(len(sorted_journey) - 1):
            layer = sorted_journey[i]['layer']
            if sorted_journey[i+1]['layer'] == layer + 1:
                from_e = sorted_journey[i]['experts'][0]
                to_e = sorted_journey[i+1]['experts'][0]
                layer_trans[layer][from_e, to_e] += 1

    # Normalize
    layer_probs = {}
    for layer in range(31):
        trans = layer_trans[layer]
        row_sums = trans.sum(axis=1, keepdims=True)
        layer_probs[layer] = np.divide(trans, row_sums, where=row_sums > 0)

    results = {}

    for set_size in set_sizes:
        hits = 0
        total = 0

        for token_idx, journey in journeys.items():
            sorted_journey = sorted(journey, key=lambda x: x['layer'])
            for i in range(len(sorted_journey) - 1):
                layer = sorted_journey[i]['layer']
                if sorted_journey[i+1]['layer'] == layer + 1:
                    from_e = sorted_journey[i]['experts'][0]
                    actual = sorted_journey[i+1]['experts'][0]

                    # Get top-k predictions
                    probs = layer_probs[layer][from_e, :]
                    topk = np.argsort(probs)[-set_size:]

                    if actual in topk:
                        hits += 1
                    total += 1

        hit_rate = hits / total if total > 0 else 0
        results[set_size] = hit_rate
        print(f"  Top-{set_size} prediction hit rate: {hit_rate:.3f}")

    print("\n  Interpretation:")
    print(f"  - With top-2 preloading, we hit {results.get(2, 0)*100:.1f}% of the time")
    print(f"  - With top-3 preloading, we hit {results.get(3, 0)*100:.1f}% of the time")
    print(f"  - Random baseline (k/8): top-2={2/8:.3f}, top-3={3/8:.3f}")

    return results

# ============================================================================
# ITERATION 11: Second Expert as Hint
# ============================================================================

def analyze_second_expert_hint(journeys):
    """
    NOVEL IDEA: Use secondary expert selection as a "hint" for next layer
    Maybe the secondary expert at N is more likely to be primary at N+1
    """
    print("\n" + "=" * 70)
    print("ITERATION 11: Secondary Expert as Next-Layer Hint")
    print("=" * 70)

    secondary_becomes_primary = 0
    primary_stays_primary = 0
    total = 0

    for token_idx, journey in journeys.items():
        sorted_journey = sorted(journey, key=lambda x: x['layer'])
        for i in range(len(sorted_journey) - 1):
            if sorted_journey[i+1]['layer'] == sorted_journey[i]['layer'] + 1:
                current_primary = sorted_journey[i]['experts'][0]
                current_secondary = sorted_journey[i]['experts'][1]
                next_primary = sorted_journey[i+1]['experts'][0]

                if current_secondary == next_primary:
                    secondary_becomes_primary += 1
                if current_primary == next_primary:
                    primary_stays_primary += 1
                total += 1

    sec_to_pri_rate = secondary_becomes_primary / total if total > 0 else 0
    persistence_rate = primary_stays_primary / total if total > 0 else 0

    print(f"\nSecondary expert at N becomes primary at N+1: {sec_to_pri_rate:.3f}")
    print(f"Primary expert at N stays primary at N+1: {persistence_rate:.3f}")
    print(f"Combined (either): {min(1.0, sec_to_pri_rate + persistence_rate):.3f}")

    # What if we preload BOTH primary and secondary?
    # Check: is next primary in {current primary, current secondary}?
    either_hits = 0
    for token_idx, journey in journeys.items():
        sorted_journey = sorted(journey, key=lambda x: x['layer'])
        for i in range(len(sorted_journey) - 1):
            if sorted_journey[i+1]['layer'] == sorted_journey[i]['layer'] + 1:
                current_pair = set(sorted_journey[i]['experts'])
                next_primary = sorted_journey[i+1]['experts'][0]
                if next_primary in current_pair:
                    either_hits += 1

    either_rate = either_hits / total if total > 0 else 0
    print(f"\nNext primary in current {'{'}primary, secondary{'}'}: {either_rate:.3f}")
    print(f"This is a KEY INSIGHT for speculation!")

    return sec_to_pri_rate, persistence_rate, either_rate

# ============================================================================
# ITERATION 12: Layer Block Prediction
# ============================================================================

def analyze_layer_block_prediction(journeys, block_size=4):
    """
    NOVEL IDEA: Predict experts for an entire block of layers at once
    Useful for pipeline parallelism with multiple stages
    """
    print("\n" + "=" * 70)
    print(f"ITERATION 12: Block Prediction (blocks of {block_size} layers)")
    print("=" * 70)

    # Build block transition model
    # Given experts at block 0, predict experts at block 1
    num_blocks = 32 // block_size

    block_patterns = defaultdict(Counter)

    for token_idx, journey in journeys.items():
        layer_to_record = {r['layer']: r for r in journey}

        for block_idx in range(num_blocks - 1):
            start_layer = block_idx * block_size
            next_start = (block_idx + 1) * block_size

            # Get pattern for current block
            current_pattern = tuple(
                layer_to_record[l]['experts'][0]
                for l in range(start_layer, start_layer + block_size)
                if l in layer_to_record
            )

            # Get pattern for next block
            next_pattern = tuple(
                layer_to_record[l]['experts'][0]
                for l in range(next_start, next_start + block_size)
                if l in layer_to_record
            )

            if len(current_pattern) == block_size and len(next_pattern) == block_size:
                block_patterns[(block_idx, current_pattern)][next_pattern] += 1

    # Compute predictability
    total_predictions = 0
    correct_predictions = 0

    for (block_idx, pattern), next_counter in block_patterns.items():
        total = sum(next_counter.values())
        best_count = next_counter.most_common(1)[0][1] if next_counter else 0
        total_predictions += total
        correct_predictions += best_count

    block_accuracy = correct_predictions / total_predictions if total_predictions > 0 else 0
    print(f"\nBlock-level prediction accuracy: {block_accuracy:.3f}")

    # Show some example block transitions
    print("\nTop block transitions (from block 0 to block 1):")
    block0_patterns = {k: v for k, v in block_patterns.items() if k[0] == 0}
    top_patterns = sorted(block0_patterns.items(), key=lambda x: sum(x[1].values()), reverse=True)[:5]

    for (_, pattern), next_counter in top_patterns:
        total = sum(next_counter.values())
        best_next, best_count = next_counter.most_common(1)[0]
        prob = best_count / total
        print(f"  {pattern} -> {best_next}: {prob:.3f} (n={total})")

    return block_accuracy

# ============================================================================
# ITERATION 13: Expert Usage Correlation Across Layers
# ============================================================================

def analyze_expert_correlation(journeys):
    """
    NOVEL IDEA: Find pairs of layers where expert choice is correlated
    If layer i and layer j are correlated, we can use layer i to predict layer j
    """
    print("\n" + "=" * 70)
    print("ITERATION 13: Expert Usage Correlation Matrix")
    print("=" * 70)

    # Build correlation matrix between layers
    # For each pair of layers, compute mutual information or correlation

    layer_experts = defaultdict(list)

    for token_idx, journey in journeys.items():
        layer_to_record = {r['layer']: r for r in journey}
        for layer in range(32):
            if layer in layer_to_record:
                layer_experts[layer].append(layer_to_record[layer]['experts'][0])
            else:
                layer_experts[layer].append(-1)  # Missing

    # Compute correlation (agreement rate) between layers
    agreement_matrix = np.zeros((32, 32))

    for l1 in range(32):
        for l2 in range(32):
            experts1 = layer_experts[l1]
            experts2 = layer_experts[l2]

            matches = sum(1 for e1, e2 in zip(experts1, experts2) if e1 == e2 and e1 >= 0)
            valid = sum(1 for e1, e2 in zip(experts1, experts2) if e1 >= 0 and e2 >= 0)

            agreement_matrix[l1, l2] = matches / valid if valid > 0 else 0

    # Find highly correlated non-adjacent layer pairs
    print("\nHighly correlated layer pairs (agreement > 20%, distance > 4):")
    correlations = []
    for l1 in range(32):
        for l2 in range(l1 + 5, 32):  # Skip adjacent layers
            corr = agreement_matrix[l1, l2]
            if corr > 0.2:
                correlations.append((l1, l2, corr))

    correlations.sort(key=lambda x: -x[2])
    for l1, l2, corr in correlations[:15]:
        print(f"  Layer {l1} <-> Layer {l2}: {corr:.3f}")

    return agreement_matrix

# ============================================================================
# MAIN
# ============================================================================

def run_deep_analysis():
    print("=" * 80)
    print("DEEP ANALYSIS - ITERATIONS 2-13")
    print("=" * 80)

    # Load data
    routing_data = load_routing_data('/home/user/temp/humaneval_2_routing.jsonl')

    # Split by problem
    problem_data = defaultdict(list)
    for record in routing_data:
        problem_data[record['problem_id']].append(record)

    # Use problem 0 as primary, problem 1 for validation
    primary_journeys = extract_token_journeys(problem_data[0])
    validation_journeys = extract_token_journeys(problem_data[1])

    print(f"\nPrimary dataset (Problem 0): {len(primary_journeys)} tokens")
    print(f"Validation dataset (Problem 1): {len(validation_journeys)} tokens")

    findings = []

    # Run all iterations
    lookahead_results = analyze_multi_layer_lookahead(primary_journeys)

    static_assignment, static_accuracy = analyze_static_expert_assignment(primary_journeys)
    if static_accuracy > 0.2:
        findings.append({
            'iteration': 3,
            'hypothesis': 'Static expert assignment per layer',
            'metric': static_accuracy,
            'details': f'{static_accuracy:.1%} accuracy (vs 12.5% random)'
        })

    topk_results = analyze_topk_static_assignment(primary_journeys)
    if topk_results.get(2, 0) > 0.4:
        findings.append({
            'iteration': 4,
            'hypothesis': 'Top-2 static assignment',
            'metric': topk_results[2],
            'details': f'{topk_results[2]:.1%} hit rate'
        })

    markov_probs, markov_accuracy = analyze_markov_chain(primary_journeys)
    if markov_accuracy > 0.3:
        findings.append({
            'iteration': 5,
            'hypothesis': 'Global Markov chain prediction',
            'metric': markov_accuracy,
            'details': f'{markov_accuracy:.1%} accuracy'
        })

    layer_probs, layer_accuracy = analyze_layer_specific_markov(primary_journeys)
    avg_layer_acc = np.mean(list(layer_accuracy.values()))
    if avg_layer_acc > 0.35:
        findings.append({
            'iteration': 6,
            'hypothesis': 'Layer-specific Markov prediction',
            'metric': avg_layer_acc,
            'details': f'{avg_layer_acc:.1%} average accuracy'
        })

    cooc, seq_cooc = analyze_expert_clusters(primary_journeys)

    path_counts, entropy = analyze_routing_path_entropy(primary_journeys)

    bucket_stats, bucket_best = analyze_conditional_speculation(primary_journeys)

    set_results = analyze_expert_set_prediction(primary_journeys)
    if set_results.get(2, 0) > 0.5:
        findings.append({
            'iteration': 10,
            'hypothesis': 'Top-2 expert set prediction',
            'metric': set_results[2],
            'details': f'{set_results[2]:.1%} hit rate (vs 25% random)'
        })

    sec_rate, persist_rate, either_rate = analyze_second_expert_hint(primary_journeys)
    if either_rate > 0.35:
        findings.append({
            'iteration': 11,
            'hypothesis': 'Secondary expert as next-layer hint',
            'metric': either_rate,
            'details': f'Next primary in current pair: {either_rate:.1%}'
        })

    block_accuracy = analyze_layer_block_prediction(primary_journeys)

    agreement_matrix = analyze_expert_correlation(primary_journeys)

    # ========================================================================
    # VALIDATION ON PROBLEM 1
    # ========================================================================
    print("\n" + "=" * 80)
    print("VALIDATION ON PROBLEM 1")
    print("=" * 80)

    print("\nValidating key findings on problem 1...")

    # Validate static assignment
    val_correct = 0
    val_total = 0
    for token_idx, journey in validation_journeys.items():
        for record in journey:
            layer = record['layer']
            if layer in static_assignment:
                if record['experts'][0] == static_assignment[layer]:
                    val_correct += 1
                val_total += 1
    val_static_acc = val_correct / val_total if val_total > 0 else 0
    print(f"Static assignment accuracy: {val_static_acc:.3f} (training: {static_accuracy:.3f})")

    # Validate secondary expert hint
    val_either = 0
    val_total = 0
    for token_idx, journey in validation_journeys.items():
        sorted_journey = sorted(journey, key=lambda x: x['layer'])
        for i in range(len(sorted_journey) - 1):
            if sorted_journey[i+1]['layer'] == sorted_journey[i]['layer'] + 1:
                current_pair = set(sorted_journey[i]['experts'])
                next_primary = sorted_journey[i+1]['experts'][0]
                if next_primary in current_pair:
                    val_either += 1
                val_total += 1
    val_either_rate = val_either / val_total if val_total > 0 else 0
    print(f"Next primary in current pair: {val_either_rate:.3f} (training: {either_rate:.3f})")

    # Validate top-2 prediction
    val_top2_hits = 0
    val_total = 0
    for token_idx, journey in validation_journeys.items():
        sorted_journey = sorted(journey, key=lambda x: x['layer'])
        for i in range(len(sorted_journey) - 1):
            layer = sorted_journey[i]['layer']
            if sorted_journey[i+1]['layer'] == layer + 1 and layer in layer_probs:
                from_e = sorted_journey[i]['experts'][0]
                actual = sorted_journey[i+1]['experts'][0]
                probs = layer_probs[layer][from_e, :]
                top2 = np.argsort(probs)[-2:]
                if actual in top2:
                    val_top2_hits += 1
                val_total += 1
    val_top2_rate = val_top2_hits / val_total if val_total > 0 else 0
    print(f"Top-2 prediction hit rate: {val_top2_rate:.3f} (training: {set_results.get(2, 0):.3f})")

    # ========================================================================
    # FINAL SUMMARY
    # ========================================================================
    print("\n" + "=" * 80)
    print("FINAL SUMMARY - ACTIONABLE FINDINGS")
    print("=" * 80)

    print("\n🎯 KEY FINDINGS FOR SPECULATIVE PIPELINE PARALLELISM:")

    print(f"""
1. **STATIC TOP-2 PRELOADING** (Simplest Strategy)
   - Just preload the 2 most common experts at each layer
   - Expected hit rate: ~{topk_results.get(2, 0)*100:.0f}%
   - No runtime prediction needed!

2. **CURRENT PAIR PREDICTION** (Dynamic Strategy)
   - Next layer's primary is often in current layer's {{primary, secondary}}
   - Hit rate: ~{either_rate*100:.0f}%
   - Simple: just keep both current experts loaded

3. **LAYER-SPECIFIC MARKOV** (Best Accuracy)
   - Use layer-specific transition probabilities
   - Average accuracy: ~{avg_layer_acc*100:.0f}%
   - Requires storing 31 transition matrices (31 * 8 * 8 = ~2KB)

4. **TOP-K DYNAMIC PREDICTION** (Best Trade-off)
   - Predict top-2 experts using Markov model
   - Hit rate: ~{set_results.get(2, 0)*100:.0f}%
   - Good balance of accuracy vs memory
""")

    print("\n📊 COMPARISON TABLE:")
    print("-" * 50)
    print(f"{'Strategy':<30} {'Hit Rate':<10} {'Cost':<10}")
    print("-" * 50)
    print(f"{'Random (baseline)':<30} {'12.5%':<10} {'None':<10}")
    print(f"{'Static Top-1 per layer':<30} {f'{static_accuracy*100:.1f}%':<10} {'32 bytes':<10}")
    print(f"{'Static Top-2 per layer':<30} {f'{topk_results.get(2,0)*100:.1f}%':<10} {'64 bytes':<10}")
    print(f"{'Current pair prediction':<30} {f'{either_rate*100:.1f}%':<10} {'Runtime':<10}")
    print(f"{'Global Markov':<30} {f'{markov_accuracy*100:.1f}%':<10} {'64 bytes':<10}")
    print(f"{'Layer Markov Top-1':<30} {f'{avg_layer_acc*100:.1f}%':<10} {'~2KB':<10}")
    print(f"{'Layer Markov Top-2':<30} {f'{set_results.get(2,0)*100:.1f}%':<10} {'~2KB':<10}")
    print("-" * 50)

    return findings

if __name__ == "__main__":
    findings = run_deep_analysis()
