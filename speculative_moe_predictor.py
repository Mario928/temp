#!/usr/bin/env python3
"""
Speculative MoE Expert Predictor
Implementation of novel routing prediction strategies for Mixtral 8x7B

Based on routing analysis findings:
1. Skip Layer Prediction (54.5% accuracy)
2. Expert Pair Overlap (46.1% hit rate)
3. Anti-Momentum Effect (72.5% reversal)
4. Layer-Specific Markov (65.8% Top-2 hit rate)

Usage:
    predictor = SpeculativeMoEPredictor()
    predictor.load_transition_matrices(routing_data)

    # At runtime
    predictions = predictor.predict_next_experts(
        layer=15,
        primary_expert=3,
        secondary_expert=5,
        prev_primary=1  # optional, for anti-momentum
    )
"""

import numpy as np
from collections import defaultdict, Counter
import json


class SpeculativeMoEPredictor:
    """
    Predicts expert selections for speculative pipeline parallelism in MoE models.
    """

    def __init__(self, num_layers=32, num_experts=8):
        self.num_layers = num_layers
        self.num_experts = num_experts

        # Layer-specific transition matrices
        self.transition_probs = None  # Shape: (num_layers-1, num_experts, num_experts)

        # Static top-k experts per layer
        self.static_topk = {}  # layer -> [expert_ids]

        # Skip layer prediction mapping
        self.skip_predictions = {}  # (from_layer, to_layer) -> transition_matrix

        # Best skip pairs discovered
        self.best_skip_pairs = [
            (18, 26, 0.545),  # 54.5% accuracy
            (10, 30, 0.510),
            (11, 31, 0.484),
            (19, 31, 0.484),
            (5, 31, 0.482),
        ]

    def load_from_routing_data(self, routing_data_path):
        """Load routing data and compute all prediction matrices."""
        data = []
        with open(routing_data_path, 'r') as f:
            for line in f:
                if line.strip():
                    data.append(json.loads(line))

        # Extract token journeys
        token_journeys = defaultdict(list)
        for record in data:
            token_journeys[(record['problem_id'], record['token_idx'])].append(record)

        for key in token_journeys:
            token_journeys[key].sort(key=lambda x: x['layer'])

        self._compute_transition_matrices(token_journeys)
        self._compute_static_topk(token_journeys)
        self._compute_skip_matrices(token_journeys)

        return self

    def _compute_transition_matrices(self, token_journeys):
        """Compute layer-specific Markov transition matrices."""
        transitions = np.zeros((self.num_layers - 1, self.num_experts, self.num_experts))

        for key, journey in token_journeys.items():
            for i in range(len(journey) - 1):
                if journey[i+1]['layer'] == journey[i]['layer'] + 1:
                    layer = journey[i]['layer']
                    from_e = journey[i]['experts'][0]
                    to_e = journey[i+1]['experts'][0]
                    transitions[layer, from_e, to_e] += 1

        # Normalize
        self.transition_probs = np.zeros_like(transitions)
        for layer in range(self.num_layers - 1):
            for e in range(self.num_experts):
                total = transitions[layer, e, :].sum()
                if total > 0:
                    self.transition_probs[layer, e, :] = transitions[layer, e, :] / total

        print(f"Computed transition matrices for {self.num_layers - 1} layer pairs")

    def _compute_static_topk(self, token_journeys, k=3):
        """Compute static top-k experts per layer."""
        layer_counts = defaultdict(Counter)

        for key, journey in token_journeys.items():
            for record in journey:
                layer = record['layer']
                layer_counts[layer][record['experts'][0]] += 1

        for layer in range(self.num_layers):
            self.static_topk[layer] = [e for e, _ in layer_counts[layer].most_common(k)]

        print(f"Computed static top-{k} experts for {self.num_layers} layers")

    def _compute_skip_matrices(self, token_journeys):
        """Compute skip prediction matrices for best skip pairs."""
        for from_layer, to_layer, _ in self.best_skip_pairs:
            skip_trans = np.zeros((self.num_experts, self.num_experts))

            for key, journey in token_journeys.items():
                layer_to_record = {r['layer']: r for r in journey}
                if from_layer in layer_to_record and to_layer in layer_to_record:
                    from_e = layer_to_record[from_layer]['experts'][0]
                    to_e = layer_to_record[to_layer]['experts'][0]
                    skip_trans[from_e, to_e] += 1

            # Normalize
            skip_probs = np.zeros_like(skip_trans)
            for e in range(self.num_experts):
                total = skip_trans[e, :].sum()
                if total > 0:
                    skip_probs[e, :] = skip_trans[e, :] / total

            self.skip_predictions[(from_layer, to_layer)] = skip_probs

        print(f"Computed skip matrices for {len(self.best_skip_pairs)} layer pairs")

    def predict_next_experts(self, layer, primary_expert, secondary_expert=None,
                             prev_primary=None, k=2):
        """
        Predict top-k experts for next layer using hybrid strategy.

        Args:
            layer: Current layer index
            primary_expert: Primary expert selected at current layer
            secondary_expert: Secondary expert selected at current layer (optional)
            prev_primary: Primary expert from previous layer (for anti-momentum)
            k: Number of experts to predict

        Returns:
            dict with:
                - 'top_k': List of top-k predicted experts
                - 'confidence': Confidence scores
                - 'strategy_contributions': Which strategies contributed
        """
        if layer >= self.num_layers - 1:
            return {'top_k': [], 'confidence': [], 'strategy_contributions': {}}

        scores = np.zeros(self.num_experts)
        contributions = defaultdict(list)

        # Strategy 1: Static baseline (weight=1.0)
        if layer + 1 in self.static_topk:
            for i, e in enumerate(self.static_topk[layer + 1][:k]):
                score = 1.0 / (i + 1)  # Decreasing weight by rank
                scores[e] += score
                contributions['static'].append((e, score))

        # Strategy 2: Markov prediction (weight=2.0)
        if self.transition_probs is not None:
            markov_probs = self.transition_probs[layer, primary_expert, :]
            for e in range(self.num_experts):
                score = markov_probs[e] * 2.0
                if score > 0:
                    scores[e] += score
                    contributions['markov'].append((e, score))

        # Strategy 3: Current pair (weight=1.5)
        if secondary_expert is not None:
            scores[primary_expert] += 0.8
            scores[secondary_expert] += 0.7
            contributions['current_pair'].append((primary_expert, 0.8))
            contributions['current_pair'].append((secondary_expert, 0.7))

        # Strategy 4: Anti-momentum (weight=0.5)
        if prev_primary is not None:
            direction = primary_expert - prev_primary
            if direction > 0:  # Went up, likely go down
                for e in range(primary_expert):
                    scores[e] += 0.3
                    contributions['anti_momentum'].append((e, 0.3))
            elif direction < 0:  # Went down, likely go up
                for e in range(primary_expert + 1, self.num_experts):
                    scores[e] += 0.3
                    contributions['anti_momentum'].append((e, 0.3))

        # Get top-k
        top_k_indices = np.argsort(scores)[-k:][::-1]
        top_k_scores = scores[top_k_indices]

        # Normalize scores to probabilities
        total = top_k_scores.sum()
        confidence = top_k_scores / total if total > 0 else top_k_scores

        return {
            'top_k': top_k_indices.tolist(),
            'confidence': confidence.tolist(),
            'strategy_contributions': dict(contributions)
        }

    def predict_skip_layer(self, from_layer, primary_expert, k=2):
        """
        Predict experts for a layer far ahead (skip prediction).

        Args:
            from_layer: Source layer
            primary_expert: Primary expert at source layer
            k: Number of experts to predict

        Returns:
            dict with predictions for each valid skip target
        """
        results = {}

        for from_l, to_l, accuracy in self.best_skip_pairs:
            if from_l == from_layer:
                key = (from_l, to_l)
                if key in self.skip_predictions:
                    probs = self.skip_predictions[key][primary_expert, :]
                    top_k_indices = np.argsort(probs)[-k:][::-1]

                    results[to_l] = {
                        'experts': top_k_indices.tolist(),
                        'probabilities': probs[top_k_indices].tolist(),
                        'expected_accuracy': accuracy
                    }

        return results

    def get_preload_plan(self, current_layer, primary_expert, secondary_expert=None,
                         prev_primary=None, lookahead=4):
        """
        Generate a complete preload plan for speculative pipeline parallelism.

        Args:
            current_layer: Current layer being processed
            primary_expert: Primary expert at current layer
            secondary_expert: Secondary expert at current layer
            prev_primary: Primary from previous layer
            lookahead: How many layers ahead to plan

        Returns:
            dict mapping layer -> list of experts to preload
        """
        plan = {}

        # Immediate next layer
        next_pred = self.predict_next_experts(
            current_layer, primary_expert, secondary_expert, prev_primary, k=3
        )
        plan[current_layer + 1] = {
            'experts': next_pred['top_k'],
            'confidence': next_pred['confidence'],
            'source': 'hybrid_prediction'
        }

        # Lookahead layers (use static + any skip predictions)
        for offset in range(2, min(lookahead + 1, self.num_layers - current_layer)):
            target_layer = current_layer + offset

            # Check for skip predictions
            skip_pred = self.predict_skip_layer(current_layer, primary_expert, k=2)
            if target_layer in skip_pred:
                plan[target_layer] = {
                    'experts': skip_pred[target_layer]['experts'],
                    'confidence': skip_pred[target_layer]['probabilities'],
                    'source': f'skip_prediction_from_{current_layer}'
                }
            elif target_layer in self.static_topk:
                plan[target_layer] = {
                    'experts': self.static_topk[target_layer][:2],
                    'confidence': [0.5, 0.3],  # Lower confidence for static
                    'source': 'static_fallback'
                }

        return plan

    def evaluate_predictions(self, routing_data_path):
        """Evaluate prediction accuracy on routing data."""
        data = []
        with open(routing_data_path, 'r') as f:
            for line in f:
                if line.strip():
                    data.append(json.loads(line))

        # Group by token
        token_journeys = defaultdict(list)
        for record in data:
            token_journeys[(record['problem_id'], record['token_idx'])].append(record)

        for key in token_journeys:
            token_journeys[key].sort(key=lambda x: x['layer'])

        # Evaluate
        top1_hits, top2_hits, top3_hits = 0, 0, 0
        total = 0

        for key, journey in token_journeys.items():
            prev_primary = None
            for i in range(len(journey) - 1):
                if journey[i+1]['layer'] == journey[i]['layer'] + 1:
                    layer = journey[i]['layer']
                    primary = journey[i]['experts'][0]
                    secondary = journey[i]['experts'][1]
                    actual_next = journey[i+1]['experts'][0]

                    pred = self.predict_next_experts(
                        layer, primary, secondary, prev_primary, k=3
                    )

                    if actual_next == pred['top_k'][0]:
                        top1_hits += 1
                    if actual_next in pred['top_k'][:2]:
                        top2_hits += 1
                    if actual_next in pred['top_k'][:3]:
                        top3_hits += 1

                    total += 1
                    prev_primary = primary

        return {
            'top1_accuracy': top1_hits / total if total > 0 else 0,
            'top2_accuracy': top2_hits / total if total > 0 else 0,
            'top3_accuracy': top3_hits / total if total > 0 else 0,
            'total_predictions': total
        }


def main():
    """Demo the predictor on routing data."""
    print("=" * 70)
    print("Speculative MoE Expert Predictor")
    print("=" * 70)

    # Initialize and load
    predictor = SpeculativeMoEPredictor()
    predictor.load_from_routing_data('/home/user/temp/humaneval_2_routing.jsonl')

    # Evaluate
    print("\n" + "-" * 50)
    print("Evaluating prediction accuracy...")
    results = predictor.evaluate_predictions('/home/user/temp/humaneval_2_routing.jsonl')

    print(f"\nResults on {results['total_predictions']} predictions:")
    print(f"  Top-1 accuracy: {results['top1_accuracy']*100:.1f}%")
    print(f"  Top-2 accuracy: {results['top2_accuracy']*100:.1f}%")
    print(f"  Top-3 accuracy: {results['top3_accuracy']*100:.1f}%")

    print(f"\nBaseline comparison:")
    print(f"  Random Top-1: 12.5%")
    print(f"  Random Top-2: 25.0%")
    print(f"  Random Top-3: 37.5%")

    # Demo prediction
    print("\n" + "-" * 50)
    print("Demo: Predict next experts from Layer 15, Expert 3")

    demo_pred = predictor.predict_next_experts(
        layer=15,
        primary_expert=3,
        secondary_expert=5,
        prev_primary=1,
        k=3
    )

    print(f"\nPredicted experts: {demo_pred['top_k']}")
    print(f"Confidence scores: {[f'{c:.3f}' for c in demo_pred['confidence']]}")
    print(f"Strategy contributions: {list(demo_pred['strategy_contributions'].keys())}")

    # Demo preload plan
    print("\n" + "-" * 50)
    print("Demo: Preload plan from Layer 18")

    plan = predictor.get_preload_plan(
        current_layer=18,
        primary_expert=2,
        secondary_expert=6,
        prev_primary=4,
        lookahead=10
    )

    print("\nPreload plan:")
    for layer, info in sorted(plan.items()):
        print(f"  Layer {layer}: experts={info['experts']}, source={info['source']}")


if __name__ == "__main__":
    main()
