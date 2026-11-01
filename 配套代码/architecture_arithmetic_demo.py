"""Chapters 84/85: toy routing and scalar recurrence, not public model reproduction."""
import json
import math


def top_two(scores):
    ids = sorted(range(len(scores)), key=lambda i: (-scores[i], i))[:2]
    weights = [math.exp(scores[i] - max(scores[j] for j in ids)) for i in ids]
    total = sum(weights)
    return ids, [w / total for w in weights]


def compose(later, earlier):
    # later(earlier(s)); an affine map is represented by (a, b).
    a2, b2 = later
    a1, b1 = earlier
    return a2 * a1, a2 * b1 + b2


def main():
    ids, weights = top_two([2, 1, 0, -1])
    assert ids == [0, 1] and abs(sum(weights) - 1) < 1e-12
    experts = [(2, 0), (0, 4)]
    output = [sum(w * e[j] for w, e in zip(weights, experts)) for j in range(2)]
    assert abs(output[0] - 1.4621171572600098) < 1e-12
    assert abs(output[1] - 1.0757656854799804) < 1e-12
    counts, capacity = [5, 1, 1, 1], 2
    overflow = sum(max(0, count - capacity) for count in counts)
    assert overflow == 3
    state = 0.0
    states = []
    for value in [2, 0, 4]:
        state = .5 * state + value
        states.append(state)
    assert states == [2, 1, 4.5]
    f1, f2, f3 = (.5, 2), (.5, 0), (.5, 4)
    left = compose(f3, compose(f2, f1))
    right = compose(compose(f3, f2), f1)
    assert left == right == (.125, 4.5)
    # Two distinct histories can collide in a fixed scalar state.
    assert .5 * 2 + 0 == .5 * 0 + 1
    selective = 0
    for value, keep in [(2, True), (100, False), (4, True)]:
        selective += value if keep else 0
    assert selective == 6
    print(json.dumps({"passed": True, "routing_weights": weights, "expert_output": output,
                      "overflow": overflow, "states": states, "affine_composition": left,
                      "selective_sum": selective, "scope": "toy mechanisms; no MoE/Mamba training"}))


if __name__ == "__main__":
    main()
