"""Chapters 87/89: reward/pass@k/budget arithmetic, no model training or inference."""
import json
import math


def normalize_rewards(rewards):
    mean = sum(rewards) / len(rewards)
    variance = sum((r - mean) ** 2 for r in rewards) / len(rewards)
    if variance == 0:
        return [0.0] * len(rewards)
    std = math.sqrt(variance)
    return [(r - mean) / std for r in rewards]


def pass_at_k(n, c, k):
    if not 0 <= c <= n or not 1 <= k <= n:
        raise ValueError("invalid sample counts")
    return 1.0 if n - c < k else 1 - math.comb(n - c, k) / math.comb(n, k)


def main():
    advantages = normalize_rewards([0, 0, 1, 1])
    assert advantages == [-1, -1, 1, 1]
    assert normalize_rewards([1, 1]) == [0, 0]
    assert pass_at_k(4, 1, 1) == .25
    assert pass_at_k(4, 1, 2) == .5
    candidate_answers = [14, 15, 16, 17]
    expected = 7 + 9
    accepted = [answer for answer in candidate_answers if answer == expected]
    assert accepted == [16]
    independent_success = 1 - (1 - .4)**4
    assert abs(independent_success - .8704) < 1e-12
    # Selection can still fail if the verifier accepts the wrong candidate.
    flawed_verifier = lambda answer: answer % 2 == 0
    assert [x for x in candidate_answers if flawed_verifier(x)] == [14, 16]
    assert 20 * 200 + 80 * 1000 == 84000
    print(json.dumps({"passed": True, "advantages": advantages, "pass_at_2": pass_at_k(4, 1, 2),
                      "independent_candidate_ceiling": independent_success, "verified_toy_answer": accepted,
                      "scope": "arithmetic and toy verifier; not DeepSeek-R1 or test-time scaling reproduction"}))


if __name__ == "__main__":
    main()
