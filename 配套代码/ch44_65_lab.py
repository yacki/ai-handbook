"""Independent CPU arithmetic labs. Python 3.10+, standard library only.
No model download and no 195M training. All examples are pedagogical.
Run: python ch44_65_lab.py --output ../实战数值记录44-65.json
"""
import argparse, json, math, random
from pathlib import Path

N = 195072768

def softmax(z, temperature=1.0):
    if temperature <= 0:
        raise ValueError('temperature must be positive')
    x = [v / temperature for v in z]
    m = max(x)
    e = [math.exp(v - m) for v in x]
    return [v / sum(e) for v in e]

def matmul(a, b):
    return [[sum(x * y for x, y in zip(row, col)) for col in zip(*b)] for row in a]

def monitor():
    logs = [(600, 1800.0, .6), (400, 1000.0, .4)]
    count = sum(r[0] for r in logs)
    return {'target_tokens': count, 'weighted_loss': sum(r[1] for r in logs) / count,
            'target_tokens_per_second': count / sum(r[2] for r in logs),
            'wrong_unweighted_loss': sum(r[1] / r[0] for r in logs) / len(logs)}

def sft_mask():
    # Input [BOS, USER, QUESTION, ASSISTANT, ANSWER_A, ANSWER_B]
    # Shifted targets [USER, QUESTION, ASSISTANT, ANSWER_A, ANSWER_B, EOS]
    probabilities = [.2, .1, .5, .4, .25, .5]
    mask = [0, 0, 0, 1, 1, 1]
    selected = [-math.log(p) for p, m in zip(probabilities, mask) if m]
    return {'loss_sum': sum(selected), 'token_count': sum(mask),
            'loss': sum(selected) / sum(mask),
            'first_supervised_input_position': mask.index(1)}

def lora():
    w = [[1., 0., -1.], [0., 2., 1.]]
    a, b = [[1., 2., -1.]], [[.1], [.2]]
    delta = matmul(b, a)
    merged = [[v + u for v, u in zip(row, dr)] for row, dr in zip(w, delta)]
    x = [[2.], [1.], [3.]]
    base, update = matmul(w, x), matmul(delta, x)
    separate = [[r[0] + d[0]] for r, d in zip(base, update)]
    combined = matmul(merged, x)
    assert max(abs(r[0] - s[0]) for r, s in zip(separate, combined)) < 1e-12
    qv = 24 * 2 * 8 * (768 + 768)
    return {'delta': delta, 'separate_output': separate, 'merged_output': combined,
            'qv_rank8_trainable': qv, 'trainable_fraction': qv / N}

def dpo():
    policy_w, policy_l, ref_w, ref_l, beta = -2., -3., -2.5, -2.8, .2
    advantage = (policy_w - ref_w) - (policy_l - ref_l)
    z = beta * advantage
    loss = math.log1p(math.exp(-z))
    grad_w = -beta / (1 + math.exp(z))
    return {'relative_margin': advantage, 'scaled_margin': z, 'loss': loss,
            'd_loss_d_policy_logp_w': grad_w, 'd_loss_d_policy_logp_l': -grad_w}

def bandit():
    # Exact expected reward gradient for a two-action policy, not sampled PPO.
    z = 0.
    lr = .5
    trace = []
    for step in range(20):
        p = 1 / (1 + math.exp(-z))
        trace.append({'step': step, 'p_rewarded_action': p, 'expected_reward': p})
        z += lr * p * (1 - p)
    return {'description': 'two-action exact-gradient educational bandit', 'initial': trace[0], 'final': trace[-1]}

def distillation():
    teacher = softmax([2., 1., 0.], 2.)
    student = softmax([0., 0., 0.], 2.)
    kl = sum(p * math.log(p / q) for p, q in zip(teacher, student))
    grad = [2. * (q - p) for p, q in zip(teacher, student)]
    return {'temperature': 2, 'teacher': teacher, 'student': student,
            'kl': kl, 'T_squared_KL': 4 * kl, 'gradient_T_squared_KL': grad}

def budget():
    cases = []
    for tokens in [50_000_000, 100_000_000, 20 * N, 4_000_000_000]:
        flops = 6 * N * tokens
        throughput = 20_000  # assumed, not measured GPU throughput
        active_hours = tokens / throughput / 3600
        wall_hours = active_hours / .8
        cases.append({'tokens': tokens, 'approx_flops': flops,
                      'assumed_tokens_per_second': throughput,
                      'active_hours': active_hours, 'availability': .8,
                      'wall_hours': wall_hours, 'assumed_currency_per_hour': 2.,
                      'estimated_training_cost': 2 * wall_hours})
    return {'parameter_count': N, 'cases': cases, 'note': 'planning assumptions; not a GPU benchmark'}

def filter_sampling():
    p = softmax([2., 1., 0., -1.])
    order = sorted(range(len(p)), key=lambda i: p[i], reverse=True)
    keep, mass = [], 0.
    for i in order:
        keep.append(i); mass += p[i]
        if mass >= .8:
            break
    filtered = [p[i] / mass if i in keep else 0. for i in range(len(p))]
    return {'base': p, 'temperature_2': softmax([2.,1.,0.,-1.],2),
            'top_p': .8, 'retained_indices': keep, 'filtered': filtered}

def kv():
    per_token = 2 * 24 * 12 * 64 * 2
    return {'MHA_bytes_per_token': per_token,
            'MHA_2048_one_sequence_bytes': per_token * 2048,
            'MHA_2048_one_sequence_MiB': per_token * 2048 / 2**20,
            'GQA4_bytes_per_token': per_token // 3,
            'MQA_bytes_per_token': per_token // 12}

def online_softmax():
    scores, values = [2., 1., 3., -1.], [10., 20., 30., 40.]
    direct = sum(p*v for p,v in zip(softmax(scores), values))
    m, denom, num = -math.inf, 0., 0.
    for start in range(0, len(scores), 2):
        zs, vs = scores[start:start+2], values[start:start+2]
        new_m = max(m, max(zs))
        rescale = 0. if m == -math.inf else math.exp(m - new_m)
        denom = rescale * denom + sum(math.exp(z - new_m) for z in zs)
        num = rescale * num + sum(math.exp(z - new_m) * v for z,v in zip(zs,vs))
        m = new_m
    blocked = num / denom
    assert abs(direct - blocked) < 1e-12
    return {'direct': direct, 'blocked': blocked, 'absolute_error': abs(direct-blocked)}

def quantization():
    w = [-1., -.4, .2, .9]
    scale = max(abs(v) for v in w) / 127
    q = [max(-127, min(127, round(v / scale))) for v in w]
    restored = [v * scale for v in q]
    return {'weights': w, 'scale': scale, 'int8': q, 'restored': restored,
            'max_abs_error': max(abs(a-b) for a,b in zip(w,restored)),
            'raw_fp16_bytes': 2*N, 'raw_int8_bytes': N,
            'raw_int4_bytes': (N+1)//2}

def paging():
    lengths, page_size = [17, 33, 5], 16
    pages = [math.ceil(t/page_size) for t in lengths]
    slots = sum(pages) * page_size
    return {'lengths': lengths, 'pages': pages, 'used_slots': sum(lengths),
            'allocated_slots': slots, 'tail_waste': slots - sum(lengths)}

def speculative(trials=200000):
    # One-position exact rejection correction, not a GPU speed test.
    p, q = [.5,.3,.2], [.2,.5,.3]
    accept = [min(1., a / b) for a,b in zip(p,q)]
    residual = [max(0., a-b) for a,b in zip(p,q)]
    total = sum(residual)
    residual = [v/total for v in residual]
    exact = [q[i]*accept[i] + (1-sum(q[j]*accept[j] for j in range(3)))*residual[i] for i in range(3)]
    rng = random.Random(65)
    def draw(prob):
        r = rng.random(); acc = 0.
        for i,v in enumerate(prob):
            acc += v
            if r < acc:
                return i
        return len(prob)-1
    counts = [0,0,0]
    for _ in range(trials):
        proposal = draw(q)
        result = proposal if rng.random() < accept[proposal] else draw(residual)
        counts[result] += 1
    empirical = [v/trials for v in counts]
    assert max(abs(a-b) for a,b in zip(exact,p)) < 1e-12
    assert max(abs(a-b) for a,b in zip(empirical,p)) < .01
    return {'target_p': p, 'draft_q': q, 'acceptance': accept,
            'residual_distribution': residual, 'exact_output': exact,
            'trials': trials, 'seed': 65, 'empirical_output': empirical}

def main():
    parser = argparse.ArgumentParser(); parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    labs = {name: fn() for name,fn in [('monitor',monitor),('sft_mask',sft_mask),('lora',lora),
            ('dpo',dpo),('bandit',bandit),('distillation',distillation),('budget',budget),
            ('sampling',filter_sampling),('kv',kv),('online_softmax',online_softmax),
            ('quantization',quantization),('paging',paging),('speculative',speculative)]}
    payload = {'scope': 'CPU arithmetic labs; not 195M training', 'results': labs}
    result = json.dumps(payload,ensure_ascii=False,indent=2)
    if args.output:
        args.output.write_text(result + '\n')
    print(result)

if __name__ == '__main__':
    main()
