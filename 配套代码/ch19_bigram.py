"""第19章bigram教学基线。CPU运行；不下载数据或模型。"""
import torch
import torch.nn.functional as F


class Bigram(torch.nn.Module):
    def __init__(self, vocabulary_size):
        super().__init__()
        self.table = torch.nn.Embedding(vocabulary_size, vocabulary_size)
        torch.nn.init.zeros_(self.table.weight)

    def forward(self, token_ids):
        return self.table(token_ids)

    @torch.no_grad()
    def generate(self, seed, maximum_new_tokens=8):
        generator = torch.Generator().manual_seed(seed)
        result = [4]
        for _ in range(maximum_new_tokens):
            logits = self(torch.tensor([result[-1]]))[0]
            probabilities = F.softmax(logits, dim=-1)
            token = torch.multinomial(probabilities, 1, generator=generator).item()
            result.append(token)
            if token == 3:
                break
        return result


def main():
    torch.set_num_threads(1)
    model = Bigram(5)
    inputs = torch.tensor([[4, 0, 1], [4, 0, 2]])
    targets = torch.tensor([[0, 1, 3], [0, 2, 3]])
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.1, weight_decay=0.0)
    for step in range(401):
        optimizer.zero_grad(set_to_none=True)
        logits = model(inputs)
        loss = F.cross_entropy(logits.reshape(-1, 5), targets.reshape(-1))
        if step in (0, 1, 10, 100, 400):
            print(f"step={step:3d}, loss={loss.item():.6f}")
        if step == 400:
            break
        loss.backward()
        if step == 0:
            print("first_gradient_row_A", model.table.weight.grad[0].tolist())
        optimizer.step()
    model.eval()
    with torch.no_grad():
        probabilities = F.softmax(model(torch.tensor([0])), dim=-1)[0]
        print("P(next | A)", [round(x, 6) for x in probabilities.tolist()])
    print("parameters", sum(p.numel() for p in model.parameters()))
    for seed in (7, 8, 9):
        print("generation", seed, model.generate(seed))


if __name__ == "__main__":
    main()
