"""第13章：CPU即可运行的小型自动求导演示，不下载数据或模型。"""
import torch


def one_step():
    w = torch.tensor(1.0, dtype=torch.float64, requires_grad=True)
    b = torch.tensor(0.0, dtype=torch.float64, requires_grad=True)
    prediction = w * 2.0 + b
    loss = 0.5 * (prediction - 4.0).square()
    loss.backward()
    print(f"第一次前向：prediction={prediction.item():.6f}, loss={loss.item():.6f}")
    print(f"自动求导：grad_w={w.grad.item():.6f}, grad_b={b.grad.item():.6f}")
    with torch.no_grad():
        w -= 0.1 * w.grad
        b -= 0.1 * b.grad
    new_prediction = w * 2.0 + b
    new_loss = 0.5 * (new_prediction - 4.0).square()
    print(f"一次更新：w={w.item():.6f}, b={b.item():.6f}, new_loss={new_loss.item():.6f}")


def train():
    x = torch.tensor([[0.0], [1.0], [2.0], [3.0]], dtype=torch.float64)
    y = 2.0 * x + 1.0
    model = torch.nn.Linear(1, 1, dtype=torch.float64)
    with torch.no_grad():
        model.weight.zero_()
        model.bias.zero_()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.05)
    for step in range(401):
        optimizer.zero_grad(set_to_none=True)
        prediction = model(x)
        loss = 0.5 * (prediction - y).square().mean()
        if step in (0, 1, 10, 100, 400):
            print(f"训练 step={step:3d}, loss={loss.item():.8f}")
        if step == 400:
            break
        loss.backward()
        optimizer.step()
    model.eval()
    with torch.no_grad():
        value = model(torch.tensor([[4.0]], dtype=torch.float64)).item()
    print(f"学到参数：w={model.weight.item():.6f}, b={model.bias.item():.6f}")
    print(f"训练外输入 x=4：prediction={value:.6f}, target=9.000000")


if __name__ == "__main__":
    print("PyTorch", torch.__version__)
    one_step()
    train()
