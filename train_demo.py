from pathlib import Path
import sys, torch
from torch import nn
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from torch2rtl.models import DemoMLP

torch.manual_seed(7)
model = DemoMLP()
x = torch.randn(512, 16)
teacher = torch.randn(16, 4)
y = (x @ teacher).argmax(dim=1)

opt = torch.optim.Adam(model.parameters(), lr=0.01)
loss_fn = nn.CrossEntropyLoss()
for epoch in range(80):
    opt.zero_grad()
    loss = loss_fn(model(x), y)
    loss.backward()
    opt.step()

out = Path("outputs")
out.mkdir(exist_ok=True)
torch.save({"model_state_dict": model.state_dict()}, out/"demo_mlp.pth")
print("Saved outputs/demo_mlp.pth; final loss:", float(loss))
