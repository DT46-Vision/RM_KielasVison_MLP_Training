# Optimized for RoboMaster Vision Group MLP Training
import onnx
import os
import torch
import torchvision
from torch import nn

# 强制环境检查提示：如果运行报错，请执行：
# pip install "numpy<2.0" onnxscript onnxruntime

class MLP(nn.Module):
    def __init__(self, num_classes: int):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Flatten(),
            nn.Linear(20 * 28 * 1, 120),  # 输入尺寸 (1,28,20) → 560
            nn.ReLU(),
            nn.Linear(120, 84),
            nn.ReLU(),
            nn.Linear(84, num_classes)   # 分类数动态设置
        )

    def forward(self, x):
        return self.layers(x)


def save_model(model, num_classes: int, device: str):
    """
    导出并检查 ONNX 模型，确保不产生额外的 .data 文件
    """
    model.eval()
    model.to(device)

    # 1. 导出到临时文件
    temp_onnx = "mlp_temp.onnx"
    dummy_input = torch.randn(1, 1, 28, 20).to(device)

    # 使用 opset_version=12 提高 C++ 端兼容性
    torch.onnx.export(
        model,
        dummy_input,
        temp_onnx,
        input_names=["input"],
        output_names=["output"],
        opset_version=12,
        do_constant_folding=True
    )

    # 2. 强制合并权重（解决 .onnx.data 问题）
    try:
        loaded_model = onnx.load(temp_onnx)
        onnx.save_model(
            loaded_model,
            "mlp.onnx",
            save_as_external_data=False # 强制不使用外部数据存储
        )

        # 清理临时文件
        if os.path.exists(temp_onnx):
            os.remove(temp_onnx)
        if os.path.exists(temp_onnx + ".data"):
            os.remove(temp_onnx + ".data")

        # 验证模型
        final_model = onnx.load("mlp.onnx")
        onnx.checker.check_model(final_model)

        print(f"✅ Exported unified mlp.onnx, classes = {num_classes}")
    except Exception as e:
        print(f"❌ Export failed: {e}")


def main():
    # 设置运行设备
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Data transforms
    transform = torchvision.transforms.Compose([
        torchvision.transforms.Grayscale(),         # 转灰度，保证 1 通道
        torchvision.transforms.Resize((28, 20)),    # 固定输入大小 H=28, W=20
        torchvision.transforms.RandomAffine(
            degrees=(-5, 5), translate=(0.08, 0.08), scale=(0.9, 1.1)),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.RandomErasing(scale=(0.02, 0.02))
    ])

    # Load dataset
    dataset_path = os.path.join(os.path.dirname(__file__), '..', 'datasets')
    if not os.path.exists(dataset_path):
        print(f"❌ Error: Dataset path {dataset_path} not found!")
        return

    dataset = torchvision.datasets.ImageFolder(
        root=dataset_path,
        transform=transform
    )

    num_classes = len(dataset.classes)
    print(f"Dataset loaded: {len(dataset)} images, {num_classes} classes.")
    print("Classes:", dataset.classes, "\n")

    # Save label names to file
    with open("labels.txt", "w") as f:
        for name in dataset.classes:
            f.write(name + "\n")
    print("✅ Saved labels.txt")

    # Init model
    model = MLP(num_classes).to(device)

    # Split dataset
    train_size = int(len(dataset) * 0.8)
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])

    # Loader
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=64, shuffle=False)

    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    # Train loop
    epochs = 10 # 适当增加 Epoch 以获得更好的收敛
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)

            y_pred = model(x)
            loss = loss_fn(y_pred, y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        # Evaluate
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for x, y in test_loader:
                x, y = x.to(device), y.to(device)
                y_pred = model(x)
                _, predicted = torch.max(y_pred.data, 1)
                total += y.size(0)
                correct += (predicted == y).sum().item()

        acc = 100 * correct / total
        print(f'Epoch: {epoch+1}/{epochs}, Loss: {total_loss/len(train_loader):.4f}, Accuracy: {acc:.2f}%')

    # Final Save
    save_model(model, num_classes, device)


if __name__ == '__main__':
    main()
