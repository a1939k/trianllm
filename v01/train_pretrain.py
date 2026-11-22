import torch
from torch.utils.data import DataLoader
from pretriandataset import PretrainDataset
from zerotransformer import ZeroTransformerModel,ZeroTransformerConfig
import torch.nn.functional as F

#开始训练

#训练数据
trian_dataset_all=PretrainDataset(
    data_dir="/data/datasets/dataset/pretrain_t2t_mini.jsonl",
    tokenizer_path="/data/datasets/tokenizer/tokenizer.json",
    max_sequence_length=ZeroTransformerConfig.max_sequence_length)
train_dataset, val_dataset = torch.utils.data.random_split(trian_dataset_all, [0.9, 0.1],random_seed=42)
train_loader = DataLoader(train_dataset, batch_size=12, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=12, shuffle=False)



#开始训练
model=ZeroTransformer(config=ZeroTransformerConfig())
optimizer=torch.optim.Adam(model.parameters(),lr=3e-4)
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)

# 训练循环
def train(model, optimizer, train_loader, device):
    model.train()
    total_loss = 0
    for batch_idx, (x, y) in enumerate(train_loader):
        # 将数据移到设备上
        x = x.to(device)
        y = y.to(device)
        # 前向传播
        logits,loss = model(x)
        # 计算损失
#        loss = F.cross_entropy(logits, y)
        # 反向传播
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        if batch_idx % 100 == 0:
            print(f"Batch {batch_idx}/{len(train_loader)}, Loss: {total_loss / (batch_idx + 1):.3f}")
    return total_loss
def eval(model,val_loader,device):
    #验证
    model.eval()
    val_loss=0
    with torch.no_grad():
        for batch_idx, (x, y) in enumerate(val_loader):
            # 将数据移到设备上
            x = x.to(device)
            y = y.to(device)
            # 前向传播
            logits,loss = model(x)
            # 计算损失
#            loss = F.cross_entropy(logits, y)
            val_loss += loss.item()
    return val_loss

for epoch in range(10):
    train_loss=train(model,optimizer,train_loader,device)
    val_loss=eval(model,val_loader,device)
    print(f'Epoch: {epoch}, Train Loss: {train_loss/len(train_loader):.4f}, Val Loss: {val_loss/len(val_loader):.4f}')
    # 保存模型
    avg_val_loss = val_loss / len(val_loader)
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'val_loss': avg_val_loss,
    }
    # 保存每个epoch的模型
    torch.save(checkpoint, f'checkpoints/model_epoch_{epoch}.pt')