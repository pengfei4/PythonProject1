import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import os

corpus = """
datawhale agent learns deep learning
datawhale agent works on nlp projects
deep learning is fascinating
nlp projects are challenging
agent learns new skills
datawhale team builds great models
deep learning models need data
nlp requires understanding language
"""

tokens = corpus.strip().split()
vocab = sorted(set(tokens))
word2idx = {w: i for i, w in enumerate(vocab)}
idx2word = {i: w for w, i in word2idx.items()}
vocab_size = len(vocab)
print(f"词表大小: {vocab_size}")
print(f"词表: {vocab}")

class TextDataset(Dataset):
    def __init__(self, tokens, word2idx, seq_len=4):
        self.tokens = tokens
        self.word2idx = word2idx
        self.seq_len = seq_len
        self.data = []
        for i in range(len(tokens) - seq_len):
            src = tokens[i:i+seq_len]
            tgt = tokens[i+1:i+seq_len+1]
            self.data.append((
                torch.tensor([word2idx[w] for w in src], dtype=torch.long),
                torch.tensor([word2idx[w] for w in tgt], dtype=torch.long)
            ))
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx]

class SimpleLM(nn.Module):
    def __init__(self, vocab_size, embed_dim=32, hidden_dim=64):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.rnn = nn.GRU(embed_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, vocab_size)
    
    def forward(self, x):
        embedded = self.embedding(x)
        output, _ = self.rnn(embedded)
        logits = self.fc(output)
        return logits

dataset = TextDataset(tokens, word2idx, seq_len=4)
dataloader = DataLoader(dataset, batch_size=2, shuffle=True)
print(f"数据集样本数: {len(dataset)}")

model = SimpleLM(vocab_size)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

checkpoint_dir = "checkpoints"
os.makedirs(checkpoint_dir, exist_ok=True)

print("\n开始训练...")
num_epochs = 20
for epoch in range(num_epochs):
    model.train()
    total_loss = 0
    for batch_idx, (src, tgt) in enumerate(dataloader):
        optimizer.zero_grad()
        logits = model(src)
        loss = criterion(logits.view(-1, vocab_size), tgt.view(-1))
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    
    avg_loss = total_loss / len(dataloader)
    if (epoch + 1) % 5 == 0 or epoch == 0:
        print(f"Epoch {epoch+1:2d}/{num_epochs} | Loss: {avg_loss:.4f}")

print("\n保存 checkpoint...")
checkpoint_path = os.path.join(checkpoint_dir, "model_checkpoint.pt")
torch.save({
    'epoch': num_epochs,
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'loss': avg_loss,
    'vocab': vocab,
    'word2idx': word2idx,
}, checkpoint_path)
print(f"Checkpoint 已保存至: {checkpoint_path}")

print("\n验证 checkpoint 加载...")
checkpoint = torch.load(checkpoint_path, weights_only=False)
model2 = SimpleLM(vocab_size)
model2.load_state_dict(checkpoint['model_state_dict'])
print(f"加载成功! 保存时的 loss: {checkpoint['loss']:.4f}")

print("\n测试模型生成:")
model.eval()
with torch.no_grad():
    test_input = torch.tensor([[word2idx['datawhale'], word2idx['agent'], word2idx['learns'], word2idx['deep']]])
    logits = model(test_input)
    pred_idx = logits[0, -1].argmax().item()
    print(f"输入: 'datawhale agent learns deep'")
    print(f"预测下一个词: '{idx2word[pred_idx]}'")
