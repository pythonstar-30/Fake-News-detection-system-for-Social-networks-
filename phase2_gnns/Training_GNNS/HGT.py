import torch
import torch.nn.functional as F
import torch_geometric.nn as pyg_nn
import os
import pandas as pd

# --- 1. CONFIGURATION ---
GRAPH_PATH = 'final_hetero_graph_4.pt'
HIDDEN_CHANNELS = 64
NUM_HEADS = 4  # Specific to Transformer architectures
LEARNING_RATE = 0.005 # HGT often benefits from a slightly lower LR than SAGE
EPOCHS = 100

print("--- INITIALIZING HGT GNN TRAINING ---")

# --- 2. LOAD DATA ---
if not os.path.exists(GRAPH_PATH):
    print(f" Error: {GRAPH_PATH} not found.")
    exit()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
data = torch.load(GRAPH_PATH, weights_only=False).to(device)

# --- 3. DEFINE THE HGT MODEL ---
class HGTModel(torch.nn.Module):
    def __init__(self, data_metadata, hidden_channels, out_channels, num_heads):
        super().__init__()
        
        # 1. Linear layers to project different node features to the same dimension
        self.lin_dict = torch.nn.ModuleDict()
        for node_type in data_metadata[0]:
            self.lin_dict[node_type] = pyg_nn.Linear(-1, hidden_channels)

        # 2. HGT Convolutional Layers
        # HGT handles heterogeneous relations natively, so no to_hetero() is needed!
        self.conv1 = pyg_nn.HGTConv(hidden_channels, hidden_channels, data_metadata, num_heads)
        self.conv2 = pyg_nn.HGTConv(hidden_channels, hidden_channels, data_metadata, num_heads)

        # 3. Final Classifiers for each node type we want to predict
        self.user_classifier = torch.nn.Linear(hidden_channels, 2) # Bot vs Human
        self.news_classifier = torch.nn.Linear(hidden_channels, 2) # Fake vs Real

    def forward(self, x_dict, edge_index_dict):
        # Initial projection of features
        x_dict = {node_type: self.lin_dict[node_type](x).relu() for node_type, x in x_dict.items()}

        # HGT Layer 1
        x_dict = self.conv1(x_dict, edge_index_dict)
        x_dict = {node_type: F.gelu(x) for node_type, x in x_dict.items()} # HGT often uses GELU
        
        # HGT Layer 2
        x_dict = self.conv2(x_dict, edge_index_dict)
        x_dict = {node_type: F.gelu(x) for node_type, x in x_dict.items()}

        # Predict specific node types
        out = {
            'user': self.user_classifier(x_dict['user']),
            'news': self.news_classifier(x_dict['news'])
        }
        return out

# --- 4. INITIALIZE MODEL ---
# data.metadata() provides (node_types, edge_types)
model = HGTModel(data.metadata(), HIDDEN_CHANNELS, 2, NUM_HEADS).to(device)

optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
criterion = torch.nn.CrossEntropyLoss()

# --- 5. TRAIN & TEST FUNCTIONS ---
def train():
    model.train()
    optimizer.zero_grad()
    
    out = model(data.x_dict, data.edge_index_dict)
    
    # Combined Loss for Multi-Task Learning
    u_mask, n_mask = data['user'].train_mask, data['news'].train_mask
    user_loss = criterion(out['user'][u_mask], data['user'].y[u_mask])
    news_loss = criterion(out['news'][n_mask], data['news'].y[n_mask])
    
    total_loss = user_loss + news_loss
    total_loss.backward()
    optimizer.step()
    return float(total_loss), float(user_loss), float(news_loss)

@torch.no_grad()
def test():
    model.eval()
    out = model(data.x_dict, data.edge_index_dict)
    
    u_acc = (out['user'].argmax(dim=1)[data['user'].test_mask] == data['user'].y[data['user'].test_mask]).float().mean()
    n_acc = (out['news'].argmax(dim=1)[data['news'].test_mask] == data['news'].y[data['news'].test_mask]).float().mean()
    
    return u_acc.item(), n_acc.item()

# --- 6. EXECUTION LOOP ---
for epoch in range(1, EPOCHS + 1):
    loss, u_l, n_l = train()
    u_acc, n_acc = test()
    if epoch % 10 == 0:
        print(f"Epoch {epoch:03d} | Loss: {loss:.4f} | User Acc: {u_acc:.4f} | News Acc: {n_acc:.4f}")

# --- 7. SAVE PREDICTIONS (Simplified) ---
model.eval()
with torch.no_grad():
    final_out = model(data.x_dict, data.edge_index_dict)
    
    for node_type, filename in zip(['user', 'news'], ['user_preds_final.csv', 'news_preds_final.csv']):
        mask = data[node_type].test_mask
        preds = final_out[node_type].argmax(dim=1)[mask].cpu().numpy()
        actual = data[node_type].y[mask].cpu().numpy()
        
        df = pd.DataFrame({'Predicted': preds, 'Actual': actual})
        df.to_csv(filename, index=False)
        print(f"Saved {node_type} predictions to {filename}")