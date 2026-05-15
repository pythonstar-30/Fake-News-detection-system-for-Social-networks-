import torch
import pandas as pd
import numpy as np
import random
from faker import Faker
from sklearn.preprocessing import LabelEncoder
from torch_geometric.data import HeteroData

# --- 1. CONFIGURATION ---
NUM_USERS = 9000
SAVE_PATH = 'user_graph_data3.pt'
CSV_SAVE_PATH = 'synthetic_users3.csv'
SEED = 42

Faker.seed(SEED)
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
fake = Faker()

# --- 2. GENERATOR: REALISTIC SYNTHETIC USERS ---
def generate_synthetic_users(num_users):
    print(f"--- Generating {num_users} Realistic Synthetic Users ---")
    users = []
    
    for uid in range(num_users):
        # 30% Bots, 70% Humans
        is_bot = 1 if random.random() < 0.3 else 0
        
        if is_bot:
            # --- STEALTH BOT PERSONA ---
            # 15% of bots are 'Sleeper/Hijacked' accounts (Old and look human)
            if random.random() < 0.15:
                account_age_days = random.randint(400, 2500)
                followers = random.randint(100, 2000)
                location = fake.country() # Stealth bots use real locations
            else:
                account_age_days = random.randint(1, 120) # Younger but overlapping
                followers = random.randint(0, 150)
                location = "Unknown" if random.random() < 0.7 else fake.country()

            following = random.randint(200, 3000)
            verified = 0 # Most bots aren't verified
        else:
            # --- REALISTIC HUMAN PERSONA ---
            # 10% of humans are 'New Users' (Look like bots!)
            if random.random() < 0.10:
                account_age_days = random.randint(1, 60)
                followers = random.randint(0, 50)
                location = "Unknown" if random.random() < 0.3 else fake.country()
            else:
                account_age_days = random.randint(150, 3650)
                followers = random.randint(50, 15000)
                location = fake.country()

            following = random.randint(20, 1200)
            verified = 1 if random.random() < 0.08 else 0
            
        users.append({
            'user_id': uid,
            'name': fake.name(),
            'account_age_days': account_age_days,
            'followers': followers,
            'following': following,
            'verified': verified,
            'location': location,
            'label': is_bot 
        })
        
    return pd.DataFrame(users)

# --- 3. MAIN DATASET BUILDER ---
def build_user_nodes():
    df_users = generate_synthetic_users(NUM_USERS)
    df_users.to_csv(CSV_SAVE_PATH, index=False)
    
    print("--- Processing Features into Tensors with Noise ---")
    
    # Helper to add Gaussian noise to prevent 100% memorization
    def add_noise(t, std=0.02):
        return t + torch.randn_like(t) * std

    # Feature A: Account Age (Normalize to 10 years max)
    age_tensor = add_noise(torch.tensor(df_users['account_age_days'].values / 3650.0, dtype=torch.float).view(-1, 1))
    
    # Feature B & C: Followers & Following (Log Transform for scale)
    fol_tensor = add_noise(torch.tensor(np.log1p(df_users['followers'].values) / 10.0, dtype=torch.float).view(-1, 1))
    fwi_tensor = add_noise(torch.tensor(np.log1p(df_users['following'].values) / 10.0, dtype=torch.float).view(-1, 1))
    
    # Feature D: Verified
    ver_tensor = torch.tensor(df_users['verified'].values, dtype=torch.float).view(-1, 1)
    
    # Feature E: Location (Encoding + Noise)
    le = LabelEncoder()
    loc_indices = le.fit_transform(df_users['location'])
    loc_tensor = add_noise(torch.tensor(loc_indices / max(1, len(le.classes_)), dtype=torch.float).view(-1, 1))
    
    # Stack features [NUM_USERS, 5]
    user_features = torch.cat([age_tensor, fol_tensor, fwi_tensor, ver_tensor, loc_tensor], dim=1)
    user_labels = torch.tensor(df_users['label'].values).long()
    
    # --- 4. CREATE HETERODATA & RANDOMIZED MASKS ---
    data = HeteroData()
    data['user'].x = user_features
    data['user'].y = user_labels
    
    # RANDOMIZED MASKING (CRITICAL for valid reports)
    indices = torch.randperm(NUM_USERS)
    train_size = int(NUM_USERS * 0.8)
    
    train_mask = torch.zeros(NUM_USERS, dtype=torch.bool)
    test_mask = torch.zeros(NUM_USERS, dtype=torch.bool)
    
    train_mask[indices[:train_size]] = True
    test_mask[indices[train_size:]] = True
    
    data['user'].train_mask = train_mask
    data['user'].test_mask = test_mask
    
    torch.save(data, SAVE_PATH)
    print(f"\nSUCCESS! Saved realistic graph to {SAVE_PATH}")
    print(f"Test Set Size: {test_mask.sum().item()} users")
    return data

if __name__ == "__main__":
    build_user_nodes()