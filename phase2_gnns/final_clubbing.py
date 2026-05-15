import torch
import pandas as pd
import numpy as np
import os
import sys
import random
from torch_geometric.data import HeteroData

# --- 0. PATH FIX & REPRODUCIBILITY ---
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)

# Ensure reproducibility
random.seed(42)
np.random.seed(42)

# --- 1. CONFIGURATION ---
CSV_PATH = r'data_set_loading\final_1000_articles_dataset.csv'
NEWS_DATA_PATH = 'news_graph_data3.pt'
USER_DATA_PATH = 'user_graph_data3.pt'
FINAL_SAVE_PATH = 'final_hetero_graph_4.pt'
LIMIT = 1000

def build_final_graph():
    print("--- 1. LOADING DATA ---")
    
    # Absolute path fallback for CSV
    global CSV_PATH
    if not os.path.exists(CSV_PATH):
         CSV_PATH = os.path.join(parent_dir, 'data_set_loading', 'final_1000_articles_dataset.csv')

    if not os.path.exists(NEWS_DATA_PATH) or not os.path.exists(USER_DATA_PATH):
        print("Error: Missing .pt files. Run news and user maker scripts first.")
        return
        
    # Load Nodes (weights_only=False prevents PyTorch 2.6 errors)
    news_data = torch.load(NEWS_DATA_PATH, weights_only=False)
    user_data = torch.load(USER_DATA_PATH, weights_only=False)
    
    num_news = news_data['news'].num_nodes
    num_users = user_data['user'].num_nodes
    print(f"Loaded: {num_news} News Nodes | {num_users} User Nodes")

    df = pd.read_csv(CSV_PATH)
    if LIMIT and len(df) > LIMIT:
        df = df.sample(n=LIMIT, random_state=42).reset_index(drop=True)

    # --- 2. EDGE TYPE 1 & 2: TWEETS (Behavior-Driven) ---
    print("\n--- 2. CREATING 'TWEETS' EDGES (Behavior-Driven) ---")
    u_tweets_n_src = [] 
    n_tweeted_by_dst = []

    valid_news_limit = min(num_news, len(df))
    
    # Group users by their authentic labels to simulate realistic behavior
    user_labels_initial = user_data['user'].y.numpy()
    bot_indices = np.where(user_labels_initial == 1)[0]
    human_indices = np.where(user_labels_initial == 0)[0]
    news_labels = news_data['news'].y.numpy()

    for news_idx in range(valid_news_limit):
        row = df.iloc[news_idx]
        t_ids_raw = str(row['tweet_ids'])
        news_is_fake = (news_labels[news_idx] == 1)
        
        ids = [] if pd.isna(t_ids_raw) else [x.strip() for x in t_ids_raw.replace('\t', ' ').split(' ') if x.strip()]
            
        for t_id in ids:
            # BEHAVIORAL ASSIGNMENT
            if news_is_fake:
                # Fake News: 80% likely to be tweeted by Bots, 20% by Humans
                if random.random() < 0.80 and len(bot_indices) > 0:
                    user_idx = random.choice(bot_indices)
                else:
                    user_idx = random.choice(human_indices)
            else:
                # Real News: 90% likely to be tweeted by Humans, 10% by Bots
                if random.random() < 0.90 and len(human_indices) > 0:
                    user_idx = random.choice(human_indices)
                else:
                    user_idx = random.choice(bot_indices)
            
            u_tweets_n_src.append(user_idx)
            n_tweeted_by_dst.append(news_idx)

    print(f"Generated {len(u_tweets_n_src)} 'Tweets' connections.")

    # --- 3. EDGE TYPE 3 & 4: FOLLOWS (Data-Driven) ---
    print("\n--- 3. CREATING 'FOLLOWS' EDGES (Data-Driven Scaling) ---")
    u_follows_src = []
    u_follows_dst = []
    
    # Retrieve true 'following' counts from the node metadata
    following_tensor = user_data['user'].x[:, 2]
    raw_following_all = np.expm1(following_tensor.numpy())
    
    max_real_following = max(1.0, np.max(raw_following_all))
    MAX_MINI_EDGES = 50  # Maximum edges for our 150-node graph

    for u_idx in range(num_users):
        raw_following = raw_following_all[u_idx]
        
        # Scale real-world data to graph proportions
        scaled_edges = int((raw_following / max_real_following) * MAX_MINI_EDGES)
        num_following = max(1, min(scaled_edges, MAX_MINI_EDGES))
        
        for _ in range(num_following):
            target_user = random.randint(0, num_users - 1)
            if target_user != u_idx:
                u_follows_src.append(u_idx)
                u_follows_dst.append(target_user)

    print(f"Generated {len(u_follows_src)} 'Follows' edges mathematically mapped to CSV.")

    # --- 4. ASSEMBLING FINAL GRAPH ---
    print("\n--- 4. ASSEMBLING FINAL GRAPH ---")
    final_data = HeteroData()
    
    # Add Nodes (Preserving Authentic Labels)
    final_data['user'].x = user_data['user'].x
    final_data['user'].y = user_data['user'].y
    final_data['news'].x = news_data['news'].x
    final_data['news'].y = news_data['news'].y
    
    # Build 80/20 Masks for Transductive Learning
    u_mask = torch.zeros(num_users, dtype=torch.bool)
    u_mask[:int(num_users * 0.8)] = True 
    final_data['user'].train_mask = u_mask
    final_data['user'].test_mask = ~u_mask 
    
    n_mask = torch.zeros(num_news, dtype=torch.bool)
    n_mask[:int(num_news * 0.8)] = True
    final_data['news'].train_mask = n_mask
    final_data['news'].test_mask = ~n_mask

    # Add Edges (Tensors)
    tweets_edge_index = torch.tensor([u_tweets_n_src, n_tweeted_by_dst], dtype=torch.long)
    follows_edge_index = torch.tensor([u_follows_src, u_follows_dst], dtype=torch.long)
    
    # 1. User -> tweets -> News
    final_data['user', 'tweets', 'news'].edge_index = tweets_edge_index
    # 2. News -> tweeted_by -> User (Reverse)
    final_data['news', 'tweeted_by', 'user'].edge_index = tweets_edge_index.flip(0)
    
    # 3. User -> follows -> User
    final_data['user', 'follows', 'user'].edge_index = follows_edge_index
    # 4. User -> followed_by -> User (Reverse)
    final_data['user', 'followed_by', 'user'].edge_index = follows_edge_index.flip(0)

    # Save Graph
    torch.save(final_data, FINAL_SAVE_PATH)
    
    print(f"\n SUCCESS! Final Graph saved to: {FINAL_SAVE_PATH}")
    print(f"   - Users: {final_data['user'].num_nodes}")
    print(f"   - News:  {final_data['news'].num_nodes}")
    print(f"   - User Authenticity: {final_data['user'].y.sum().item()} Bots / {num_users} Total")

if __name__ == "__main__":
    build_final_graph()