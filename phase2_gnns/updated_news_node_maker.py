import torch
import pandas as pd
import numpy as np
import os
import sys
import json
from torch_geometric.data import HeteroData
from sentence_transformers import SentenceTransformer

# --- 0. PATH FIX (Crucial for Imports) ---
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
phase1_dir = os.path.join(parent_dir, 'phase1')

if parent_dir not in sys.path: sys.path.append(parent_dir)
if phase1_dir not in sys.path: sys.path.append(phase1_dir)
    
# --- 1. IMPORT PIPELINE ---
try:
    import phase1.llm_pipeline_groq_v4_streamlit as pipeline
    print("Successfully imported phase1.llm_pipeline_groq_v4_streamlit")
except ImportError:
    try:
        import llm_pipeline_groq_v4_streamlit as pipeline
        print("Successfully imported llm_pipeline_groq_v4_streamlit (Direct)")
    except ImportError as e:
        print(f"Error: Could not import pipeline. \n   Looking in: {sys.path}")
        exit()

# --- 2. CONFIGURATION ---
BERT_MODEL_NAME = 'all-MiniLM-L6-v2'
bert_embedder = SentenceTransformer(BERT_MODEL_NAME)

# --- 3. HELPER: FEATURE EXTRACTOR ---
def extract_normalized_features(text, headline, url):
    """
    Calls functions from 'llm_pipeline_groq_v4_streamlit' and normalizes outputs to 0.0-1.0
    """
    try:
        read_res = pipeline.analyze_readability(text)
        flesch = read_res.get('flesch_reading_ease', 50.0)
        feat_readability = max(0.0, min(1.0, flesch / 100.0))
    except: feat_readability = 0.5

    try:
        headline_for_clickbait = (headline or "").strip() or (text[:200] if text else "No headline")
        click_res = pipeline.analyze_clickbait(headline_for_clickbait)
        feat_clickbait = float(click_res.get('clickbait_score', 0.0))
    except: feat_clickbait = 0.0

    try:
        sent_res = pipeline.analyze_sentiment(text)
        scores = sent_res.get('scores', {})
        feat_sentiment = scores.get('NEGATIVE', 0.0) 
        if feat_sentiment == 0.0 and 'POSITIVE' in scores:
             feat_sentiment = 1.0 - scores['POSITIVE']
    except: feat_sentiment = 0.5

    try:
        subj_res = pipeline.analyze_subjectivity(text[:3000])
        val = subj_res.get('score', subj_res.get('subjectivity_score', 0.5))
        feat_subj_score = float(val) if isinstance(val, (int, float)) else 0.5
        
        cat_str = str(subj_res.get('final_category', 'Objective')).lower()
        feat_subj_type = 1.0 if 'subjective' in cat_str else 0.0
    except: 
        feat_subj_score = 0.5
        feat_subj_type = 0.0

    try:
        hedge_res = pipeline.analyze_hedging(text[:3000])
        val = hedge_res.get('hedging_score', 0.0)
        feat_hedging = float(val) if isinstance(val, (int, float)) else 0.0
    except: feat_hedging = 0.0

    try:
        pr_score = pipeline.get_pagerank(url)
        feat_pagerank = pr_score / 10.0 
    except: feat_pagerank = 0.0

    features = np.array([
        feat_readability, feat_clickbait, feat_sentiment, 
        feat_subj_score, feat_subj_type, feat_hedging, feat_pagerank
    ], dtype=np.float32)

    return features

# --- 4. MAIN DATASET BUILDER (WITH CSV & PT CHECKPOINTING) ---

def build_gnn_dataset_from_csv(csv_path, save_path='news_graph_data3.pt', csv_output_path='extracted_news_features3.csv', limit=1000):
    print(f"--- Loading Dataset: {csv_path} ---")
    df = pd.read_csv(csv_path)
    
    if limit is not None:
        print(f"Configuration: Limiting to {limit} random articles.")
        if len(df) > limit:
            df = df.sample(n=limit, random_state=42).reset_index(drop=True)
    else:
        print(f"Configuration: Processing ALL {len(df)} articles.")
    
    all_features = []
    all_labels = []
    all_csv_data = [] # NEW: List to hold data for our CSV export
    successful_indices = []
    
    print(f"Starting processing loop (Saving after EVERY article)...")
    
    for i, (idx, row) in enumerate(df.iterrows()):
        url = str(row['news_url'])
        label_raw = str(row['dataset_label'])
        
        print(f"\n[{i+1}/{len(df)}] Processing: {url[:40]}...")
        
        try:
            # 1. Scrape
            article_data = pipeline.get_article_content(url)
            text = article_data.get('text', "")
            headline = article_data.get('headline', "")
            
            if not text or len(text) < 50:
                print(f"   ⚠️ Content too short. Skipping.")
                continue 
            if len(text) > 10000:
                print(f"   ⚠️ Content too long ({len(text)}). Skipping.")
                continue
            # 2. Features
            bert_emb = bert_embedder.encode(text[:1000])
            style_feat = extract_normalized_features(text, headline, url)
            combined = np.concatenate([bert_emb, style_feat])
            
            # 3. Label
            label_str = str(label_raw).lower().strip()
            if 'fake' in label_str: label = 1
            elif 'real' in label_str: label = 0
            elif label_str == '1': label = 1
            elif label_str == '0': label = 0
            else: label = 0
            
            # 4. Append to Graph Lists
            all_features.append(combined)
            all_labels.append(label)
            successful_indices.append(idx)
            
            # --- NEW: Append to CSV Data List ---
            row_dict = {
                'original_index': idx,
                'news_url': url,
                'label': label,
                'readability_score': style_feat[0],
                'clickbait_score': style_feat[1],
                'sentiment_negative': style_feat[2],
                'subjectivity_score': style_feat[3],
                'subjectivity_type': style_feat[4],
                'hedging_score': style_feat[5],
                'pagerank_score': style_feat[6],
                # Save BERT embedding as a JSON string so it doesn't create 384 ugly columns
                'bert_embedding_384d': json.dumps(bert_emb.tolist()) 
            }
            all_csv_data.append(row_dict)
            
            # --- B. SAVE CHECKPOINTS ---
            
            # Save Graph Checkpoint
            temp_data = HeteroData()
            temp_data['news'].x = torch.tensor(np.array(all_features), dtype=torch.float)
            temp_data['news'].y = torch.tensor(all_labels).long()
            
            curr_len = len(all_labels)
            mask = torch.zeros(curr_len, dtype=torch.bool)
            mask[:int(curr_len * 0.8)] = True
            temp_data['news'].train_mask = mask
            temp_data['news'].test_mask = ~mask
            
            torch.save(temp_data, save_path)
            
            # Save CSV Checkpoint
            pd.DataFrame(all_csv_data).to_csv(csv_output_path, index=False)
            
            print(f"   💾 Checkpoints Saved! ({curr_len} articles in '{save_path}' and '{csv_output_path}')")
                
        except Exception as e:
            print(f"   ❌ Error: {e}")
            continue

    if not all_features:
        print("Error: No articles were successfully processed.")
        return None

    print(f"\n✅ FINAL SUCCESS!")
    print(f"Total Processed: {len(all_labels)} articles")
    print(f"Final Graph Saved to: {save_path}")
    print(f"Final CSV Saved to: {csv_output_path}")
    
    return torch.load(save_path, weights_only=False)

# --- EXECUTION ---
if __name__ == "__main__":
    csv_location = r'data_set_loading\final_1000_articles_dataset.csv'
    
    if not os.path.exists(csv_location):
        csv_location = os.path.join(parent_dir, 'data_set_loading', 'final_1000_articles_dataset.csv')
    
    build_gnn_dataset_from_csv(
        csv_location, 
        limit=None 
    )