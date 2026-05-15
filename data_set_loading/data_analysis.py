import gc
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import os
import pandas as pd
import random
import seaborn as sns
import torch
import torch.nn.functional as F

from sklearn.metrics import accuracy_score, confusion_matrix, confusion_matrix, classification_report, f1_score, precision_score, recall_score, roc_auc_score, roc_curve
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from torch.nn import Linear, Dropout, BatchNorm1d
from dataset_loader import (
    df_gossip_fake,
    df_politic_fake,
    df_gossip_real,
    df_politic_real
)

df_gossip_fake = pd.read_csv('gossipcop_fake.csv')
df_politic_fake = pd.read_csv('politifact_fake.csv')
df_gossip_real = pd.read_csv('gossipcop_real.csv')
df_politic_real = pd.read_csv('politifact_real.csv')

df_gossip_fake['label'] = 1
df_politic_fake['label'] = 1
df_gossip_real['label'] = 0
df_politic_real['label'] = 0

df = pd.concat([df_gossip_fake, df_politic_fake, df_gossip_real, df_politic_real], ignore_index=True)

df.info()

class_distribution_percent = df['label'].value_counts(normalize=True) * 100

plt.figure(figsize=(4, 4))
sns.countplot(x=df['label'], hue=df['label'], palette='pastel')
plt.title('Class Distribution', fontsize=14)
plt.xlabel('Class')
plt.ylabel('Number of Samples')
plt.xticks([0, 1], [f'{round(class_distribution_percent[0])}%', f'{round(class_distribution_percent[1])}%'])
plt.show()