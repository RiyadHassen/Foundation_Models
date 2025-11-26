import os
import pickle
import torch
import numpy as np
import re
dataset_path = 'data/reward_fineweb.txt'
#open a file and split each line by \n and store it in a list
with open(dataset_path, 'r') as f: 
    data = f.read().splitlines()

def reward_fn(line):
    # reward function: base is proportional to text length, but
    # downweight lines that contain contracted words (e.g. "don't", "'ll", "I'm").
    # This encourages more explicit (non-contracted) responses.
    L = len(line)

    if L <=20.0:
        return 2.0 
    
    frac = (L - 20)/ 80
    penalty = frac ** 1.5
    # patterns that indicate contractions; case-insensitive
    #contraction_re = re.compile(r"(?i)\b(?:\w+'(?:\w+|t)|'(?:ll|re|ve|d|m|s|t))\b")
    #num_contractions = len(contraction_re.findall(line))
    # reduce weight by 25% per contraction('ll he's i'm) , but keep a minimum multiplier
    score = max(0.2, 2.0 - penalty)
    return score

reward_scores = [reward_fn(line)  for line in data if len(line) > 0]
dataset = [line for line in data if len(line) > 0]


for d, r in zip(dataset[:10], reward_scores[:10]):
    print(f"Line: {d}  Reward: {r:.2f}")

print(f"Sample data line: {data[0]} Reward score: {reward_scores[0]}")
# calculate number of vocab
chars = sorted(set("".join(data)))
vocab_size = len(chars)

# tokenize the data

stoi = { ch:i for i,ch in enumerate(chars)}
itos = { i:ch for i,ch in enumerate(chars)}


def encode(s):
    return [stoi[c] for c in s] # encoder: take a string, output a list of integers
def decode(l):
    return ''.join([itos[i] for i in l]) # decoder: take a list of integers, output a string

train_data = dataset[:int(0.9*len(dataset))]
val_data = dataset[int(0.9*len(dataset)):]

train_reward = reward_scores[:int(0.9*len(reward_scores))]
val_reward = reward_scores[int(0.9*len(reward_scores)):]

tokenized_train = [ encode(line) for line in train_data]
tokenized_val = [ encode(line) for line in val_data]

print(f"tokenized train sample:{tokenized_train[0]}")
print(f"reward score sample:{train_reward[0]}")

print(f"Number of training samples: {len(tokenized_train)}")
print(f"Number of validation samples: {len(tokenized_val)}")

# 
class RewardDataset(torch.utils.data.Dataset):
    def __init__(self, tokenized_data, reward_scores):
        self.tokenized_data = tokenized_data
        self.reward_scores = reward_scores

    def __len__(self):
        return len(self.tokenized_data)

    def __getitem__(self, idx):
        tokens = torch.tensor(self.tokenized_data[idx], dtype=torch.long)
        reward = torch.tensor(self.reward_scores[idx], dtype=torch.float)
        return tokens, reward



rewardDataset = RewardDataset(tokenized_train, train_reward)
valRewardDataset = RewardDataset(tokenized_val, val_reward)

# Save the PyTorch Dataset objects so they can be loaded easily later with `torch.load`.
save_dir = os.path.dirname(__file__)
torch.save(rewardDataset, os.path.join(save_dir, 'train_reward_dataset.pt'))
torch.save(valRewardDataset, os.path.join(save_dir, 'val_reward_dataset.pt'))

# Also save tokenized arrays and reward scores in a compact NPZ for quick array-based loading.
#np.savez_compressed(os.path.join(save_dir, 'train_data.npz'), tokens=tokenized_train, rewards=train_reward)
#np.savez_compressed(os.path.join(save_dir, 'val_data.npz'), tokens=tokenized_val, rewards=val_reward)

# save the meta information as well, to help us encode/decode later
meta = {
    'vocab_size': vocab_size,
    'itos': itos,
    'stoi': stoi,
}
with open(os.path.join(os.path.dirname(__file__), 'meta.pkl'), 'wb') as f:
    pickle.dump(meta, f)
    

