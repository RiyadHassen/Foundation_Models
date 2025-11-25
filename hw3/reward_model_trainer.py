import os
import pickle
import numpy as np
import torch
import torch.nn.functional as F
from model import GPT, GPTConfig
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
import prepare_dataset
import yaml

class RewardModelTrainer:
    """
    Reward Model Trainer
    """
    @staticmethod
    def train(model, train_loader, val_loader, config):
        """
        Train the reward model using the provided datasets and configuration.

        Args:
            model: The reward model to be trained.
            train_dataset: The training dataset.
            val_dataset: The validation dataset.
            config: A dictionary containing training configurations.
        """
        device = config['device']
        model.to(device)
        
        optimizer = torch.optim.AdamW(model.parameters(), lr=config['learning_rate'])
        for i in range(config['max_iters']):
            model.train()
            train_loss = 0
            num_train_batches = 0
            for tokens, reward in train_loader:
                # Forward pass
                # for each batch, we have two sequences: chosen and rejected
                # print("batch training:", batch)
                # chosen_outputs = model(batch[0])
                # rejected_outputs = model(batch[1])
                # Compute loss
                #bradlytaly loss function for reward model
                #loss = - F.sigmoid(chosen_outputs - rejected_outputs).mean()

                #for scalar reward model
                tokens = tokens.to(device)
                reward = reward.to(device)
                pred_reward = model(tokens)
                loss = F.mse_loss(pred_reward, reward)
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
                train_loss += loss.item()
                num_train_batches += 1

                if (num_train_batches) % 20 == 0:
                    avg_train_loss = train_loss / num_train_batches if num_train_batches > 0 else 0
                    print(f"Iteration {i+1}, Batch {num_train_batches}, Avg Training Loss: {avg_train_loss:.6f}")

            # Validation loop
            model.eval()
            val_loss = 0
            num_batches = 0
            with torch.no_grad():
                for tokens, reward in val_loader:
                    tokens = tokens.to(config['device'])
                    reward = reward.to(config['device'])
                    val_reward = model(tokens)
                    val_loss += F.mse_loss(val_reward, reward).mean().item()
                    num_batches += 1
            if (i+1) % 20 == 0:
                #print(f"Epoch {i+1}/{config['max_iters']}, Validation Loss: {val_loss/len(val_dataset)}")
                avg_val_loss = val_loss / num_batches if num_batches > 0 else 0
                print(f"Epoch  {i+1}/{config['max_iters']}, Validation Loss: {avg_val_loss:.6f}")


        # at last save the model
        torch.save(model.state_dict(), os.path.join(os.path.dirname(__file__), 'reward_model.pth'))

if __name__ == "__main__":

    with open('config/config_reward.yaml') as f:
        conf = yaml.load(f, Loader=yaml.FullLoader)
        # nested dictionary structure
        config = {}               
        for k, v in conf.items():
            for k2, v2 in v.items():
                config[k2] = v2
    print(config)

    

   
    base_dir = os.path.dirname(__file__)
    train_path = os.path.join(base_dir, 'train_reward_dataset.pt')
    val_path = os.path.join(base_dir, 'val_reward_dataset.pt')

    print("===== Loading dataset =====")
    meta_path = os.path.join(base_dir, 'meta.pkl')
    train_dataset = torch.load(train_path)
    val_dataset = torch.load(val_path)

    # load meta (stoi/itos/vocab_size) if you need it
    with open(meta_path, 'rb') as f:
        meta = pickle.load(f)
    vocab_size = meta['vocab_size']


    config['device'] = 'cuda' if torch.cuda.is_available() else 'cpu'
    config['vocab_size'] = vocab_size
    print(f"Using device: {config['device']}")
    print(f"Vocab size: {vocab_size}, Block size: {config['block_size']}")

    # reward model # model init

    def collate_fn(batch):
        
        tokens_list, rewards_list = zip(*batch)
        block_size = config['block_size']

        filtered = [(t, r) for t, r in zip(tokens_list, rewards_list) if t is not None and t.size(0) > 0]
        if len(filtered) == 0:
            raise ValueError("All sequences in the batch are empty.")
        tokens_list, rewards_list = zip(*filtered)
        # keep rightmost recent tokens and truncate to block_size
        tokens_list = [t[-block_size:] if t.size(0) > block_size else t for t in tokens_list]
        tokens_padded = pad_sequence(tokens_list, batch_first=True, padding_value=0)  # shape (B, Lmax)
        
        # ensure no token exceeds vocab_size - 1
        #tokens_padded = torch.clamp(tokens_padded, max=vocab_size- 1)
        tokens_padded = tokens_padded
        rewards = torch.stack(rewards_list)
        
        #print(f"DEBUG collate: tokens shape {tokens_padded.shape}, max token {tokens_padded.max().item()}, device {tokens_padded.device}")
        return tokens_padded, rewards

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, collate_fn=collate_fn)

    # trainer model
    model_args = dict(n_layer=config['n_layer'], n_head=config['n_head'], n_embd=config['n_embd'], block_size=config['block_size'],
                    bias=config['bias'], vocab_size=vocab_size, dropout=config['dropout'], ) # start with model_args from command line
    model = GPT(GPTConfig(**model_args))
    reward_trainer = RewardModelTrainer.train(model, train_loader, val_loader, config)


