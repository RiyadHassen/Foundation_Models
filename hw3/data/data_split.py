
import os
import random

file_path = os.path.join(os.path.abspath(os.path.dirname(__file__)), 'finetune.txt')

def split_random_chunks(line, max_len=100):
    chunks = []
    i = 0
    n = len(line)
    while i < n:
        # pick a random chunk length between 20 and max_len
        chunk_size = random.randint(20, max_len)
        chunks.append(line[i:i+chunk_size])
        i += chunk_size
    return chunks

all_chunks = []

with open(file_path, 'r', encoding='utf-8') as f:
    lines = f.readlines()

for line in lines:
    line = line.rstrip("\n")
    # skip empty lines
    if not line.strip():
        continue

    chunks = split_random_chunks(line, max_len=100)
    all_chunks.extend(chunks)

with open('reward_fineweb.txt','w', encoding='utf-8') as f:
    for chunk in all_chunks:
        f.write(chunk + '\n')   

print("Example chunks:", all_chunks[:10])
