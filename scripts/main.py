import argparse
# Create the parser
import logging


from utils import *
import gc
import torch
import math


OPTIMIZERS = ['adam', 'sgd']
SPLITS = ['iid', 'non_iid']

def read_options():
    ''' Parse command line arguments or load defaults '''
    return {
        'optimizer': 'adam',
        'data_split': 'iid',
        'num_rounds': 100,
        'clients_per_round': 10,
        'batch_size': 5,
        'num_epochs': 5,
        'learning_rate': 0.003,
        'poison': 20
    }

def main():
    # Read options
    parsed = read_options()

    # Print options
    print("\nOPTIMIZER:", parsed["optimizer"])
    print("Data Split:", parsed["data_split"])
    print("COMM_ROUNDS:", parsed["num_rounds"])
    print("POISON CLIENTS:", parsed["clients_per_round"])
    print("BATCH SIZE:", parsed["batch_size"])
    print("LOCAL EPOCHS:", parsed["num_epochs"])
    print("LEARNING RATE:", parsed["learning_rate"])
    print("POISON LEVEL:", parsed["poison"])

    poison_level = parsed["poison"]
    num_clients = parsed["clients_per_round"]
    poison_clients = math.ceil(num_clients * (poison_level / 100))
    print(poison_clients)
    if poison_clients > num_clients:
        logging.error("Poison level cannot be greater than 100 %")
    else:
        train(batch_size=parsed["batch_size"], poison=poison_clients, data_split=parsed["data_split"],
              optimizer=parsed["optimizer"], comm_rounds=parsed["num_rounds"], local_epochs=parsed["num_epochs"],
              lr=parsed["learning_rate"], num_clients=parsed["clients_per_round"])

if __name__ == '__main__':
    gc.collect()
    torch.cuda.empty_cache()
    main()

    

import pickle
import matplotlib.pyplot as plt
import pandas as pd

# Load data from pickle files
with open('acc-10-epoch-local-1.pickle', 'rb') as w:
    data_acc = pickle.load(w)

with open('loss-10-epoch-local-1.pickle', 'rb') as s:
    data_loss = pickle.load(s)

# Create DataFrames to store the accuracy and loss values
df_acc = pd.DataFrame(data_acc)
df_loss = pd.DataFrame(data_loss)

# Plot both accuracy and loss graphs side by side
fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(15, 6))

# Plot accuracy graph
for client in df_acc.columns:
    axes[0].plot(df_acc.index, df_acc[client], label=client)

axes[0].set_title('Accuracy vs Communication Round')
axes[0].set_xlabel('Communication Round')
axes[0].set_ylabel('Accuracy')
axes[0].legend()
axes[0].grid(True)

# Plot loss graph
for client in df_loss.columns:
    axes[1].plot(df_loss.index, df_loss[client], label=client)

axes[1].set_title('Loss vs Communication Round')
axes[1].set_xlabel('Communication Round')
axes[1].set_ylabel('Loss')
axes[1].legend()
axes[1].grid(True)

plt.tight_layout()
plt.show()
