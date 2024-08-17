# -*- coding: utf-8 -*-
"""Copy of QuantumCompression_in_GPTs.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1tacnNSQdwOqzArIx9X7lPdED27S86wEd
"""

# type: ignore
# %%
from typing import List, Dict, Optional, Tuple, Iterable
from pydantic import BaseModel
from torch.utils.data import Dataset, DataLoader

import matplotlib.pyplot as plt
import numpy as np
import pathlib
from sklearn.linear_model import LinearRegression
from sklearn.decomposition import PCA
import seaborn as sns
from transformer_lens import HookedTransformer, HookedTransformerConfig
from tqdm.notebook import tqdm  # for simply implemented progress bar

import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.utils.data import IterableDataset, DataLoader

#from epsilon_transformers.process.processes import Mess3
from epsilon_transformers.process.dataset import ProcessDataset
from epsilon_transformers.analysis.activation_analysis import get_beliefs_for_transformer_inputs
from epsilon_transformers import training
from epsilon_transformers.visualization.plots import _project_to_simplex

train_config = {
    'seed': 42,
    'n_ctx': 10,
    'act_fn': 'relu',
    'd_head': 16,
    'd_model': 256,
    'd_vocab': 2,
    'n_heads': 4,
    'n_layers': 10,
    'attn_only': False,
    'optimizer': 'adam',
    'batch_size': 1000,
    'num_epochs': 1000,
    'weight_decay': 0.0,
    'attention_dir': 'causal',
    'learning_rate': 1e-3,
    'normalization_type': 'LN',
    'fixed_train_dataset': False,
    'num_train_samples': 65238,
}
print("\n".join(f"{k}: {v}" for k, v in train_config.items()))

# change to   'optimizer': 'sgd', ?

"""### Example: FRDN qutrit process with no finite HMM

Observable alphabet:
$\mathcal{X} = \{ a, b \}$

Classical latent states:
$\mathcal{S} = \mathbb{N}_{\geq 0}$

$\quad \Pr(S_{t+1} = \ell | S_{t} = 0) = f_\ell = \lambda^\ell \sin^2(\ell \alpha / 2)$ for $\ell > 0$, $f_0 = 1 - \sum_{\ell>0} f_\ell$, while

$\quad \Pr(S_{t+1} = \ell-1 | S_t = \ell) = 1$ for $\ell > 0$

$\alpha \in \mathbb{R}$, $0 < \lambda \leq 1/2$.

Function of the Markov chain:

$\quad 0 \mapsto a$,

$\quad \ell \mapsto b$ if $\ell > 0$.

No finite-dimensional HMM when $\alpha / \pi$ is irrational.

## Choose parameter setting:
"""

alpha = 3.4 #1./2
r = 1.
lamb = 1./3

alpha = 2000
r = 1.
lamb = 0.49

from numpy import array, cos, linalg, log, ones, outer, pi, random, sin, zeros

a_la = (1-lamb*cos(alpha) + lamb*sin(alpha))/(1-2*lamb*cos(alpha) + lamb**2)
b_la = (1-lamb*cos(alpha) - lamb*sin(alpha))/(1-2*lamb*cos(alpha) + lamb**2)

tau = ones(4)

# The reset distribution:
pi0 = array([1-(2/(1-lamb) - a_la - b_la)/4, 1/(2*(1-lamb)), -a_la/4, -b_la/4])
pi0

w = array([1, 1-lamb, 1+lamb*(sin(alpha) - cos(alpha)), 1-lamb*(sin(alpha) + cos(alpha))])
Da = outer(w,pi0)

Da

# Note that there is a sine sign error in Fanizza's Eq. (27).  Fixed here.
Db = lamb*array([[0,0,0,0],[0,1,0,0],[0,0,cos(alpha),-sin(alpha)],[0,0,sin(alpha),cos(alpha)]])
Db

D = Da + Db
D

D @ tau

right_eigstuff = linalg.eig(D)
left_eigstuff = linalg.eig(D.T)

left_eigstuff

for index, eigval in enumerate(left_eigstuff[0]):
    if abs(eigval - 1.) < 0.0000001:
        stationary = left_eigstuff[1][:,index]
        stationary /= stationary.sum()  # to normalize in probability
        stationary = stationary.real
print(stationary)

stationary @ tau

stationary @ D

def Db_tothe(ell):
    return array([[0, 0, 0, 0], [0, 1, 0, 0], [0, 0, cos(ell*alpha), -sin(ell*alpha)], [0, 0, sin(ell*alpha), cos(ell*alpha)] ])

import numpy as np
data_3d_colored = array([stationary, pi0] + [(pi0 @ Db_tothe(ell) ) / (pi0 @ Db_tothe(ell) @ tau) for ell in range(1,train_config['n_ctx']+1)] + [(stationary @ Db_tothe(ell) ) / (stationary @ Db_tothe(ell) @ tau) for ell in range(1,train_config['n_ctx']+1)])

msp_beliefs = [tuple(round(b, 5) for b in belief) for belief in data_3d_colored]
msp_belief_index = {b: i for i, b in enumerate(set(msp_beliefs))}

# Calculate mean and standard deviation
mean_3d_colored = np.mean(data_3d_colored, axis=1, keepdims=True)
std_3d_colored = np.std(data_3d_colored, axis=1, keepdims=True)

# Normalize feature values to [0, 1] for RGB coloring
min_vals = data_3d_colored.min(axis=0)
max_vals = data_3d_colored.max(axis=0)
scaled_colors = (data_3d_colored - min_vals) / (max_vals - min_vals)

viewpoint = np.array([0,0,0,0])

# Calculate distances from the origin (viewpoint) for depth effect
distances_original = np.linalg.norm(data_3d_colored - viewpoint, axis=1)
#distances_normalized = np.linalg.norm(normalized_data_3d_colored - viewpoint, axis=1)

# Scale sizes inversely with distance (normalize first to avoid very large sizes)
size_scale_original = 100 / (1 + distances_original)  # 500 is a scaling factor for visibility
#size_scale_normalized = 200 / (1 + distances_normalized)

fig = plt.figure(figsize=(6, 6))

ax1 = fig.add_subplot(111)
for i in range(len(data_3d_colored)):
    ax1.scatter(data_3d_colored[i, 2], data_3d_colored[i, 3],
                vmin=-2, vmax=2, color=scaled_colors[i], s=size_scale_original[i], alpha=0.7)

#plt.savefig('Prediction4.pdf')

plt.show()

#def Db_tothe(ell):
#    return array([[1, 0, 0], [0, cos(ell*alpha), -sin(ell*alpha)], [0, sin(ell*alpha), cos(ell*alpha)] ])

"""
import numpy as np
data_3d_colored = array([(pi0[1:] @ Db_tothe(ell) ) / (pi0[1:] @ Db_tothe(ell) @ tau[1:]) for ell in range(train_config['n_ctx'])] + [(stationary[1:] @ Db_tothe(ell) ) / (stationary[1:] @ Db_tothe(ell) @ tau[1:]) for ell in range(train_config['n_ctx'])])

msp_beliefs = [tuple(round(b, 5) for b in belief) for belief in data_3d_colored]
msp_belief_index = {b: i for i, b in enumerate(set(msp_beliefs))}

# Calculate mean and standard deviation
mean_3d_colored = np.mean(data_3d_colored, axis=1, keepdims=True)
std_3d_colored = np.std(data_3d_colored, axis=1, keepdims=True)

# Normalize feature values to [0, 1] for RGB coloring
min_vals = data_3d_colored.min(axis=0)
max_vals = data_3d_colored.max(axis=0)
scaled_colors = (data_3d_colored - min_vals) / (max_vals - min_vals)

viewpoint = np.array([-1,1,-1])

# Calculate distances from the origin (viewpoint) for depth effect
distances_original = np.linalg.norm(data_3d_colored - viewpoint, axis=1)
#distances_normalized = np.linalg.norm(normalized_data_3d_colored - viewpoint, axis=1)

# Scale sizes inversely with distance (normalize first to avoid very large sizes)
size_scale_original = 100 / (1 + distances_original)  # 500 is a scaling factor for visibility
#size_scale_normalized = 200 / (1 + distances_normalized)

fig = plt.figure(figsize=(6, 6))

ax1 = fig.add_subplot(111)
for i in range(len(data_3d_colored)):
    ax1.scatter(data_3d_colored[i, 1], data_3d_colored[i, 2],
                vmin=-2, vmax=2, color=scaled_colors[i], s=size_scale_original[i], alpha=0.7)

#plt.savefig('Prediction4.pdf')

plt.show()
"""

2**16

3**10



class GHMM:
    """
    base GHMM class
    Accomodates both Generalized HMMs (GHMMs) and standard HMMs

    GHMM(dict_labeledTMs, dict_tokens)
    --------------------------------------------

    It's assumed that the two dictionaries---for
    labeled (generalized) transition matrices and the tokens that induce them---
    have the same keys, which range from 0 to
    alphabet_size-1
    """
    def __init__(self, dict_labeledTMs, dict_tokens):
        self.transition_matrices = dict_labeledTMs
        self.tokens = dict_tokens
        self.int_from_token = {self.tokens[key]: key for key in self.tokens}
        self.latent_dim = dict_labeledTMs[0].shape[0] # dimension of latent space
        self.one = ones(self.latent_dim) # Note that this is not generally the right eigenstate of T for a GHMM
        self.alphabet_size = len(dict_tokens.keys())
        T = zeros((self.latent_dim, self.latent_dim))
        for x in dict_labeledTMs.keys():
            T+= dict_labeledTMs[x]
        self.T = T
        left_eigstuff = linalg.eig(T.transpose())
        for index, eigval in enumerate(left_eigstuff[0]):
            if abs(eigval - 1.) < 0.0000001:
                stationary = left_eigstuff[1][:,index]
                stationary /= stationary.sum()  # to normalize in probability
                self.stationary = stationary.real
        self.current_distr = self.stationary.copy()

    def sample(self):
        """
        update latent distribution while emmitting a symbol
        """
        #current_state_vec = zeros(self.latent_dim)
        #current_state_vec[self.current_state] = 1.
        x_t = random.choice(self.alphabet_size, p= [(self.current_distr @ self.transition_matrices[x]).sum() for x in range(self.alphabet_size)] )  # NOTE: assumes tau = self.one
        new_vec = self.current_distr @ self.transition_matrices[x_t]
        self.current_distr = new_vec / new_vec.sum()  # NOTE: assumes tau = self.one
        return self.tokens[x_t]

    def yield_emissions(self, sequence_len, reset_distr=False):
        if reset_distr: self.current_distr = self.stationary.copy()
        for _ in range(sequence_len):
            new_token = self.sample()
            yield new_token

    def evolve(self, token_key):
        """
        update latent distribution via given token
        """
        new_vec = self.current_distr @ self.transition_matrices[token_key]
        self.current_distr = new_vec / new_vec.sum()  # NOTE: assumes tau = self.one

    def probability(self, word, reset_distr=True):
        # returns the probability of the word, given the initial current_distr
        if reset_distr: self.current_distr = self.stationary.copy()
        prob = 1.
        for token in word:
            token_int = self.int_from_token[token]
            prob *= (self.current_distr @ self.transition_matrices[token_int]).sum()  # NOTE: assumes tau = self.one
            self.evolve(token_int)
        return prob

    def supports(self, word, min_prob=1E-12, reset_distr=True):
        if reset_distr: self.current_distr = self.stationary.copy()
        if self.probability(word) > min_prob:
            return True
        else:
            return False



#dict_tokens = {0: '0', 1:'1'}
#dict_tokens = {0: int(0), 1:int(1)}
dict_tokens = {0:0, 1:1}
dict_labeledTMs = {0: Da, 1: Db}

FRDN = GHMM(dict_labeledTMs, dict_tokens)

def sequence_to_belief(process, int_sequence):
    # assumes process is an instance of a GHMM, like FRDN
    process.current_distr = process.stationary.copy()
    for token_key in sequence:
        process.evolve(token_key)
    belief = process.current_distr.copy()
    return belief

def H2(p):
    # binary entropy function in nats
    if p == 0. or p == 1.:
        return 0.
    else:
        return -p*log(p) - (1.-p)*log(1.-p)

def FRDN_myopic_entropy_rate(through_L=16):
    W0 = zeros((2*through_L+2, 2*through_L+2))
    W1 = zeros((2*through_L+2, 2*through_L+2))
    # Initial through_L+1 will represent the \pi D_b^\ell states for \ell in 0:L
    # Subsequent through_L+1 will represent the \pi_0 D_b^\ell for \ell in 0:L
    next_token_entropy_vec = zeros(2*through_L+2)
    mixed_states = zeros((2*through_L+2,4))
    mixed_states[0,:] = FRDN.stationary.copy()
    for ell in range(through_L):
        prob_b = (mixed_states[ell,:] @ Db).sum()
        W1[ell,ell+1] = prob_b
        W0[ell,through_L+1] = 1 - prob_b
        next_token_entropy_vec[ell] = H2(prob_b)
        mixed_states[ell+1,:] = (mixed_states[ell,:] @ Db) / prob_b
    mixed_states[through_L+1] = pi0
    for ell in range(through_L+1, 2*through_L+1):
        prob_b = (mixed_states[ell,:] @ Db).sum()
        W1[ell,ell+1] = prob_b
        W0[ell,through_L+1] = 1 - prob_b
        next_token_entropy_vec[ell] = H2(prob_b)
        mixed_states[ell+1,:] = (mixed_states[ell,:] @ Db) / prob_b
    W = W0 + W1

    MSP_distr = zeros(2*through_L+2)
    MSP_distr[0] = 1.
    myopic_entropy_vec = zeros(through_L)
    myopic_entropy_vec[0] = MSP_distr @ next_token_entropy_vec
    for index in range(1,through_L):
        MSP_distr = MSP_distr @ W
        myopic_entropy_vec[index] = MSP_distr @ next_token_entropy_vec

    return array(myopic_entropy_vec)

myopic_entropy_rate = FRDN_myopic_entropy_rate(through_L=train_config['n_ctx']+1)
minimum_cross_entropy = myopic_entropy_rate[1:]
print(f"myopic_entropy_rate: {myopic_entropy_rate}")



def project_beliefs(belief_matrix):
    number_of_beliefs = belief_matrix.shape[0]
    belief_shape = belief_matrix.shape[1]
    assert belief_shape <= 4
    return ([belief_matrix[index, belief_shape-2] for index in range(number_of_beliefs)], [belief_matrix[index, belief_shape-1] for index in range(number_of_beliefs)])

ground_truth_projection = project_beliefs(np.array(list(msp_belief_index.keys())))
plt.figure(figsize=(4.5, 4))
#plt.scatter(ground_truth_projection[0], ground_truth_projection[1], c=list(msp_belief_index.keys()), alpha=.1, s=1)
plt.scatter(ground_truth_projection[0], ground_truth_projection[1], c = 'k', alpha=.4, s=2)
#plt.scatter(ground_truth_projection[0], ground_truth_projection[1], c = scaled_colors, alpha=.4, s=2)


plt.title("Ground Truth Simplex")
plt.gca().set_axis_off()
plt.show()





device = torch.device(
    "mps"
    if torch.backends.mps.is_available()
    else ("cuda" if torch.cuda.is_available() else "cpu")
)
print(f"Using device: {device}")

# we now have activations [batch, n_ctx, d_model]
# and we have transformer_input_beliefs [batch, n_ctx, belief_dim]
# and we have transformer_input_belief_indices [batch, n_ctx]

# in the end we want to do linear regression between the activations and the transformer_input_beliefs
def run_activation_to_beliefs_regression(activations, ground_truth_beliefs):

    # make sure the first two dimensions are the same
    assert activations.shape[0] == ground_truth_beliefs.shape[0]
    assert activations.shape[1] == ground_truth_beliefs.shape[1]

    # flatten the activations
    batch_size, n_ctx, d_model = activations.shape
    belief_dim = ground_truth_beliefs.shape[-1]
    activations_flattened = activations.view(-1, d_model) # [batch * n_ctx, d_model]
    ground_truth_beliefs_flattened = ground_truth_beliefs.view(-1, belief_dim) # [batch * n_ctx, belief_dim]

    # run the regression
    regression = LinearRegression()
    regression.fit(activations_flattened, ground_truth_beliefs_flattened)

    # get the belief predictions
    belief_predictions = regression.predict(activations_flattened) # [batch * n_ctx, belief_dim]
    belief_predictions = belief_predictions.reshape(batch_size, n_ctx, belief_dim)

    return regression, belief_predictions

msp_belief_index

# now lets set up all the inputs as they arrive into the transformer

process = FRDN

str_format = '0' + str(train_config['n_ctx']) + 'b'
word_belief = dict()
for index in range(2**train_config['n_ctx']):
    word = format(index, str_format)
    word = tuple(int(token) for token in word)  # Assumes int words...
    valid = FRDN.supports(word)
    if valid:
        word_belief[word] = FRDN.current_distr.copy()

#transformer_inputs = [word for word in word_belief.keys()]
transformer_inputs = [[process.int_from_token[token] for token in word] for word in word_belief.keys()]
# unique rows
unique_words = []

for word in transformer_inputs:
    if word not in unique_words:
        unique_words.append(word)
print(f'number of unique words: {len(unique_words)}')
transformer_inputs = torch.tensor(unique_words, dtype=torch.int).to(device)

# are there duplicate rows in transformer inputs?
print(transformer_inputs.shape)

# print first few batches
print(transformer_inputs[:5])
print(transformer_inputs.shape)

# %%

transformer_input_beliefs = []
transformer_input_belief_indices = []

#str_format = '0' + str(train_config['n_ctx']) + 'b'
#word_belief = dict()
for index in range(2**train_config['n_ctx']):
    word = format(index, str_format)
    word = tuple(int(token) for token in word)  # Assumes int words...
    valid = process.supports(word)
    if valid:
        belief_list = []
        belief_index_list = []
        #tokenkey_list = []
        process.current_distr = process.stationary.copy()
        for token in word:
            token_key = process.int_from_token[token]
            #tokenkey_list.append(token_key)
            process.evolve(token_key)
            belief = tuple(round(b, 5) for b in process.current_distr.copy())
            belief_list.append(belief)
            belief_index = msp_belief_index[belief]  # may need to round...
            belief_index_list.append(belief_index)
        transformer_input_beliefs.append(belief_list)
        transformer_input_belief_indices.append(belief_index_list)

#transformer_input_beliefs = array(transformer_input_beliefs)
#transformer_input_belief_indices = array(transformer_input_belief_indices)
transformer_input_beliefs = torch.tensor(transformer_input_beliefs, dtype=torch.float32).to(device)
transformer_input_belief_indices = torch.tensor(transformer_input_belief_indices, dtype=torch.float32).to(device)

# torch.tensor(__).to(device)?
print(f"Transformer Input Beliefs: {transformer_input_beliefs.shape}, Transformer Input Belief Indices: {transformer_input_belief_indices.shape}")

#%%
def get_activations(model, transformer_inputs):
    name = f"blocks.{train_config['n_layers'] - 1}.hook_resid_post"
    activations = {}
    for batch in torch.split(transformer_inputs, train_config['batch_size']):
        _, batch_activations = model.run_with_cache(batch, names_filter=lambda x: name in x)
        for key, val in batch_activations.items():
            activations.setdefault(key, []).append(val)
    for key, val in activations.items():
        activations[key] = torch.cat(val)
    # print(activations.keys())
    acts = activations[name]
    return acts
#%%

def get_simplex(activations, transformer_input_beliefs, fname):
    _regression, belief_predictions = run_activation_to_beliefs_regression(activations.cpu(), transformer_input_beliefs.cpu())
    print(f"Shape of belief_predictions: {belief_predictions.shape}")
    #belief_predictions = belief_predictions.cpu() # ???
    #belief_predictions_flattened = belief_predictions.reshape(-1, 3)
    #transformer_input_belief_flattened = transformer_input_beliefs.reshape(-1, 3)
    belief_predictions_flattened = belief_predictions.reshape(-1, 4)
    transformer_input_belief_flattened = transformer_input_beliefs.reshape(-1, 4)
    transformer_input_belief_flattened = transformer_input_belief_flattened.cpu() #??

    # project to simplex
    belief_true_projected = project_beliefs(transformer_input_belief_flattened) # do dimensions make sense here?
    belief_pred_projected = project_beliefs(belief_predictions_flattened)

    #rgb_colors =  transformer_input_belief_flattened.cpu().numpy()
    #rgb_colors = rgb_colors.astype(int)

    sns.set_context("paper")
    fig, axes = plt.subplots(1, 2, figsize=(6, 3))

    # Move tensors to CPU before converting to NumPy arrays
    #belief_true_projected = belief_true_projected.cpu().numpy()

    # Plotting the true beliefs projected onto the simplex
    axes[0].scatter(belief_true_projected[0], belief_true_projected[1], marker='.', c='k', alpha=0.2, s=0.5)  # colors?
    axes[0].axis('off')
    axes[0].set_title("Ground Truth Simplex")

    # Plotting the predicted beliefs projected onto the simplex
    axes[1].scatter(belief_pred_projected[0], belief_pred_projected[1], marker='.', c='k', alpha=0.3, s=0.01)  # colors?
    axes[1].axis('off')
    axes[1].set_title("Residual Stream Simplex")

    # Adjust layout for better spacing
    plt.tight_layout()

    # Save the figure
    plt.savefig(f"{fname}.png", bbox_inches='tight', pad_inches=0)
    plt.savefig(f"{fname}.pdf", bbox_inches='tight', pad_inches=0)

    # Display the figure
    plt.show()

def plot_simplex(model, transformer_inputs, transformer_input_beliefs, fname):
    activations = get_activations(model, transformer_inputs)
    get_simplex(activations, transformer_input_beliefs, fname)

config = HookedTransformerConfig(
    d_model=train_config['d_model'],
    d_head=train_config['d_head'],
    n_layers=train_config['n_layers'],
    n_ctx=train_config['n_ctx'],
    n_heads=train_config['n_heads'],
    d_mlp=4*train_config['d_model'],
    d_vocab=train_config['d_vocab'],
    seed=train_config['seed'],
    device=device,
    act_fn=train_config['act_fn'],
    attn_only=train_config['attn_only'],
    normalization_type=train_config['normalization_type'],
)

model = HookedTransformer(config)
print(model)
for name, p in model.named_parameters():
    if p.requires_grad:
        print(name)
        print(f"     {p.numel()}, {list(p.shape)}")
total_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Total model parameters: {total_parameters}")

optimizer_dict = {
    'sgd': torch.optim.SGD,
    'adam': torch.optim.Adam,
    'adamw': torch.optim.AdamW,
}
optim = optimizer_dict[train_config['optimizer'].lower()]
optimizer = optim(model.parameters(), lr=train_config['learning_rate'], weight_decay=train_config['weight_decay'])
criterion = nn.CrossEntropyLoss(reduction="none")
minimum_cross_entropy = torch.tensor(minimum_cross_entropy, dtype=torch.float32).to(device)
#%%
#%%
import os
import json
import glob

# Load the most recent model and hyperparameters
LOAD_LATEST_MODEL = True
if LOAD_LATEST_MODEL:
    # Find the most recent model file
    model_files = glob.glob("model_*.pt")
    if model_files:
        latest_model = max(model_files, key=os.path.getctime)
        model.load_state_dict(torch.load(latest_model))
        print(f"Loaded model from {latest_model}")
        
        # Find and load the corresponding config file
        config_file = f"config_{latest_model.split('_')[1].split('.')[0]}.json"
        if os.path.exists(config_file):
            with open(config_file, "r") as f:
                loaded_config = json.load(f)
            train_config.update(loaded_config)
            print(f"Loaded configuration from {config_file}")
        else:
            print(f"No matching config file found for {latest_model}")
    else:
        print("No saved models found.")

    
#%%

model.train()
X_input = transformer_inputs[:,:-1]
Y_true = transformer_inputs[:,1:]
print(f"X_input: {X_input.shape}, Y_true: {Y_true.shape}")
print(f'first 5 examples')
print(X_input[:5])
print(Y_true[:5])
# go through each row of transformer_inputs
probs = []
for i in tqdm(range(len(transformer_inputs))):
    row = transformer_inputs[i]
    # convert torch row to list of ints
    row = row.tolist()
    prob = process.probability(row)
    probs.append(prob)

probs = torch.tensor(probs, dtype=torch.float32).to(device)
# extend probs to the size of X_input
probs = probs.unsqueeze(1)
# make it so that probs is copied into 15 columsn
probs = probs.expand(X_input.shape[0], 9)
#%%
# Clear memory
if device.type == "mps":
    torch.mps.empty_cache()
elif device.type == "cuda":
    torch.cuda.empty_cache()

# Create learning rate scheduler
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.9, patience=1000, verbose=True)

model.train()
Y_flattened = Y_true.reshape(-1)
# move Y_true to cpu
Y_true = Y_true.cpu()

for epoch in range(10000):
    optimizer.zero_grad()
    logits = model(X_input) # [batch_size, n_ctx - 1, d_vocab]
    loss = criterion(logits.view(-1, model.cfg.d_vocab), Y_flattened) # [batch_size * (n_ctx - 1)]
    loss = loss.reshape(X_input.shape[0], X_input.shape[1]) # [batch_size, n_ctx - 1]
    # Element-wise multiplication of loss and probs
    loss = torch.mul(loss, probs)
    loss = loss.sum(dim=0) # [n_ctx - 1]
    #print(f"loss_per_position: {loss/minimum_cross_entropy[:-1]}")
    
    # First subplot: original plot
    # only plot every 100 epochs
    if epoch % 100 == 0:
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10))

        ax1.plot(loss.detach().cpu().numpy(), label='Loss')
        ax1.plot(minimum_cross_entropy[:-1].cpu().numpy(), label='Minimum Cross Entropy')
        ax1.legend()
        ax1.set_title('Loss and Minimum Cross Entropy')
    
    # Second subplot: loss per position / minimum cross entropy
    if epoch % 100 == 0:
        loss_per_position = loss.detach().cpu().numpy() / minimum_cross_entropy[:-1].cpu().numpy()
        ax2.plot(loss_per_position)
        ax2.set_title('Loss per Position / Minimum Cross Entropy')
    
        plt.tight_layout()
        plt.show()
    loss = loss.mean()
    print(f"epoch {epoch}, loss: {loss}")
    # backprob
    loss.backward()
    optimizer.step()
    
    # Step the scheduler
    scheduler.step(loss)
   
#%%
import uuid
import json
# save model and hyperparams
SAVE_TRAINING_RUN = True
if SAVE_TRAINING_RUN:
    model_name = f"model_{uuid.uuid4()}.pt"
    torch.save(model.state_dict(), model_name)
    with open(f"config_{uuid.uuid4()}.json", "w") as f:
        json.dump(train_config, f)

    
#%%
def train_epoch(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    train_dataloader: torch.utils.data.DataLoader,
    valid_dataloader: torch.utils.data.DataLoader,
    criterion: torch.nn.Module,
    minimum_cross_entropy: torch.Tensor,
):
    model.train()
    optimizer.zero_grad()
    total_weighted_loss = 0
    total_mean_loss = 0
    total_relative_loss = torch.zeros_like(minimum_cross_entropy[:-1])
    
    accumulation_steps = 100000
    for i, (X, Y_true, probs) in enumerate(train_dataloader):
        X, Y_true, probs = X.to(device), Y_true.to(device), probs.to(device)
        
        Y = model(X)  # Forward pass
        loss = criterion(Y.view(-1, model.cfg.d_vocab), Y_true.view(-1))
        loss = loss.view(X.shape[0], X.shape[1])  # (batch_size, n_ctx - 1)
        
        # Weight the loss by probabilities
        weighted_loss = (loss * probs.unsqueeze(1)).mean()
        weighted_loss = weighted_loss / accumulation_steps  # Normalize the loss
        weighted_loss.backward()  # Accumulate gradients
        
        mean_loss, relative_loss = compute_losses(loss, minimum_cross_entropy[:-1])  # Use all but the last element
        total_weighted_loss += weighted_loss.item() * accumulation_steps  # Undo normalization for logging
        total_mean_loss += mean_loss
        total_relative_loss += relative_loss
        
        if (i + 1) % accumulation_steps == 0:
            optimizer.step()
            optimizer.zero_grad()
        
        
    
    # Handle any remaining gradients
    if (i + 1) % accumulation_steps != 0:
        optimizer.step()
        optimizer.zero_grad()
    
    # Compute average losses
    num_batches = len(train_dataloader)
    avg_weighted_loss = total_weighted_loss / num_batches
    avg_mean_loss = total_mean_loss / num_batches
    avg_relative_loss = total_relative_loss / num_batches
        
    log_data = {
        "loss": avg_weighted_loss,
        "relative_loss": avg_relative_loss.mean().item(),
    }
    for i, rel_loss in enumerate(avg_relative_loss):
        log_data[f"relative_loss_{i}"] = rel_loss.item()

    # validation
    # Use training log for validation metrics
    log_data["val_loss"] = log_data["loss"]
    log_data["val_relative_loss"] = log_data["relative_loss"]
    for i in range(len(avg_relative_loss)):
        log_data[f"val_relative_loss_{i}"] = log_data[f"relative_loss_{i}"]

    return log_data


def compute_losses(
    loss: torch.Tensor, minimum_cross_entropy: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    per_position_loss = loss.mean(dim=0)  # (n_ctx - 1,)
    relative_loss = per_position_loss / minimum_cross_entropy
    mean_loss = per_position_loss.mean()
    return mean_loss, relative_loss



def plot_loss(log_data, epoch, fname):
    sns.set_context("paper")
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    min_cross_entropy = minimum_cross_entropy.cpu().numpy()
    n_positions = len(min_cross_entropy) - 1  # We're now using one less position

    hor_axis = list(range(n_positions))
    train_vert_axis = [log_data[f"relative_loss_{index}"] * min_cross_entropy[index] for index in hor_axis]
    val_vert_axis = [log_data[f"val_relative_loss_{index}"] * min_cross_entropy[index] for index in hor_axis]

    axes[0].plot(hor_axis, train_vert_axis, label="Train")
    axes[0].plot(hor_axis, min_cross_entropy[:-1], label="min")  # Plot all but the last element
    axes[0].set_title(f"Train loss at epoch {epoch}")
    axes[0].set_xlabel("Position")
    axes[0].set_ylabel("Loss")
    axes[0].legend()

    axes[1].plot(hor_axis, val_vert_axis, label="Validation")
    axes[1].plot(hor_axis, min_cross_entropy[:-1], label="min")  # Plot all but the last element
    axes[1].set_title(f"Validation loss at epoch {epoch}")
    axes[1].set_xlabel("Position")
    axes[1].set_ylabel("Loss")
    axes[1].legend()

    plt.show()

    # Plot relative losses
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    train_relative = [log_data[f"relative_loss_{index}"] for index in hor_axis]
    val_relative = [log_data[f"val_relative_loss_{index}"] for index in hor_axis]

    axes[0].plot(hor_axis, train_relative, label="Train")
    axes[0].set_title(f"Train relative loss at epoch {epoch}")
    axes[0].set_xlabel("Position")
    axes[0].set_ylabel("Relative Loss")
    axes[0].legend()

    axes[1].plot(hor_axis, val_relative, label="Validation")
    axes[1].set_title(f"Validation relative loss at epoch {epoch}")
    axes[1].set_xlabel("Position")
    axes[1].set_ylabel("Relative Loss")
    axes[1].legend()

    plt.show()

class CustomProcessDataset(IterableDataset):

    def __init__(self, process, sequence_length, num_samples, fixed=False):
        super().__init__()
        self.process = process
        self.sequence_length = sequence_length
        self.num_samples = num_samples
        self.fixed = fixed
        if self.fixed:
          self.samples = list(self._get_samples())
        else:
          self.samples = None

    def _get_samples(self):
      return process.yield_emissions(
            sequence_len=self.num_samples * (self.sequence_length + 1)
        )

    def __len__(self):
        return self.num_samples

    def __iter__(self) -> Iterable[Tuple[List[int]]]:
        samples = self._get_samples() if self.samples is None else iter(self.samples)
        for _ in range(self.num_samples):
            process_history = [
                next(samples) for _ in range(self.sequence_length + 1)
            ]
            yield (process_history[:-1], process_history[1:])


def process_dataset_collate_fn(batch: List[Tuple[List[int]]]):
    data = [x[0] for x in batch]
    labels = [x[1] for x in batch]
    return torch.tensor(data, dtype=torch.long), torch.tensor(labels, dtype=torch.long)

class FullDistributionDataset(Dataset):
    def __init__(self, process, sequence_length):
        self.process = process
        self.sequence_length = sequence_length
        self.data = []
        self.probabilities = []
        
        # Generate all possible sequences
        for i in range(2**sequence_length):
            sequence = [int(b) for b in format(i, f'0{sequence_length}b')]
            if self.process.supports(sequence):
                self.data.append(sequence)
                self.probabilities.append(self.process.probability(sequence))
        
        # Normalize probabilities
        total_prob = sum(self.probabilities)
        self.probabilities = [p / total_prob for p in self.probabilities]
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        sequence = self.data[idx]
        probability = self.probabilities[idx]
        return (torch.tensor(sequence[:-1], dtype=torch.long), 
                torch.tensor(sequence[1:], dtype=torch.long), 
                torch.tensor(probability, dtype=torch.float32))

def full_distribution_collate_fn(batch):
    inputs, targets, probs = zip(*batch)
    return (torch.stack(inputs).to(device), 
            torch.stack(targets).to(device), 
            torch.stack(probs).to(device))

# Create datasets
train_dataset = FullDistributionDataset(process, sequence_length=train_config['n_ctx'])

# Create dataloaders
train_dataloader = DataLoader(train_dataset, batch_size=train_config['batch_size'], collate_fn=full_distribution_collate_fn, shuffle=True)

# go through the train_dataloader and collect probabilities
train_probabilities = []
for inputs, targets, probs in train_dataloader:
    train_probabilities.append(probs)
train_probabilities = torch.cat(train_probabilities)
#%%
# plot a histogram of the probabilities
plt.hist(train_probabilities.cpu().numpy(), bins=20)
plt.show()

# %%
plt.plot(loss.detach().cpu().numpy())
plt.plot(minimum_cross_entropy[:-1].cpu().numpy())
plt.ylim([0.574, 0.576])
plt.show()
# %%
# %%
print(f'the curren learning rate is {optimizer.param_groups[0]["lr"]}')
# %%
model.to('cpu')
X_input.to('cpu')
transformer_input_beliefs.to('cpu')
transformer_input_belief_indices.to('cpu')
_, acts = model.run_with_cache(X_input, names_filter=lambda x: 'resid' in x)
#acts = acts['blocks.1.hook_resid_post'] # [batch_size, n_ctx, d_model]
# concate all acts
a_ = []
for act in acts:
    a_ = acts[act]

#acts = torch.cat(a_, dim=2)
acts = a_
print(acts.shape)


# %%
transformer_input_beliefs.shape # [batch_size, n_ctx+1, d_model]
beliefs = transformer_input_beliefs[:,:-1,:] # [batch_size, n_ctx, d_model]
# %%
acts = acts.detach().cpu().numpy()
beliefs = beliefs.detach().cpu().numpy()
probs = probs.detach().cpu().numpy()
#%%
# Reshape acts and beliefs for linear regression
acts_reshaped = acts.reshape(-1, acts.shape[-1])  # Flatten batch and context dimensions
beliefs_reshaped = beliefs.reshape(-1, beliefs.shape[-1])  # Flatten batch and context dimensions

# Perform linear regression with whitening
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler

# Whiten the input data
scaler = StandardScaler()
acts_whitened = acts_reshaped#scaler.fit_transform(acts_reshaped)

# Perform linear regression on whitened data
reg = LinearRegression().fit(acts_whitened, beliefs_reshaped)

# Calculate R-squared score
r_squared = reg.score(acts_reshaped, beliefs_reshaped)

print(f"R-squared score: {r_squared}")

# Calculate and print the coefficients
coefficients = reg.coef_
intercept = reg.intercept_

print("Regression coefficients:")
print(coefficients)
print("Intercept:")
print(intercept)

# Optionally, you can visualize the relationship between predicted and actual values
import matplotlib.pyplot as plt

predicted_beliefs = reg.predict(acts_reshaped)

plt.figure(figsize=(10, 6))
plt.scatter(beliefs_reshaped[:, 1], predicted_beliefs[:, 1], alpha=0.5)
plt.xlabel("Actual Activation")
plt.ylabel("Predicted Activation")
plt.title("Actual vs Predicted Activation (First Dimension)")
plt.show()

# %%
# scatter plot of beliefs_reshaped :,1 and :,2
plt.figure(figsize=(10, 6))
plt.scatter(beliefs_reshaped[:, 2], beliefs_reshaped[:, 3], alpha=0.5)
plt.xlabel("Actual Activation")
plt.ylabel("Predicted Activation")
plt.title("Actual vs Predicted Activation (First Dimension)")
plt.show()
# %%

plt.figure(figsize=(10, 6))
plt.scatter(predicted_beliefs[:, 2], predicted_beliefs[:, 3], alpha=0.5)
plt.xlabel("Actual Activation")
plt.ylabel("Predicted Activation")
plt.title("Actual vs Predicted Activation (First Dimension)")
plt.show()
# %%
belief_inds = transformer_input_belief_indices.to('cpu')[:,:-1]
belief_inds.shape
belief_inds_reshaped = belief_inds.reshape(-1)
# %%
# scatter plot of beliefs_reshaped :,2 and :,3 colored by belief_inds
plt.figure(figsize=(10, 6))
unique_inds = np.unique(belief_inds_reshaped)
cmap = plt.colormaps['tab20']  # Discrete colormap with 20 colors
norm = plt.Normalize(vmin=unique_inds.min(), vmax=unique_inds.max())
scatter = plt.scatter(beliefs_reshaped[:, 2], beliefs_reshaped[:, 3], 
                      c=belief_inds_reshaped, cmap=cmap, norm=norm, alpha=0.5)
plt.colorbar(scatter, label='Belief Index', ticks=unique_inds)
plt.xlabel("Belief Dimension 3")
plt.ylabel("Belief Dimension 4")
plt.title("Belief Dimensions 3 vs 4 (Colored by Belief Index)")
plt.show()
# %%

# scatter plot of predicted_beliefs :,2 and :,3 colored by belief_inds
plt.figure(figsize=(10, 6))
scatter = plt.scatter(predicted_beliefs[:, 2], predicted_beliefs[:, 3], 
                      c=belief_inds_reshaped, cmap=cmap, norm=norm, alpha=0.1)
plt.colorbar(scatter, label='Belief Index', ticks=unique_inds)
plt.xlabel("Predicted Belief Dimension 3")
plt.ylabel("Predicted Belief Dimension 4")
plt.title("Predicted Belief Dimensions 3 vs 4 (Colored by Belief Index)")
plt.show()

# %%
# Calculate centers of mass for predicted and ground truth beliefs
from pypalettes import load_cmap
import matplotlib.colors as mcolors

cmap = load_cmap("Signac")

# Create a professional-looking colormap
n_colors = len(np.unique(belief_inds_reshaped))
colors_sequential = cmap(np.linspace(0, 1, n_colors))
cmap_sequential = mcolors.ListedColormap(colors_sequential)

# Choose one of the above options to use
cmap = cmap_sequential

unique_inds = np.unique(belief_inds_reshaped)
predicted_centers = []
ground_truth_centers = []

for ind in unique_inds:
    mask = belief_inds_reshaped == ind
    predicted_center = np.mean(predicted_beliefs[mask], axis=0)
    predicted_centers.append(predicted_center)
    
    ground_truth_center = np.mean(beliefs_reshaped[mask], axis=0)
    ground_truth_centers.append(ground_truth_center)

predicted_centers = np.array(predicted_centers)
ground_truth_centers = np.array(ground_truth_centers)

# Create side-by-side plot
fig, (ax2, ax1) = plt.subplots(1, 2, figsize=(8, 4))

# Determine the overall min and max for both plots
x_min = min(ground_truth_centers[:, 2].min(), predicted_centers[:, 2].min())
x_max = max(ground_truth_centers[:, 2].max(), predicted_centers[:, 2].max())
y_min = min(ground_truth_centers[:, 3].min(), predicted_centers[:, 3].min())
y_max = max(ground_truth_centers[:, 3].max(), predicted_centers[:, 3].max())

# Plot for ground truth beliefs
scatter2 = ax2.scatter(ground_truth_centers[:, 2], ground_truth_centers[:, 3], c=unique_inds, cmap=cmap, s=150)
ax2.set_xlim(x_min, x_max)
ax2.set_ylim(y_min, y_max)

# Plot for predicted beliefs
for i, ind in enumerate(unique_inds):
    mask = belief_inds_reshaped == ind
    color = cmap((ind-1)/20)
    ax1.scatter(predicted_beliefs[mask, 2], predicted_beliefs[mask, 3], 
                color=color, alpha=.1, s=1)

scatter1 = ax1.scatter(predicted_centers[:, 2], predicted_centers[:, 3], c=unique_inds, cmap=cmap, s=150)
ax1.set_xlim(x_min, x_max)
ax1.set_ylim(y_min, y_max)

# Make the box all around and black, with 3 ticks per side
for ax in [ax1, ax2]:
    ax.spines['top'].set_visible(True)
    ax.spines['right'].set_visible(True)
    ax.spines['left'].set_visible(True)
    ax.spines['bottom'].set_visible(True)
    for spine in ax.spines.values():
        spine.set_color('black')
    ax.tick_params(colors='black', which='both', top=True, right=True)
    
    # Set 3 ticks per side
    ax.set_xticks([])
    ax.set_yticks([])
    
    # Label the ticks
    ax.set_xticklabels([])
    ax.set_yticklabels([])

# Adjust layout and save
plt.tight_layout()
plt.savefig('belief_centers_comparison_cmap.svg', dpi=300, bbox_inches='tight')
plt.show()

# %%
st01 = [FRDN.sample() for _ in range(25)]
# convert to string of AB
st_AB = ['A' if x == 0 else 'B' for x in st01]
st01_string = ''.join([str(x) for x in st01])

stAB_string = ''.join(st_AB)
print(st01_string)
print(stAB_string)
# %%
