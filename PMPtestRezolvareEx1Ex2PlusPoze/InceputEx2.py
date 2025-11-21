import numpy as np
from hmmlearn import hmm

model = hmm.CategoricalHMM(n_components=3, random_state=42)
#Initial State Probabilities
model.startprob_ = np.array([0.4, 0.3, 0.3])
#Transition Probabilities
model.transmat_ = np.array([
    [0.6, 0.3, 0.1],
    [0.2, 0.7, 0.1],
    [0.3, 0.2, 0.5]
])
#Emission Probabilities
model.emissionprob_ = np.array([
    [0.1, 0.7, 0.2],
    [0.05, 0.25, 0.7],
    [0.8, 0.15, 0.05]
])
print("Model constructed successfully.")


#Forward Algorithm
obs_sequence = np.array([[1, 2, 0]]).T  

# model.score returns the Log Probability. We exponentiate it to get the actual probability.
log_prob = model.score(obs_sequence)
prob = np.exp(log_prob)
print(f"\n Forward Algorithm")
print(f"Observation Sequence: [Medium, High, Low]")
print(f"Log Probability: {log_prob:.5f}")
print(f"Probability: {prob:.5f}")

#Generate 10,000 sequences and estimate probability of [M, H, L]
n_simulations = 10000
match_count = 0
target_sequence = [1, 2, 0]

for _ in range(n_simulations):
    # Generate a sequence of length 3
    X, Z = model.sample(n_samples=3)

    current_obs = X.flatten().tolist()
    
    if current_obs == target_sequence:
        match_count += 1

empirical_prob = match_count / n_simulations

print(f"\n Empirical Estimation")
print(f"Number of matches in {n_simulations} runs: {match_count}")
print(f"Empirical Probability: {empirical_prob:.5f}")
print(f"Difference from Exact: {abs(empirical_prob - prob):.5f}")