import numpy as np
from hmmlearn import hmm

# --- Definirea parametrilor modelului HMM ---

# Stari: 0=Dificil, 1=Mediu, 2=Usor
# Observatii: 0=FB, 1=B, 2=S, 3=NS

n_components = 3 # Numarul de stari (Dificil, Mediu, Usor)

# Probabilitatile de start (P(primul test))
# [Dificil, Mediu, Usor]
start_probability = np.array([1/3, 1/3, 1/3])

# Matricea de tranzitie P(stare_urmatoare | stare_curenta)
# Randuri: Starea curenta (De la)
# Coloane: Starea urmatoare (Catre)
#           [D,    M,    U]
transition_probability = np.array([
  [0.0, 0.5,  0.5],  # De la Dificil
  [0.5, 0.25, 0.25], # De la Mediu
  [0.5, 0.25, 0.25]  # De la Usor
])

# Probabilitatile de emisie P(observatie | stare)
# Randuri: Starea (Dificil, Mediu, Usor)
# Coloane: Observatia (FB, B, S, NS)
#           [FB,   B,    S,    NS]
emission_probability = np.array([
  [0.1,  0.2,  0.4,  0.3],   # Stare Dificil
  [0.15, 0.25, 0.5,  0.1],   # Stare Mediu
  [0.2,  0.3,  0.4,  0.1]    # Stare Usor
])

# --- a) Definirea modelului HMM ---

# Initializarea modelului HMM Categorical
model = hmm.CategoricalHMM(n_components=n_components)

# Setarea parametrilor invatati manual
model.startprob_ = start_probability
model.transmat_ = transition_probability
model.emissionprob_ = emission_probability

print("--- Partea a) ---")
print("Model HMM definit cu succes.")
print("-" * 20)


# --- b) Determinarea probabilitatii observatiilor ---

# Secventa de observatii data: FB, S, B, S, B, NS, B, B
# Mapata la indici:           [0, 2, 1, 2, 1, 3, 1, 1]
observations_sequence = np.array([0, 2, 1, 2, 1, 3, 1, 1]).reshape(-1, 1)

# Calcularea log-probabilitatii secventei folosind algoritmul Forward
log_prob_observations = model.score(observations_sequence)

# Conversia din log-probabilitate in probabilitate normala
prob_observations = np.exp(log_prob_observations)

print("--- Partea b) ---")
print(f"Secventa observata: ['FB', 'S', 'B', 'S', 'B', 'NS', 'B', 'B']")
print(f"Probabilitatea de a obtine aceasta secventa:")
print(f"{prob_observations:.10e}") # Format stiintific pentru precizie
print("-" * 20)


# --- c) Determinarea celei mai probabile secvente de stari ---

# Gasirea celei mai probabile secvente de stari folosind algoritmul Viterbi
log_prob_viterbi, hidden_states = model.decode(observations_sequence)

# Conversia din log-probabilitate in probabilitate normala
prob_viterbi = np.exp(log_prob_viterbi)

# Maparea indicilor starilor inapoi la nume pentru afisare
state_map = {0: "Dificil", 1: "Mediu", 2: "Usor"}
path = [state_map[s] for s in hidden_states]

print("--- Partea c) ---")
print("Cea mai probabila secventa de dificultati (stari):")
print(path)
print(f"Probabilitatea acestei secvente specifice:")
print(f"{prob_viterbi:.10e}") 
print("-" * 20)