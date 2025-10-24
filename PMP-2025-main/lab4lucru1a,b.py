import pgmpy.models as pgm
import pgmpy.factors.discrete as pgmdf
from pgmpy.inference import VariableElimination
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import itertools


# a:


print("--- Partea (a) ---")

# Definim structura grafului (reteaua Markov)
model = pgm.MarkovNetwork()

# Definim nodurile si muchiile
nodes = ['A1', 'A2', 'A3', 'A4', 'A5']
edges = [('A1', 'A2'), ('A1', 'A3'), ('A2', 'A4'), ('A2', 'A5'), ('A3', 'A4'), ('A4', 'A5')]

model.add_nodes_from(nodes)
model.add_edges_from(edges)

# Vizualizam graful
print("Se genereaza vizualizarea grafului...")
plt.figure(figsize=(7, 5))
pos = nx.spring_layout(model, seed=42) 
nx.draw(model, pos, with_labels=True, node_size=2500, node_color='skyblue', 
        font_size=16, font_weight='bold')
plt.title("Graful Retelei Markov (MRF)", size=18)
plt.show()

# Determinam clicile modelului
cliques = list(model.get_cliques())
print(f"\nClicile maximale ale modelului sunt:")
# Afisam clicile ca seturi pentru o citire mai usoara
for clique in cliques:
    print(f"- {set(clique)}")


#b: 

print("\n--- Partea (b) ---")
print("IPOTEZA: Se asuma un model Ising cu potentiale de muchie J=1.0 si fara potentiale de nod.")

# Index 0 -> Valoare -1
# Index 1 -> Valoare 1
def get_value_from_index(index):
    return -1 if index == 0 else 1

# Definim factorii pentru fiecare muchie
factors = []
for edge in model.edges():
    var1, var2 = edge
    
    # Cream un factor pentru cele doua variabile, fiecare cu cardinalitate 2 (starile 0 si 1)
    factor = pgmdf.DiscreteFactor([var1, var2], [2, 2])
    
    values = []
    # Generam toate combinatiile de stari: (0, 0), (0, 1), (1, 0), (1, 1)
    for state_indices in itertools.product([0, 1], repeat=2):
        s1_idx, s2_idx = state_indices
        
        # Mapam indecsii (0, 1) la valorile (-1, 1)
        s1_val = get_value_from_index(s1_idx)
        s2_val = get_value_from_index(s2_idx)
        
        # Calculam valoarea potentialului conform ipotezei noastre: exp(1.0 * Ai * Aj)
        potential = np.exp(1.0 * s1_val * s2_val)
        values.append(potential)

    # Atribuim valorile calculate factorului
    factor.values = values
    factors.append(factor)

# Adaugam factorii la modelul Markov
model.add_factors(*factors)

# Cream un motor de inferenta pentru a calcula distributia comuna
inference = VariableElimination(model)

# Determinam distributia comuna
# Acesta este un factor care reprezinta produsul tuturor factorilor 
print("\nSe calculeaza distributia comuna...")
joint_factor = inference.query(variables=nodes, joint=True)

# Normalizam factorul pentru a obtine probabilitati (suma sa fie 1)
joint_factor.normalize()

# Determinam starile cu probabilitate maxima
probabilities = []
all_states_indices = list(itertools.product([0, 1], repeat=5))

for state_indices in all_states_indices:
    state_dict = dict(zip(nodes, state_indices))
    
    # Extragem probabilitatea pentru starea respectiva
    prob = joint_factor.get_value(**state_dict)
    
    # Mapam starea din {0, 1} inapoi la {-1, 1} pentru afisare
    state_values = tuple([get_value_from_index(i) for i in state_indices])
    probabilities.append((state_values, prob))

# Sortam starile dupa probabilitate (descrescator)
probabilities.sort(key=lambda x: x[1], reverse=True)

# Afisam rezultatele
print("\n--- Distributia Comuna (Top 5 stari) ---")
for i in range(min(5, len(probabilities))):
    state_vals, prob = probabilities[i]
    state_dict = dict(zip(nodes, state_vals))
    print(f"  {i+1}. Stare: {state_dict}, Probabilitate = {prob:.6f}")


print("\n--- Starea/Starile de Probabilitate Maxima (Cea mai buna configuratie) ---")
max_prob = probabilities[0][1]
for state_vals, prob in probabilities:
    if np.isclose(prob, max_prob):
        state_dict = dict(zip(nodes, state_vals))
        print(f"  Stare: {state_dict}, Probabilitate = {prob:.6f}")
    else:
        # De indata ce gasim o probabilitate mai mica, ne oprim
        break