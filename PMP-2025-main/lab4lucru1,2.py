import pgmpy.models as pgm
import pgmpy.factors.discrete as pgmdf
from pgmpy.inference import VariableElimination
import networkx as nx
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import numpy as np
import itertools
import random

# Functii ajutatoare pentru maparea starilor 
# pgmpy foloseste indecsi {0, 1}, problema foloseste valori {-1, 1}
def get_value_from_index(index):
    """Mapeaza indexul 0 la -1 si indexul 1 la 1."""
    return -1 if index == 0 else 1

def get_index_from_value(value):
    """Mapeaza valoarea -1 la 0 si valoarea 1 la 1."""
    return 0 if value == -1 else 1


# REZOLVARE EXERCITIUL 1


print("--- REZOLVARE EXERCITIUL 1 ---")

# Partea a

print("\n--- Partea (a) ---")

# Definim structura grafului
model_ex1 = pgm.MarkovNetwork()
nodes_ex1 = ['A1', 'A2', 'A3', 'A4', 'A5']
edges_ex1 = [('A1', 'A2'), ('A1', 'A3'), ('A2', 'A4'), ('A2', 'A5'), 
             ('A3', 'A4'), ('A4', 'A5')]
model_ex1.add_nodes_from(nodes_ex1)
model_ex1.add_edges_from(edges_ex1)

# Vizualizam graful (salvare in fisier)
print("Se genereaza vizualizarea grafului pentru Ex. 1...")
plt.figure(figsize=(7, 5))
pos = nx.spring_layout(model_ex1, seed=42)
nx.draw(model_ex1, pos, with_labels=True, node_size=2500, node_color='skyblue', 
        font_size=16, font_weight='bold')
plt.title("Graful Retelei Markov (Ex. 1)", size=18)
output_filename_ex1 = "mrf_graph_ex1.png"
plt.savefig(output_filename_ex1)
print(f"Graful a fost salvat in fisierul: {output_filename_ex1}")

# Determinam clicile maximale
cliques = list(model_ex1.get_cliques())
print(f"\nClicile maximale ale modelului (Ex. 1) sunt:")
for clique in cliques:
    print(f"- {set(clique)}")

# Partea b: Distributia comuna si starea de probabilitate maxima

print("\n--- Partea (b) ---")
print("IPOTEZA: Se asuma un model Ising cu J=1.0 pentru fiecare muchie.")

# Definim factorii (potentialele) pentru fiecare muchie
factors_ex1 = []
J = 1.0 # Forta cuplajului
for edge in model_ex1.edges():
    var1, var2 = edge
    # Factor cu 2 variabile, fiecare cu 2 stari (0, 1)
    factor = pgmdf.DiscreteFactor([var1, var2], [2, 2])
    
    values = []
    # Iteram prin (0,0), (0,1), (1,0), (1,1)
    for state_idx in itertools.product([0, 1], repeat=2):
        s1_val = get_value_from_index(state_idx[0]) # val -1 sau 1
        s2_val = get_value_from_index(state_idx[1]) # val -1 sau 1
        
        # Potential: exp(J * s1 * s2)
        potential = np.exp(J * s1_val * s2_val)
        values.append(potential)
        
    factor.values = values
    factors_ex1.append(factor)

# Adaugam factorii la model
model_ex1.add_factors(*factors_ex1)

# Cream motorul de inferenta
inference_ex1 = VariableElimination(model_ex1)

# Determinam distributia comuna (normalizata)
print("\nSe calculeaza distributia comuna...")
joint_factor = inference_ex1.query(variables=nodes_ex1, joint=True)
joint_factor.normalize()

# Determinam starile cu probabilitate maxima
probabilities = []
all_states_indices = list(itertools.product([0, 1], repeat=len(nodes_ex1)))

for state_indices in all_states_indices:
    state_dict = dict(zip(nodes_ex1, state_indices))
    prob = joint_factor.get_value(**state_dict)
    state_values = tuple([get_value_from_index(i) for i in state_indices])
    probabilities.append((state_values, prob))

probabilities.sort(key=lambda x: x[1], reverse=True)

# Afisam starea/starile de probabilitate maxima
print("\n--- Starea/Starile de Probabilitate Maxima (Cea mai buna configuratie) ---")
max_prob = probabilities[0][1]
for state_vals, prob in probabilities:
    if np.isclose(prob, max_prob):
        state_dict = dict(zip(nodes_ex1, state_vals))
        print(f"  Stare: {state_dict}, Probabilitate = {prob:.6f}")
    else:
        break

# REZOLVARE EXERCITIUL 2 

print("\n\n--- REZOLVARE EXERCITIUL 2 ---")
print("Rezolvare pentru de-zgomotare (image denoising) folosind MAP.")

# --- Setup initial (Imaginea, Zgomotul, Parametrii) ---
H, W = 4, 4 # Dimensiunea imaginii (4x4)
LAMBDA = 2.0 # Factorul de regularizare (ales de noi)

# Creare imagine originala (clean) - un "plus"
clean_image = np.full((H, W), -1.0)
clean_image[H//2, :] = 1.0
clean_image[:, W//2] = 1.0
print(f"\nImaginea originala (clean) {H}x{W}:\n{clean_image}")

# Adaugare zgomot (flip 2 pixeli)
noisy_image = np.copy(clean_image)
pixels_to_flip = [(0, 1), (2, 0)] # ~12.5% zgomot
for i, j in pixels_to_flip:
    noisy_image[i, j] *= -1
print(f"\nImaginea cu zgomot (noisy):\n{noisy_image}")


# Partea a: Definirea Retelei Markov (MRF)

print("\n--- Partea (a): Definire MRF ---")

denoise_model = pgm.MarkovNetwork()

# Definim nodurile (cate un nod latent 'X' pentru fiecare pixel)
nodes_ex2 = []
for i in range(H):
    for j in range(W):
        node_name = f"X_{i}_{j}"
        nodes_ex2.append(node_name)
denoise_model.add_nodes_from(nodes_ex2)

# Definim muchiile (conexiuni N, S, E, W)
edges_ex2 = []
for i in range(H):
    for j in range(W):
        current_node = f"X_{i}_{j}"
        # Conexiune cu vecinul de jos
        if i + 1 < H:
            neighbor_node = f"X_{i+1}_{j}"
            edges_ex2.append((current_node, neighbor_node))
        # Conexiune cu vecinul din dreapta 
        if j + 1 < W:
            neighbor_node = f"X_{i}_{j+1}"
            edges_ex2.append((current_node, neighbor_node))
denoise_model.add_edges_from(edges_ex2)

print(f"MRF definita cu {len(denoise_model.nodes())} noduri (pixeli) si {len(denoise_model.edges())} muchii (vecinatati).")

# Partea b

print("\n--- Partea (b): Estimare MAP ---")

# Modelul probabilitatii este P(X|Y) proportional cu exp(-E(X))
# Factorii modelului sunt phi = exp(-E_componenta)

node_factors = []
edge_factors = []

# Starile (0,0) -> (-1, -1) => E = (-1 - (-1))^2 = 0. Phi = exp(0) = 1.0
# Starile (0,1) -> (-1,  1) => E = (-1 - 1)^2 = 4.  Phi = exp(-4)
# Starile (1,0) -> ( 1, -1) => E = (1 - (-1))^2 = 4.  Phi = exp(-4)
# Starile (1,1) -> ( 1,  1) => E = (1 - 1)^2 = 0.  Phi = exp(0) = 1.0
edge_potential_val = np.exp(-4.0)
edge_factor_values = [1.0, edge_potential_val, edge_potential_val, 1.0]

for edge in denoise_model.edges():
    factor = pgmdf.DiscreteFactor(edge, [2, 2], edge_factor_values)
    edge_factors.append(factor)
 
for i in range(H):
    for j in range(W):
        node_name = f"X_{i}_{j}"
        y_val = noisy_image[i, j] # Valoarea observata (-1 sau 1)
        
        # Calculam potentialul pentru fiecare stare posibila a lui Xi
        
        # Starea 0 (Xi = -1)
        x_val_0 = get_value_from_index(0)
        energy_0 = LAMBDA * (x_val_0 - y_val)**2
        phi_0 = np.exp(-energy_0)
        
        # Starea 1 (Xi = 1)
        x_val_1 = get_value_from_index(1)
        energy_1 = LAMBDA * (x_val_1 - y_val)**2
        phi_1 = np.exp(-energy_1)
        
        # Cream factorul
        factor = pgmdf.DiscreteFactor([node_name], [2], [phi_0, phi_1])
        node_factors.append(factor)

# Adaugam toti factorii la model
denoise_model.add_factors(*node_factors)
denoise_model.add_factors(*edge_factors)

inference_ex2 = VariableElimination(denoise_model)
print("Se calculeaza starea MAP (Maximum A Posteriori)...")
map_state = inference_ex2.map_query(variables=denoise_model.nodes())

# Reconstruim imaginea din starea MAP
denoised_image = np.zeros((H, W))
for i in range(H):
    for j in range(W):
        node_name = f"X_{i}_{j}"
        state_index = map_state[node_name]
        denoised_image[i, j] = get_value_from_index(state_index)

print(f"\nImaginea de-zgomotata (MAP):\n{denoised_image}")

fig, axes = plt.subplots(1, 3, figsize=(15, 5))
cmap_choice = 'gray' 

axes[0].imshow(clean_image, cmap=cmap_choice, vmin=-1, vmax=1)
axes[0].set_title("Imagine Originala (Clean)")

axes[1].imshow(noisy_image, cmap=cmap_choice, vmin=-1, vmax=1)
axes[1].set_title("Imagine cu Zgomot (Noisy)")

axes[2].imshow(denoised_image, cmap=cmap_choice, vmin=-1, vmax=1)
axes[2].set_title("Imagine De-zgomotata (MAP)")

for ax in axes:
    ax.set_xticks([])
    ax.set_yticks([])

output_filename_ex2 = "image_denoising_ex2.png"
plt.savefig(output_filename_ex2)
print(f"\nRezultatul de-zgomotarii a fost salvat in: {output_filename_ex2}")