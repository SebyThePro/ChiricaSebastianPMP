import networkx as nx
import matplotlib 
matplotlib.use('Agg') 
import matplotlib.pyplot as plt
from pgmpy.models import DiscreteBayesianNetwork
from pgmpy.factors.discrete import TabularCPD
from pgmpy.inference import VariableElimination
import numpy as np

# Definirea Structuri Retelei: Arcele
model_spam = DiscreteBayesianNetwork([
    ('S', 'O'), 
    ('S', 'L'), 
    ('S', 'M'), 
    ('L', 'M')
])

# Definirea CPD-urilor (Tabelul Probabilitatilor Conditionate)
cpd_s = TabularCPD(variable='S', variable_card=2, values=[[0.6], [0.4]]) 

cpd_o = TabularCPD(variable='O', variable_card=2, 
                   values=[[1 - 0.1, 1 - 0.7],  
                           [0.1, 0.7]],      
                   evidence=['S'], 
                   evidence_card=[2])

cpd_l = TabularCPD(variable='L', variable_card=2, 
                   values=[[1 - 0.3, 1 - 0.8],  
                           [0.3, 0.8]],      
                   evidence=['S'], 
                   evidence_card=[2])

cpd_m = TabularCPD(variable='M', variable_card=2, 
                   values=[[1 - 0.2, 1 - 0.6, 1 - 0.5, 1 - 0.9], 
                           [0.2, 0.6, 0.5, 0.9]],          
                   evidence=['S', 'L'], 
                   evidence_card=[2, 2])

model_spam.add_cpds(cpd_s, cpd_o, cpd_l, cpd_m)

if not model_spam.check_model():
    print("Eroare: Modelul Bayesian nu este consistent.")

print("\n--- Desenarea Grafului ---")

pos = nx.shell_layout(model_spam)

plt.figure(figsize=(8, 6)) # Seteaza dimensiunea ferestrei

nx.draw(
    model_spam, 
    pos=pos, 
    with_labels=True, 
    node_size=4000, 
    font_weight='bold', 
    node_color='skyblue',
    font_color='black',
    arrowsize=20 
)
plt.title("Structura Retelei Bayesiane (S, O, L, M)")

# salvarea 
plt.savefig('grafic_retea.png') # fisierul 'grafic_retea.png'
print("Graficul a fost salvat ca 'grafic_retea.png'")

#a)
independencies = model_spam.get_independencies()

print("\nIndependente (pgmpy) ---")
print(independencies)

# b)
infer = VariableElimination(model_spam)

# Query: Calculeaza P(S | O=1, L=1, M=1)
evidence = {'O': 1, 'L': 1, 'M': 1}
result_classification = infer.query(variables=['S'], evidence=evidence)

print("\n")
print("Regula de Clasificare: Se alege S care maximizează P(S) * P(O|S) * P(L|S) * P(M|S,L)")
print("\nRezultatul Inferentei (Exemplu: O=1, L=1, M=1):")
print(result_classification)

# Extragerea rezultatului
prob_spam = result_classification.values[1]
prob_non_spam = result_classification.values[0]

if prob_spam > prob_non_spam:
    print(f"\nDecizie: P(Spam)={prob_spam:.4f} > P(Non-Spam)={prob_non_spam:.4f}. Emailul este clasificat ca: SPAM (S=1).")
else:
    print(f"\nDecizie: P(Spam)={prob_spam:.4f} < P(Non-Spam)={prob_non_spam:.4f}. Emailul este clasificat ca: NON-SPAM (S=0).")