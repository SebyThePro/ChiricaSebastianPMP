import networkx as nx
import matplotlib.pyplot as plt
from pgmpy.models import BayesianNetwork
from pgmpy.factors.discrete import TabularCPD
from pgmpy.inference import VariableElimination

#Building the Bayesian Network
model = BayesianNetwork([
    ('O', 'H'), 
    ('O', 'W'),
    ('H', 'R'), 
    ('W', 'R'),
    ('H', 'E'),
    ('R', 'C')
])
#Conditional Probability Tables
cpd_o = TabularCPD(variable='O', variable_card=2, 
                   values=[[0.3], [0.7]],
                   state_names={'O': ['cold', 'mild']})
cpd_h = TabularCPD(variable='H', variable_card=2, 
                   values=[[0.9, 0.2], 
                           [0.1, 0.8]],
                   evidence=['O'], evidence_card=[2],
                   state_names={'H': ['yes', 'no'], 'O': ['cold', 'mild']})
cpd_w = TabularCPD(variable='W', variable_card=2, 
                   values=[[0.1, 0.6], 
                           [0.9, 0.4]],
                   evidence=['O'], evidence_card=[2],
                   state_names={'W': ['yes', 'no'], 'O': ['cold', 'mild']})
cpd_r = TabularCPD(variable='R', variable_card=2, 
                   values=[[0.6, 0.9, 0.3, 0.5], 
                           [0.4, 0.1, 0.7, 0.5]],
                   evidence=['H', 'W'], evidence_card=[2, 2],
                   state_names={'R': ['warm', 'cool'], 'H': ['yes', 'no'], 'W': ['yes', 'no']})
cpd_e = TabularCPD(variable='E', variable_card=2, 
                   values=[[0.8, 0.2], 
                           [0.2, 0.8]],
                   evidence=['H'], evidence_card=[2],
                   state_names={'E': ['high', 'low'], 'H': ['yes', 'no']})
cpd_c = TabularCPD(variable='C', variable_card=2, 
                   values=[[0.85, 0.40], 
                           [0.15, 0.60]],
                   evidence=['R'], evidence_card=[2],
                   state_names={'C': ['comfortable', 'uncomfortable'], 'R': ['warm', 'cool']})
# Add CPDs to the model and verifying
model.add_cpds(cpd_o, cpd_h, cpd_w, cpd_r, cpd_e, cpd_c)
assert model.check_model()
print("Model created and verified successfully.")
#the graph 
plt.figure(figsize=(8, 6))
nx.draw_circular(model, with_labels=True, node_size=3000, node_color="skyblue", font_size=12, font_weight="bold", arrowsize=20)
plt.title("Bayesian Network Graph")
plt.show()

infer = VariableElimination(model)
print("\n--- Inference Results ---")
#Inference using Variable Elimination
#P(H = yes | C = comfortable)
q1 = infer.query(variables=['H'], evidence={'C': 'comfortable'})
print(f"\n1. P(H = yes | C = comfortable):")
print(q1)
#P(E = high | C = comfortable)
q2 = infer.query(variables=['E'], evidence={'C': 'comfortable'})
print(f"\n2. P(E = high | C = comfortable):")
print(q2)
#MAP estimate
map_est = infer.map_query(variables=['H', 'W'], evidence={'C': 'comfortable'})
print(f"\n3. MAP Estimate for (H, W) given C = comfortable:")
print(map_est)