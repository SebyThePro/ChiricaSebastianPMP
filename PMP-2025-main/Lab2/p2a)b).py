import numpy as np
import random 
import matplotlib.pyplot as plt
import seaborn as sns

LAMBDAS =[1, 2, 5, 10]
NSIM = 1000

dataFixed ={}
print("Simulare parametru fix...")
for lam in LAMBDAS:
    dataFixed[f'Poisson(lam={lam})'] = np.random.poisson(lam, NSIM)
    print(f"generat 1000 de valori pentru Poisson({lam})")
data_randomized = []
print("\nSimulare cu parametru randomizat...")
for _ in range(NSIM):
    random_lambda = random.choice(LAMBDAS) 
    poisson_value = np.random.poisson(random_lambda, 1)[0]
    data_randomized.append(poisson_value)

#aduagam setul randomizat la dictionar
dataFixed['Randomizat (Mix)'] = np.array(data_randomized)
print(f" - Generat 1000 de valori")