import numpy as np
import random 
import matplotlib.pyplot as plt
import seaborn as sns

# Setarile principale pentru simulare
LAMBDAS = [1, 2, 5, 10]
NSIM = 1000 # Numarul de simulari cerut (1000 de valori pentru fiecare set de date)

# 1. Simulare Poisson cu parametru fix (Ex. 2, punctul 1)
dataFixed ={}
print("Simulare parametru fix...")
for lam in LAMBDAS:
    # Genereaza NSIM valori din distributia Poisson(lambda)
    dataFixed[f'Poisson(lam={lam})'] = np.random.poisson(lam, NSIM)
    print(f"generat 1000 de valori pentru Poisson({lam})")

# 2. Simulare Poisson Randomizată/Mixtă (Ex. 2, punctul 2)
data_randomized = []
print("\nSimulare cu parametru randomizat...")
for _ in range(NSIM):
    # Pasul a: Selecteaza un lambda aleatoriu din lista (cu probabilitate egala)
    random_lambda = random.choice(LAMBDAS) 
    # Pasul b: Genereaza O variabila Poisson folosind acel lambda selectat
    poisson_value = np.random.poisson(random_lambda, 1)[0]
    data_randomized.append(poisson_value)

# Adaugam setul randomizat la dictionar pentru a fi plotat impreuna cu celelalte
dataFixed['Randomizat (Mix)'] = np.array(data_randomized)
print(f" - Generat 1000 de valori pentru distributia Mixta")

all_data = {**dataFixed} 
values = list(all_data.values())

# Defineste limita maxima pentru axa X si intervalele (bins) pentru o vizualizare uniforma
max_val = max(np.max(v) for v in values) if values else 15
bins = np.arange(0, max_val + 2) - 0.5 

plt.figure(figsize=(12, 10))
plt.suptitle("Ex. 2a: Comparația Distribuțiilor Poisson Fixe și Randomizate", fontsize=16)

# Itereaza si ploteaza fiecare din cele 5 distributii
for i, (name, data) in enumerate(all_data.items()):
    plt.subplot(3, 2, i + 1) # Aranjeaza ploturile in 3 randuri si 2 coloane
    
    # Creeaza histograma (fara curba KDE), afisand probabilitatea pe axa Y
    sns.histplot(data, bins=bins, kde=False, stat="probability", 
                 color=sns.color_palette("tab10")[i], 
                 edgecolor='black', linewidth=0.5)
    plt.title(f'{name} (Media ≈ {np.mean(data):.2f})')
    plt.xlabel('Număr de Apeluri')
    plt.ylabel('Probabilitate')
    plt.xlim(-0.5, max_val + 1) # Seteaza limitele axei X
    
plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.show()