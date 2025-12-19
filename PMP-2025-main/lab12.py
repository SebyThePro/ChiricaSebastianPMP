import pandas as pd
import numpy as np
import pymc as pm
import arviz as az
import matplotlib.pyplot as plt


# Incarcare si Pregatire Date
print(" Analiza Datelor ")

# Citim fisierul CSV
df = pd.read_csv('date_promovare_examen.csv')

# pentru verificare
print(df.head())

# X = Variabile independente (Studiu si Somn)
# Y = Variabila dependenta (Promovare: 0 sau 1)
X_studiu = df['Ore_Studiu'].values
X_somn = df['Ore_Somn'].values
Y_promovare = df['Promovare'].values

# Standardizam variabilele de intrare (Studiu si Somn)
def standardize(x):
    return (x - x.mean()) / x.std()

X_studiu_std = standardize(X_studiu)
X_somn_std = standardize(X_somn)

print("\nDatele au fost standardizate pentru a ajuta modelul Bayesian.\n")

# Construirea Modelului de Regresie Logistica
# Formula matematica:
# p = sigmoid(alpha + beta1 * Studiu + beta2 * Somn)
# Y ~ Bernoulli(p)

with pm.Model() as model_logistic:
    # Definierea distributiilor a priorilor 
    alpha = pm.Normal('alpha', mu=0, sigma=2)
    beta_studiu = pm.Normal('beta_studiu', mu=0, sigma=2)
    beta_somn = pm.Normal('beta_somn', mu=0, sigma=2)
    
    # Calculam scorul z (logit)
    mu = alpha + beta_studiu * X_studiu_std + beta_somn * X_somn_std
    
    # Functia de Link (Sigmoid) ---
    # Transformam scorul in probabilitate (intre 0 si 1)
    p = pm.math.sigmoid(mu)
    
    # Likelihood (Functia de verosimilitate) 
    Y_obs = pm.Bernoulli('Y_obs', p=p, observed=Y_promovare)
    
    # Sampling
    trace = pm.sample(draws=1000, tune=1000, return_inferencedata=True, progressbar=True)


# Interpretarea Rezultatelor 

print("\n--- Rezumat Statistic al Parametrilor ---\n")
summary = az.summary(trace, var_names=['alpha', 'beta_studiu', 'beta_somn'])
print(summary)

# Extragem mediile distributiilor 
mean_studiu = summary.loc['beta_studiu', 'mean']
mean_somn = summary.loc['beta_somn', 'mean']
hdi_studiu = summary.loc['beta_studiu', ['hdi_3%', 'hdi_97%']].values

print("\n" + "="*50)
print("INTERPRETARE SI RASPUNSURI (Comentarii):")
print("="*50)

# Impactul Studiului
if mean_studiu > 0 and hdi_studiu[0] > 0:
    print(f" Coeficientul pentru Ore_Studiu este pozitiv ({mean_studiu:.2f}).")
    print("   CONCLUZIE: Exista o probabilitate mare (peste 95%) ca studiul suplimentar")
    print("   sa creasca sansele de promovare.")
else:
    print(" Nu putem confirma cu certitudine statistica impactul orelor de studiu.")

#  Impactul Somnului
if mean_somn > 0:
    print(f"\n Coeficientul pentru Ore_Somn este de asemenea pozitiv ({mean_somn:.2f}).")
    print("   CONCLUZIE: Somnul influenteaza pozitiv sansele de promovare,")
    print("   dar trebuie comparat cu studiul pentru a vedea care e mai puternic.")
else:
    print("\n Somnul pare sa aiba un impact negativ sau neutru.")

#  Comparatie
if abs(mean_studiu) > abs(mean_somn):
    print("\n Comparatie: Orele de studiu par sa aiba o influenta mai mare")
    print("   asupra rezultatului decat orele de somn (magnitudinea coeficientului e mai mare).")
else:
    print("\n Comparatie: Orele de somn par sa fie la fel de importante sau mai importante")
    print("   decat studiul in acest set de date.")
