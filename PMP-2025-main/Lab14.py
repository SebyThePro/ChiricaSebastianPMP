import pandas as pd
import pymc as pm
import numpy as np
import matplotlib.pyplot as plt
import arviz as az

import matplotlib
matplotlib.use('Agg') 

def main():
    # Incarcare date
    try:
        df = pd.read_csv('date_colesterol.csv')
    except FileNotFoundError:
        print("Eroare: Fisierul date_colesterol.csv nu a fost gasit.")
        return

    x_data = df['Ore_Exercitii'].values
    y_data = df['Colesterol'].values

    print(f"Date incarcate: {len(df)} randuri.")
    print(f"Coloane: {df.columns.tolist()}")

    # Model PyMC
    with pm.Model() as model:
        # Priori
        alpha = pm.Normal('alpha', mu=y_data.mean(), sigma=50) # Intercept 
        beta = pm.Normal('beta', mu=0, sigma=20)               # Panta 
        sigma = pm.HalfNormal('sigma', sigma=20)               # Zgomot

        # Relatia liniara
        mu = alpha + beta * x_data

        # Likelihood
        y_obs = pm.Normal('y_obs', mu=mu, sigma=sigma, observed=y_data)

        # Sampling
        print("Incepe sampling-ul...")
        trace = pm.sample(1000, tune=1000, return_inferencedata=True, progressbar=True)

    # Rezultate
    rezumat = az.summary(trace, var_names=['alpha', 'beta'])
    b_mediu = rezumat.loc['alpha', 'mean'] # Bias 
    w_mediu = rezumat.loc['beta', 'mean']  # Panta 

    print("\n" + "="*30)
    print("REZULTATE:")
    print(f"Intercept (alpha/b): {b_mediu:.4f}")
    print(f"Panta (beta/w):      {w_mediu:.4f}")
    
    print("-" * 30)
    print(f"Ecuatia de regresie: Colesterol = {w_mediu:.4f} * Ore_Exercitii + {b_mediu:.4f}")
    print("="*30)

    # 4. Vizualizare si Salvare
    plt.figure(figsize=(10, 6))
    
    # Punctele de date
    plt.scatter(x_data, y_data, c='blue', alpha=0.6, label='Date Observate')
    
    # Linia de regresie
    x_range = np.linspace(x_data.min(), x_data.max(), 100)
    y_range = b_mediu + w_mediu * x_range
    plt.plot(x_range, y_range, c='red', lw=3, label='Regresie Bayesiana')
    
    # Incertitudinea 
    az.plot_hdi(x_data, trace.posterior['mu'], color='red', fill_kwargs={'alpha': 0.3}, smooth=False)
    
    plt.xlabel('Ore Exercitii')
    plt.ylabel('Colesterol')
    plt.title('Relatia Exercitii vs Colesterol')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    nume_grafic = 'rezultat_colesterol.png'
    plt.savefig(nume_grafic)
    print(f"\nGraficul a fost salvat in: {nume_grafic}")

if __name__ == "__main__":
    main()