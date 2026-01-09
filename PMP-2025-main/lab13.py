import pandas as pd
import pymc as pm
import numpy as np
import matplotlib.pyplot as plt
import arviz as az

def main():
    # Incarcarea datelor
    try:
        df = pd.read_csv('date.csv', sep='\s+', header=None, names=['x', 'y'])
    except FileNotFoundError:
        print("Fisierul 'date.csv' nu a fost gasit.")
        return

    x_data = df['x'].values
    y_data = df['y'].values
    print(f"S-au incarcat {len(df)} puncte de date.")

    # Definirea modelului
    with pm.Model() as model_regresie:
        w = pm.Normal('w', mu=0, sigma=10)
        b = pm.Normal('b', mu=0, sigma=10)
        sigma = pm.HalfNormal('sigma', sigma=5)

        mu = pm.Deterministic('mu', w * x_data + b)

        y_obs = pm.Normal('y_obs', mu=mu, sigma=sigma, observed=y_data)

        # Sampling
        print("Incepe procesul de sampling...")
        trace = pm.sample(1000, tune=1000, return_inferencedata=True, progressbar=True)

    # Analiza
    rezumat = az.summary(trace, var_names=['w', 'b'])
    w_mediu = rezumat.loc['w', 'mean']
    b_mediu = rezumat.loc['b', 'mean']

    print("\nREZULTATE:")
    print(f"Panta (w): {w_mediu:.4f}")
    print(f"Bias (b): {b_mediu:.4f}")

    # Afisare ecuatie
    print("-" * 30)
    if b_mediu >= 0:
        print(f"Ecuatia dreptei: y = {w_mediu:.4f} * x + {b_mediu:.4f}")
    else:
        print(f"Ecuatia dreptei: y = {w_mediu:.4f} * x - {abs(b_mediu):.4f}")
    print("-" * 30)

    # Vizualizare
    plt.figure(figsize=(10, 6))
    plt.scatter(x_data, y_data, c='blue', label='Date')
    
    x_linie = np.linspace(x_data.min(), x_data.max(), 100)
    y_linie = w_mediu * x_linie + b_mediu
    plt.plot(x_linie, y_linie, c='red', lw=3, label='Regresie Bayesiana')
    
    # Plot HDI
    az.plot_hdi(x_data, trace.posterior['mu'], color='red', fill_kwargs={'alpha': 0.3}, smooth=False)
    
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.legend()
    plt.show()

if __name__ == "__main__":
    main()