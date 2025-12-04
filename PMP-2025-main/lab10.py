import pymc as pm
import numpy as np
import arviz as az
import matplotlib.pyplot as plt

#LAB 10: Regresie Liniara

# Introducerea datelor din tabel
x_data = np.array([
    1.5, 2.0, 2.3, 3.0, 3.5, 4.0, 4.5, 5.0, 5.5, 6.0, 
    6.5, 7.0, 7.5, 8.0, 8.5, 9.0, 9.5, 10.0, 10.5, 11.0
])

y_data = np.array([
    5.2, 6.8, 7.5, 8.0, 9.0, 10.2, 11.5, 12.0, 13.5, 14.0, 
    15.0, 15.5, 16.2, 17.0, 18.0, 18.5, 19.5, 20.0, 21.0, 22.0
])

# Vizualizam datele initiale
plt.scatter(x_data, y_data, label='Date observate')
plt.xlabel('Publicitate (mii $)')
plt.ylabel('Vanzari (mii $)')
plt.title('Relatia Publicitate vs Vanzari')
plt.show()

# Definirea Modelului
with pm.Model() as model_reg:
    
    x_shared = pm.MutableData('x_shared', x_data)
    
    # a) Priors pentru coeficienti
    # Alpha (intercept): punctul de intersectie cu axa Y
    alpha = pm.Normal('alpha', mu=0, sigma=10)
    
    # Beta (slope)
    beta = pm.Normal('beta', mu=1, sigma=10)
    
    # Sigma (noise)
    sigma = pm.HalfNormal('sigma', sigma=10)
    
    # Ecuatia dreptei 
    mu = alpha + beta * x_shared
    
    # Likelihood
    y = pm.Normal('y', mu=mu, sigma=sigma, observed=y_data)
    
    # b) Inferenta (Sampling)
    idata = pm.sample(2000, return_inferencedata=True, progressbar=True)

print("Rezultate Regresie (Alpha, Beta, Sigma):")
summary = az.summary(idata, var_names=['alpha', 'beta', 'sigma'], hdi_prob=0.95)
print(summary)


x_new = np.linspace(0, 13, 100) #100 de puncte noi intre 0 si 13

with model_reg:
    # Schimbam datele de intrare cu cele noi
    pm.set_data({'x_shared': x_new})
    
    # Posterior Predictive
    post_pred = pm.sample_posterior_predictive(idata, predictions=True)

# Vizualizare finala (Regresia + Intervalul de incertitudine HDI)
plt.figure(figsize=(10, 6))

# Datele originale
plt.scatter(x_data, y_data, color='black', label='Date observate', zorder=5)

# Linia de regresie medie si intervalul de incertitudine
az.plot_hdi(x_new, post_pred.predictions['y'], hdi_prob=0.95, color='orange', fill_kwargs={'alpha': 0.3, 'label': '95% HDI Predictie'})

# media predictiilor pentru a desena linia centrala
y_mean_pred = post_pred.predictions['y'].mean(dim=["chain", "draw"])
plt.plot(x_new, y_mean_pred, color='orange', label='Linia de Regresie (Medie)')

plt.xlabel('Cheltuieli Publicitate (mii $)')
plt.ylabel('Vanzari (mii $)')
plt.title('Regresie Liniara Bayesiana + Predictii')
plt.legend()
plt.show()