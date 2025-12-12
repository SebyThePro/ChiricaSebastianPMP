import pandas as pd
import numpy as np
import pymc as pm
import arviz as az
import matplotlib.pyplot as plt


print("--- Incarcare si Preprocesare ---")

# Citire fisier CSV
df = pd.read_csv('Prices.csv')

df['Premium_Binary'] = df['Premium'].map({'yes': 1, 'no': 0})

def standardize(series):
    return (series - series.mean()) / series.std()

# coloane standardizate pentru modelare
df['Price_std'] = standardize(df['Price'])
df['Speed_std'] = standardize(df['Speed'])
df['HD_std'] = standardize(df['HardDrive'])
df['Ram_std'] = standardize(df['Ram'])

print("Datele au fost incarcate si standardizate.")
print(df[['Price', 'Speed', 'Premium_Binary']].head())
print("-" * 50)


# Regresie Liniara Simpla (Pret vs Viteza)

print("--- Modelare Simpla (Pret ~ Viteza) ---")

with pm.Model() as model_simple:
    # Priors
    alpha = pm.Normal('alpha', mu=0, sigma=1)
    beta = pm.Normal('beta', mu=0, sigma=1)
    sigma = pm.HalfNormal('sigma', sigma=1)

    # Ecuatia modelului: y = alpha + beta * x
    mu = alpha + beta * df['Speed_std'].values
    
    # Likelihood 
    Y_obs = pm.Normal('Y_obs', mu=mu, sigma=sigma, observed=df['Price_std'].values)

    # Sampling
    # extragem 1000 de mostre din distributia posterioara
    trace_simple = pm.sample(draws=1000, tune=1000, return_inferencedata=True, progressbar=True)

print("Rezultate Model Simplu:")
print(az.summary(trace_simple, var_names=['alpha', 'beta', 'sigma']))
print("-" * 50)


print("--- Modelare Multipla (Pret ~ Viteza + HD + Ram) ---")

with pm.Model() as model_multiple:
    # Priors
    alpha = pm.Normal('alpha', mu=0, sigma=1)
    beta_speed = pm.Normal('beta_speed', mu=0, sigma=1)
    beta_hd = pm.Normal('beta_hd', mu=0, sigma=1)
    beta_ram = pm.Normal('beta_ram', mu=0, sigma=1)
    sigma = pm.HalfNormal('sigma', sigma=1)

    # Ecuatia modelului multivariat
    mu = (alpha + 
          beta_speed * df['Speed_std'].values + 
          beta_hd * df['HD_std'].values + 
          beta_ram * df['Ram_std'].values)
    
    # Likelihood
    Y_obs = pm.Normal('Y_obs', mu=mu, sigma=sigma, observed=df['Price_std'].values)

    # Sampling
    trace_multiple = pm.sample(draws=1000, tune=1000, return_inferencedata=True, progressbar=True)

print("Rezultate Model Multiplu:")
print(az.summary(trace_multiple, var_names=['beta_speed', 'beta_hd', 'beta_ram']))
print("-" * 50)

print("--- Modelare cu Premium (Pret ~ Viteza + HD + Ram + Premium) ---")

with pm.Model() as model_premium:
    # Priors
    alpha = pm.Normal('alpha', mu=0, sigma=1)
    beta_speed = pm.Normal('beta_speed', mu=0, sigma=1)
    beta_hd = pm.Normal('beta_hd', mu=0, sigma=1)
    beta_ram = pm.Normal('beta_ram', mu=0, sigma=1)
    beta_premium = pm.Normal('beta_premium', mu=0, sigma=1) # Coeficientul de interes
    sigma = pm.HalfNormal('sigma', sigma=1)

    # Ecuatia completa
    mu = (alpha + 
          beta_speed * df['Speed_std'].values + 
          beta_hd * df['HD_std'].values + 
          beta_ram * df['Ram_std'].values +
          beta_premium * df['Premium_Binary'].values)
    
    # Likelihood
    Y_obs = pm.Normal('Y_obs', mu=mu, sigma=sigma, observed=df['Price_std'].values)

    # Sampling
    trace_premium = pm.sample(draws=1000, tune=1000, return_inferencedata=True, progressbar=True)

print("Rezultate Model Premium:")
print(az.summary(trace_premium, var_names=['beta_speed', 'beta_premium']))
print("-" * 50)

# Optional: Vizualizare grafica a impactului Premium
az.plot_forest(trace_premium, var_names=['beta_speed', 'beta_hd', 'beta_ram', 'beta_premium'], combined=True)
plt.title("Comparatie Impact Factori asupra Pretului")
plt.show()