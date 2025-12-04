import pymc as pm
import numpy as np
import arviz as az

data = np.array([56, 60, 58, 55, 57, 59, 61, 56, 58, 60])

# Definirea modelului si Inferenta
with pm.Model() as model_a:
    # Prior centrat pe media datelor (58)
    mu = pm.Normal('mu', mu=58, sigma=10)
    sigma = pm.HalfNormal('sigma', sigma=10)
    
    # Likelihood
    y = pm.Normal('y', mu=mu, sigma=sigma, observed=data)
    
    # Inferenta 
    idata_a = pm.sample(2000, return_inferencedata=True)

# Afisare rezultate
summary_a = az.summary(idata_a, hdi_prob=0.95)
print("Rezultate model A:")
print(summary_a)

# Modelul cu prior puternic
with pm.Model() as model_d:
    mu = pm.Normal('mu', mu=50, sigma=1) 
    sigma = pm.HalfNormal('sigma', sigma=10)
    
    y = pm.Normal('y', mu=mu, sigma=sigma, observed=data)
    
    idata_d = pm.sample(2000, return_inferencedata=True)

print("\nRezultate model D (Prior puternic):")
print(az.summary(idata_d, hdi_prob=0.95))