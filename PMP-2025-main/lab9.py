import pymc as pm
import numpy as np
import arviz as az
import matplotlib.pyplot as plt

y_values = [0, 5, 10]
theta_values = [0.2, 0.5]

# Configurare grafic pentru istributia lui n
fig_post, axes_post = plt.subplots(3, 2, figsize=(12, 10))
fig_post.suptitle('Distributia a Posteriori pentru n (Nr. total clienti)', fontsize=16)

# Configurare grafic pentru distributia viitoarelor vanzari
fig_pred, axes_pred = plt.subplots(3, 2, figsize=(12, 10))
fig_pred.suptitle('Posterior Predictive (Vanzari viitoare estimate)', fontsize=16)

for i, y_obs in enumerate(y_values):
    for j, theta in enumerate(theta_values):
        
        ax_p = axes_post[i, j]
        ax_pp = axes_pred[i, j]
        
        with pm.Model() as model:
            # Prior pentru n 
            n = pm.Poisson('n', mu=10)
            
            # Likelihood 
            # Y_obs cumparatori din n vizitatori cu probabilitatea theta
            y = pm.Binomial('y', n=n, p=theta, observed=y_obs)
            
            # Inferenta pentru n
            # Folosim Metropolis pentru ca n este discret
            idata = pm.sample(2000, step=pm.Metropolis(), return_inferencedata=True, progressbar=False)
            
            # Predictive Posterior 
            pm.sample_posterior_predictive(idata, extend_inferencedata=True, progressbar=False)
            
            #Plotting
            
            # Plot Posterior n
            az.plot_posterior(idata, var_names=['n'], ax=ax_p, hdi_prob=0.95)
            ax_p.set_title(f'Observed Y={y_obs}, Theta={theta}')
            
            # Plot Posterior Predictive (y)
            az.plot_dist(idata.posterior_predictive["y"], ax=ax_pp, color='orange')
            ax_pp.set_title(f'Predictie Y | Y_obs={y_obs}, Theta={theta}')
            ax_pp.set_xlabel("Vanzari viitoare")

plt.figure(fig_post.number)
plt.tight_layout()

plt.figure(fig_pred.number)
plt.tight_layout()

plt.show()