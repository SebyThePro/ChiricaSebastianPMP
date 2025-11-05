import numpy as np
import arviz as az
import matplotlib.pyplot as plt

# (a) Parametrii posteriori
alpha_post = 181
beta_post = 10.1

# (c) Calculul modului
mode_post = (alpha_post - 1) / beta_post
print(f"--- Raspuns (c) ---")
print(f"Cea mai probabila valoare (Modul): {mode_post:.4f}")

# --- Rezolvare (b) ---
print(f"\n--- Raspuns (b) ---")
# Generam 100,000 de esantioane din distributia posterioara
# Nota: np.random.gamma foloseste (shape=alpha, scale=1/beta)
samples = np.random.gamma(shape=alpha_post, scale=(1/beta_post), size=100000)

# (b) Calculam 94% HDI folosind ArviZ
hdi_94 = az.hdi(samples, hdi_prob=0.94)
print(f"Intervalul de 94% HDI este: [{hdi_94[0]:.4f}, {hdi_94[1]:.4f}]")

print("\nSe genereaza graficul posterior...")
az.plot_posterior(samples, hdi_prob=0.94, round_to=2)
plt.xlabel("$\lambda$ (Rata medie de apeluri pe ora)")
plt.title("Distributia Posterioara pentru $\lambda$")
plt.show()
plt.savefig("posterior_distribution.png")
print("Graficul a fost salvat ca 'posterior_distribution.png'")