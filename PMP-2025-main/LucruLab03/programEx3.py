import numpy as np
from scipy.stats import binom
# !!! Importul corect pentru versiunile recente de pgmpy !!!
from pgmpy.models import DiscreteBayesianNetwork
from pgmpy.factors.discrete import TabularCPD
from pgmpy.inference import VariableElimination

# Moneda masluita: 4/7
P_RIGGED = 4/7

# ==============================================================================
# PART 1: Simulare (10000 de jocuri)
# ==============================================================================

def run_simulation(num_simulations=10000):
    """
    Ruleaza simularea jocului de num_simulations ori
    pentru a estima sansele de castig ale fiecarui jucator.
    """
    print(f"--- Rularea Simularii pentru {num_simulations} de jocuri (Partea 1) ---")
    
    wins = {'P0': 0, 'P1': 0}
    p_P0_coin = 0.5    # Moneda corecta a lui P0
    p_P1_coin = P_RIGGED  # Moneda masluita a lui P1

    for _ in range(num_simulations):
        # 1. Se alege cine incepe
        starter_idx = np.random.choice([0, 1])
        starter = f'P{starter_idx}'
        
        # 2. Se determina cine arunca moneda (celalalt jucator) si bias-ul
        flipper = 'P1' if starter == 'P0' else 'P0'
        p_heads = p_P1_coin if flipper == 'P1' else p_P0_coin

        # 3. Jucatorul care incepe da cu zarul (n)
        n = np.random.randint(1, 7)    # Rezultat de la 1 la 6

        # 4. Celalalt jucator arunca moneda de 2n ori (m)
        num_trials = 2 * n
        m = np.random.binomial(num_trials, p_heads)

        # 5. Se determina castigatorul (starter castiga daca n >= m)
        if n >= m:
            winner = starter
        else:
            winner = flipper
            
        wins[winner] += 1

    # Afisarea rezultatelor simularii
    prob_P0_win = wins['P0'] / num_simulations
    prob_P1_win = wins['P1'] / num_simulations

    print("\n--- Rezultate Simulare ---")
    print(f"Victorii P0: {wins['P0']} ({prob_P0_win:.2%})")
    print(f"Victorii P1: {wins['P1']} ({prob_P1_win:.2%})")

    if prob_P1_win > prob_P0_win:
        print(f"\nEstimare: Jucatorul P1 are o sansa mai mare de a castiga ({prob_P1_win:.2%} > {prob_P0_win:.2%}).")
    elif prob_P0_win > prob_P1_win:
        print(f"\nEstimare: Jucatorul P0 are o sansa mai mare de a castiga ({prob_P0_win:.2%} > {prob_P1_win:.2%}).")
    else:
        print("\nEstimare: Jucatorii au sanse egale de a castiga.")


# ==============================================================================
# PART 2 & 3: Retea Bayesiana (pgmpy) si Inferenta
# ==============================================================================

def run_bayesian_network_analysis():
    """
    Construieste reteaua Bayesiana (Partea 2) si 
    efectueaza inferenta P(Start | M=1) (Partea 3).
    """
    print("\n" + "=" * 50)
    print("--- Reteaua Bayesiana si Inferenta (Partea 2 & 3) ---")

    # --- Definirea modelului si structurii (Partea 2) ---
    # CORECTIE: S-a folosit DiscreteBayesianNetwork in loc de BayesianNetwork
    model = DiscreteBayesianNetwork([('Start', 'M'), ('N', 'M')])

    # CPD pentru Start (S)
    cpd_start = TabularCPD(variable='Start', variable_card=2, 
                           values=[[0.5], [0.5]],
                           state_names={'Start': ['P0', 'P1']})

    # CPD pentru N (Zarul)
    cpd_n = TabularCPD(variable='N', variable_card=6,
                       values=[[1/6]] * 6,
                       state_names={'N': [1, 2, 3, 4, 5, 6]})

    # CPD pentru P(M | Start, N)
    m_states = 13  # M in {0, 1, ..., 12}
    n_states = 6
    start_states = 2

    # Initializam tabelul de valori (m_states, start_states * n_states)
    values = np.zeros((m_states, start_states * n_states))
    p_rigged = P_RIGGED  # P1 arunca moneda (p=4/7)
    p_fair = 0.5         # P0 arunca moneda (p=1/2)

    # Probabilitatea de heads a celui care arunca moneda, in functie de START:
    # Dacă S=P0, P1 aruncă (p_rigged). Dacă S=P1, P0 aruncă (p_fair).
    probabilities = {'P0': p_rigged, 'P1': p_fair} 
    state_names_start = ['P0', 'P1']
    state_names_n = [1, 2, 3, 4, 5, 6]
    state_names_m = list(range(m_states))

    col_index = 0
    for start_player in state_names_start:
        p_heads = probabilities[start_player] 
        
        for n in state_names_n:
            num_trials = 2 * n
            dist = binom(num_trials, p_heads)
            
            # Calculeaza P(M=m | Start, N=n)
            for m in state_names_m:
                if m <= num_trials:
                    values[m, col_index] = dist.pmf(m)
                else:
                    values[m, col_index] = 0.0 # m > 2n (imposibil)
            
            col_index += 1

    cpd_m = TabularCPD(variable='M', variable_card=m_states,
                       values=values,
                       evidence=['Start', 'N'],
                       evidence_card=[start_states, n_states],
                       state_names={'M': state_names_m,
                                    'Start': state_names_start,
                                    'N': state_names_n})

    model.add_cpds(cpd_start, cpd_n, cpd_m)

    if model.check_model():
        print("Modelul Bayesian (DiscreteBayesianNetwork) a fost construit si validat cu succes.")
    else:
        print("Eroare la validarea modelului!")
        return

    # --- Inferenta (Partea 3) ---
    print("\n--- Inferenta: P(Start | M=1) ---")
    print("Se calculeaza cine a inceput cel mai probabil, stiind ca M=1 (o singura 'heads')...")

    inference = VariableElimination(model)

    # Interogarea: P(Start | M=1)
    # Starea 'M' trebuie sa fie trecuta ca string sau index; in pgmpy, starea M=1 este la indexul 1.
    result = inference.query(variables=['Start'], evidence={'M': 1}) 

    print("\nRezultatul inferentei P(Start | M=1):")
    print(result)

    # Extragerea si compararea probabilitatilor
    # result.values[0] este pentru 'P0', result.values[1] pentru 'P1'
    prob_P0_started = result.values[0] 
    prob_P1_started = result.values[1] 

    print(f"\nP(Start=P0 | M=1) = {prob_P0_started:.4f}")
    print(f"P(Start=P1 | M=1) = {prob_P1_started:.4f}")

    if prob_P1_started > prob_P0_started:
        print(f"\nConcluzie (Partea 3): Este mai probabil ca jucatorul P1 sa fi inceput jocul (P={prob_P1_started:.4f}).")
    else:
        print(f"\nConcluzie (Partea 3): Este mai probabil ca jucatorul P0 sa fi inceput jocul (P={prob_P0_started:.4f}).")

if __name__ == "__main__":
    
    # Rularea tuturor partilor
    run_simulation(10000)
    
    run_bayesian_network_analysis()