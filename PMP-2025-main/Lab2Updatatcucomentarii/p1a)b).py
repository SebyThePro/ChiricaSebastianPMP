import random

# Explicatie: Functia simuleaza un singur ciclu al experimentului (Ex. 1a)
def sim1():
    # Stare initiala a urnei: 3 Rosii, 4 Albastre, 2 Negre
    red1, blue1, black1 = 3, 4, 2

    # Simuleaza aruncarea zarului (numar aleatoriu intre 1 si 6)
    dieRoll = random.randint(1, 6)

    # Variabile pentru noua stare a urnei dupa modificare
    red2, blue2, black2 = red1, blue1, black1

    # Modificarea urnei in functie de rezultatul zarului:
    # Cazul 1: Număr Prim (2, 3, 5) -> Adauga 1 Neagra
    if dieRoll in [2, 3, 5]:
        black2=black2+1
    # Cazul 2: Rolul 6 -> Adauga 1 Rosie
    elif dieRoll == 6:
        red2=red2+1
    # Cazul 3: Alte cazuri (1, 4) -> Adauga 1 Albastra
    else:
        blue2=blue2+1

    # Creeaza o lista reprezentand bilele din urna modificata (Total: 10)
    urn = ['red'] * red2 + ['blue'] * blue2 + ['black'] * black2

    # Simuleaza extragerea unei bile aleatoare
    drawnBall = random.choice(urn)

    # Returneaza True daca bila extrasa este Rosie (Succes)
    return drawnBall == 'red'

# Setarea numarului mare de simulari pentru estimarea probabilitatii (Ex. 1b)
N = 100000

# Ruleaza simularea de N ori si insumeaza rezultatele (True = 1, False = 0)
redDraws = sum(sim1() for _ in range(N))

# Calculeaza probabilitatea estimata (Frecventa relativa)
probEstimata = redDraws / N

# Afisarea rezultatelor simularii
print(f"Urna curenta:3Red,4Blue,2Black.Total:9")
print(f"Nr de simulari N:{N}")
print(f"Nr de ori in care bila rosie a fost trasa:{redDraws}")
print(f"Probabilitatea estimata de a trage o bila rosie:{probEstimata}")