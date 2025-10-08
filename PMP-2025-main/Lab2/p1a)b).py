import random

def sim1():
    red1, blue1, black1 = 3, 4, 2

    dieRoll = random.randint(1, 6)

    red2, blue2, black2 = red1, blue1, black1

    if dieRoll in [2, 3, 5]:
        black2=black2+1
    elif dieRoll == 6:
        red2=red2+1
    else:blue2=blue2+1

    urn = ['red'] * red2 + ['blue'] * blue2 + ['black'] * black2

    drawnBall = random.choice(urn)

    return drawnBall == 'red'

N = 100000
redDraws = sum(sim1() for _ in range(N))

probEstimata = redDraws / N

print(f"Urna curenta:3Red,4Blue,2Black.Total:9")
print(f"Nr de simulari N:{N}")
print(f"Nr de ori in care bila rosie a fost trasa:{redDraws}")
print(f"Probabilitatea estimata de a trage o bila rosie:{probEstimata}")



