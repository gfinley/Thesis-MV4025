import random

def do_experiment(seed, number):
    # How many 18's are rolled in a set of six 6-sided die rolls
    histogram = [0] * 7
    random.seed(seed)
    for i in range(number):
        count = 0
        for j in range(6):
            roll = random.randint(1,6) + random.randint(1,6) + random.randint(1,6)
            if roll==18:
                count += 1
        histogram[count] += 1
    return histogram


