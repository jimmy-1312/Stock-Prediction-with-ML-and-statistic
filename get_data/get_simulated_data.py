"""
introduction: using this to get pattern + noises, each data point represent a date
pattern choice: -- | noises distribution choice: --

input:(N_days:how many days do you want, variation:variation of the noises)
output: data(simulate the price) in numpy format, shape of (N_days,)


To be improved:
Design choice
--> use tensor instead of array? or turn them all to tensor at once
def generate_pattern()
--> try expect can be improved? like only use raise error?
--> use function application directly toward array (faster)
def generate_noises()
--> use pytorch?

"""

import numpy as np
import pandas as pd
import math
from pathlib import Path
import matplotlib.pyplot as plt

def generate_pattern(N_days : int, difficulty : str) -> np.array:
    """equation
    Easy = t**(1/2)
    Medium = t**(1/2) + sin(t)
    Hard = t**(1/2) + sin(t) + 1/(tan(t)+1)
    """
    
    equations = {
        "easy":lambda t:t**(1/2), 
        "medium":lambda t:t**(1/2) + math.sin(t),
        "hard":lambda t:t**(1/2) + math.sin(t) + 1/(t+1),
        "test":lambda t:math.sin(t/2)
        }

    try:
        equation = equations[difficulty]
    except NameError:
        raise NameError
    except Exception as e:
        print(f'Other error:{e}')
    
    pattern = np.zeros(N_days)
    for t in range(N_days):
        pattern[t] = equation(t)
    return pattern

def generate_noises(N_days : int, variation : float) -> np.array:
    """
    Set noises ~ N(0, variation)
    """

    rng = np.random.default_rng()

    noises = rng.normal(loc = 0, scale = variation, size = N_days)

    return noises

def generate_data(N_days : int, variation : float, difficulty : str) -> np.array:
    
    pattern = generate_pattern(N_days , difficulty)
    noises = generate_noises(N_days, variation)

    return pattern + noises

def create_csv_data(N_days, variation, difficulty, store_path):
    df = generate_data(N_days, variation, difficulty)
    df = pd.DataFrame(df,index=[f'day{i}' for i in range(N_days)],columns=["price"])
    df.to_csv(store_path)

def check_csv_data(path):
    path = Path(path)
    return path.exists()

if __name__ == "__main__":
    # pattern = generate_pattern(100,"easy")
    # plt.plot(pattern)
    # noises = generate_noises(100,1)
    # plt.plot(noises)
    # data = generate_data(100, 1, "easy")
    # plt.plot(data)
    # plt.show()
    # create_csv_data(N_days=1000,variation=0,difficulty="test",store_path="./save/data/test.csv")
    if not check_csv_data("./save/data/test.csv"):
        print("yes")
