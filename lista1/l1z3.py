import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import pandas as pd
from scipy import stats
from typing import Callable, List, Tuple
from sklearn.linear_model import LinearRegression


# Zad 1

# blad bezwzgledny
def oblicz_mse(y_prawdziwe, y_przewidywane):
    n = len(y_prawdziwe)
    mse = np.sum((y_prawdziwe - y_przewidywane) ** 2) / n
    return mse



def oblicz_r2(y_prawdziwe, y_przewidywane):
    ssr = np.sum((y_przewidywane - np.mean(y_prawdziwe)) ** 2)
    sst = np.sum((y_prawdziwe - np.mean(y_prawdziwe)) ** 2)
    r2 = (ssr / sst)
    return r2


def oblicz_mae(y_prawdziwe, y_przewidywane):
    n = len(y_prawdziwe)
    mae = np.sum(np.abs(y_prawdziwe - y_przewidywane)) / n
    return mae

# Zad 2

# Funkcja do obliczania średniej ruchomej wg. zadanej formuły
def custom_moving_average(data: pd.Series, p: int) -> pd.Series:
    ma_values = []
    n = len(data)
    window_size = 2 * p + 1

    # Iterujemy przez dane, ale tylko od p do n - p
    for t in range(p, n - p):
        window_sum = 0

        # Sumujemy wartości od -p do p dla bieżącego punktu t
        for j in range(-p, p + 1):
            window_sum += data[t + j]

        # Obliczamy średnią dla tego punktu
        ma_value = window_sum / window_size
        ma_values.append(ma_value)

    # Uzupełniamy początkowe i końcowe wartości NaN
    ma_values = [None] * p + ma_values + [None] * p
    return pd.Series(ma_values)


# Wczytaj dane z pliku tekstowego
file_path = 'zad2_lista1.txt'  # Zmień na rzeczywistą ścieżkę
data = pd.read_csv(file_path, header=None, names=['values'])  # Odczyt danych z pliku

# Kolumna z danymi
values = data['values']


# Funkcja do obliczania prostej średniej ruchomej
def moving_average(data: pd.Series, window_size: int) -> pd.Series:
    return data.rolling(window=window_size, center=True).mean()

# Wczytaj dane z pliku tekstowego
file_path = 'zad2_lista1.txt'  # Zmień na rzeczywistą ścieżkę
data = pd.read_csv(file_path, header=None, names=['values'])  # Odczyt danych z pliku


# Zad 3

# Wczytaj dane
X = np.loadtxt('zad2_lista1.txt')
Y = np.loadtxt('zad3_lista1.txt')


# Funkcja do obliczenia współczynnika nachylenia (b1) i wyrazu wolnego (b0)
def oblicz_regresje(X, Y):
    # Obliczenie średnich X i Y
    X_mean = np.mean(X)
    Y_mean = np.mean(Y)

    # Obliczenie współczynnika b1
    s1 = np.sum((X - X_mean) * (Y - Y_mean))
    s2 = np.sum((X - X_mean) ** 2)
    b1 = s1 / s2

    # Obliczenie współczynnika b0
    b0 = Y_mean - b1 * X_mean

    return b0, b1

b0, b1 = oblicz_regresje(X, Y)

print(f"Współczynnik b0: {b0}")
print(f"Współczynnik b1: {b1}")

# Rysowanie wykresu
plt.scatter(X, Y, label='Dane', color='blue')
plt.plot(X, b0 + b1 * X, label='Regresja liniowa', color='red')
plt.legend()
plt.title(f'Regresja liniowa')
plt.xlabel('X (zmienna objaśniająca)')
plt.ylabel('Y (zmienna objaśniana)')
plt.grid(True)
#plt.show()

x_smooth = custom_moving_average(X, 6)
y_smooth = custom_moving_average(Y, 6)
b00, b11 = oblicz_regresje(x_smooth, y_smooth)
print(b00, b11)
# Rysowanie wykresu
plt.scatter(x_smooth, y_smooth, label='Dane', color='blue')
plt.plot(x_smooth, b00 + b11 * x_smooth, label='Regresja liniowa', color='red')
plt.legend()
plt.title(f'Regresja liniowa po wygładzeniu')
plt.xlabel('X (zmienna objaśniająca)')
plt.ylabel('Y (zmienna objaśniana)')
plt.grid(True)
#plt.show()

## zadanie 4
x4 = np.loadtxt(r"C:\Users\Maria Nowacka\Desktop\maria\5 semestr\metody_num\l1z3.py\zad4_lista1.txt")

