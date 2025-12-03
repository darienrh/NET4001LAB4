import numpy as np
import matplotlib.pyplot as plt



def lcg(n, X0, A, C, m):
    """Generate n Uniform(0,1) random numbers using LCG."""
    xs = np.zeros(n)
    x = X0
    for i in range(n):
        x = (A * x + C) % m
        xs[i] = x / m
    return xs


def compute_stats(arr):
    """Return mean, variance, std deviation."""
    return np.mean(arr), np.var(arr), np.std(arr)


def plot_hist(arr, title, bins=30):
    plt.hist(arr, bins=bins)
    plt.title(title)
    plt.xlabel("Value")
    plt.ylabel("Occurrences")
    plt.show()


#4.1

X0 = 7
A = 11
C = 0
m = 1024

u_1024 = lcg(1024, X0, A, C, m)
plot_hist(u_1024, "Q4.1 Histogram (1024 samples)")

mean1, var1, std1 = compute_stats(u_1024)
print("Q4.1 Stats (1024 samples):")
print("Mean =", mean1)
print("Variance =", var1)
print("Std Dev =", std1)

print("\nMaximum possible period =", m)
print("Achieved period =", len(np.unique(u_1024))) 

#4.2

u_large = lcg(102400, X0, A, C, m)
plot_hist(u_large, "Q4.2 Histogram (102,400 samples)")

mean2, var2, std2 = compute_stats(u_large)
print("\nQ4.2 Stats (102,400 samples):")
print("Mean =", mean2)
print("Variance =", var2)
print("Std Dev =", std2)

#4.4
X0 = 7
A = 25214903917
C = 11
m = 2**48

u_practical = lcg(1024, X0, A, C, m)
plot_hist(u_practical, "Q4.4 Histogram (1024 samples, practical LCG)")

mean3, var3, std3 = compute_stats(u_practical)
print("\nQ4.4 Stats (1024 samples):")
print("Mean =", mean3)
print("Variance =", var3)
print("Std Dev =", std3)

# Repeat with 102,400
u_practical_large = lcg(102400, X0, A, C, m)
plot_hist(u_practical_large, "Q4.4 Histogram (102,400 samples, practical LCG)")

mean4, var4, std4 = compute_stats(u_practical_large)
print("\nQ4.4 Stats (102,400 samples):")
print("Mean =", mean4)
print("Variance =", var4)
print("Std Dev =", std4)


#4.6

lam = 3
u_exp = u_practical_large  # re-use the uniform generator
exp_samples = -np.log(1 - u_exp) / lam

plot_hist(exp_samples, "Q4.6 Exponential via Inverse Transform")

mean_exp, var_exp, std_exp = compute_stats(exp_samples)
print("\nQ4.6 Exponential Stats:")
print("Mean =", mean_exp)
print("Variance =", var_exp)
print("Std Dev =", std_exp)


#4.7
np.random.seed(5)
exp_np = np.random.exponential(scale=1/lam, size=102400)

plot_hist(exp_np, "Q4.7 NumPy Exponential (λ = 3)")

mean_np, var_np, std_np = compute_stats(exp_np)
print("\nQ4.7 Exponential Stats:")
print("Mean =", mean_np)
print("Variance =", var_np)
print("Std Dev =", std_np)

print("\nTHEORETICAL VALUES for λ = 3:")
print("Mean =", 1/lam)
print("Variance =", 1/(lam**2))
print("Std Dev =", 1/lam)
