import numpy as np

# Given data
z_score = 1.96  # for 95% confidence level
n = 30  # number of repetitions

# Original dataset
mu_original =  0.7112010717391968
std_original = 0.012070529919529182
margin_error_original = z_score * (std_original / np.sqrt(n))
ci_original = (mu_original - margin_error_original, mu_original + margin_error_original)

print(ci_original)