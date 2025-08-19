import json
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize

size = 500

fig, axs = plt.subplots(1, 2, figsize=(8, 4))

### Left sub-figure: Mean-Variance linear fitting ###
with open(rf'Noise_model_bright.json', 'r') as f:
    noise_data = json.load(f)

R_mean_list = np.array(noise_data['R_mean_list'])
G_mean_list = np.array(noise_data['G_mean_list'])
B_mean_list = np.array(noise_data['B_mean_list'])
R_var_list = np.array(noise_data['R_var_list'])
G_var_list = np.array(noise_data['G_var_list'])
B_var_list = np.array(noise_data['B_var_list'])

def fit_linear_with_b_constraint(x, y):
    def loss(params):
        k, b = params
        return np.sum((y - (k * x + b)) ** 2)
    k_init, b_init = np.polyfit(x, y, 1)
    initial_guess = [k_init, max(b_init, 1e-8)]
    bounds = [(None, None), (1e-8, None)]
    res = minimize(loss, initial_guess, bounds=bounds)
    return res.x

def plot_and_fit_linear(ax, x, y, color, label):
    k, b = fit_linear_with_b_constraint(x, y)
    y_fit = k * x + b
    ax.scatter(x, y, alpha=0.5, label=f'{label} Measurement', color=color)
    ax.plot(np.sort(x), y_fit[np.argsort(x)], color=color, linestyle='--',
            label=f'Fit: $k_c$ = {k:.4f}')
    return k, b

ax1 = axs[0]
k_r, b_r = plot_and_fit_linear(ax1, R_mean_list, R_var_list, 'red', 'Red')
k_g, b_g = plot_and_fit_linear(ax1, G_mean_list, G_var_list, 'green', 'Green')
k_b, b_b = plot_and_fit_linear(ax1, B_mean_list, B_var_list, 'blue', 'Blue')

ax1.set_xlabel('Mean of RAW value $\mu_{I_c}(p)$', fontsize=14)
ax1.set_ylabel('Variance of RAW Value $\sigma^2_{I_c}(p)$', fontsize=14)
ax1.legend()
ax1.grid(True)

parameter_json_dict = {
    'R_fit_params': {'k': k_r, 'b': b_r},
    'G_fit_params': {'k': k_g, 'b': b_g},
    'B_fit_params': {'k': k_b, 'b': b_b}
}
with open(f'Fit_parameters_bright.json', 'w') as f:
    json.dump(parameter_json_dict, f, indent=4)

### Right sub-image: Read/ADC Noise model fitting ###
with open(rf'Noise_model_dark.json', 'r') as f:
    noise_data = json.load(f)
with open(f'Fit_parameters_bright.json', 'r') as f:
    fit_parameter = json.load(f)

k_r = fit_parameter["R_fit_params"]["k"]
k_g = fit_parameter["G_fit_params"]["k"]
k_b = fit_parameter["B_fit_params"]["k"]

ISO_list = np.array(noise_data['ISO_list'])
up_ISO = 510
g_list = ISO_list / 100
g_list = g_list[ISO_list < up_ISO]
R_mean_list = np.array(noise_data['R_mean_list'])[ISO_list < up_ISO]
G_mean_list = np.array(noise_data['G_mean_list'])[ISO_list < up_ISO]
B_mean_list = np.array(noise_data['B_mean_list'])[ISO_list < up_ISO]
R_var_list = np.array(noise_data['R_var_list'])[ISO_list < up_ISO]
G_var_list = np.array(noise_data['G_var_list'])[ISO_list < up_ISO]
B_var_list = np.array(noise_data['B_var_list'])[ISO_list < up_ISO]

def fit_sigma_params(x, y, mean, k_c):
    def loss(params):
        Sigma_read, Sigma_adc = params
        pred = Sigma_read ** 2 * k_c ** 2 * x ** 2 + k_c * mean * x + Sigma_adc ** 2 * k_c ** 2
        return np.sum((y - pred) ** 2)
    initial_guess = [1.0, 1.0]
    bounds = [(1e-8, None), (1e-8, None)]
    res = minimize(loss, initial_guess, bounds=bounds)
    return res.x

def plot_and_fit_noise(ax, g_list, V_list, mean, k_c, color, label):
    Sigma_read, Sigma_adc = fit_sigma_params(g_list, V_list, mean, k_c)
    y_fit = Sigma_read ** 2 * k_c ** 2 * g_list ** 2 + k_c * mean * g_list + Sigma_adc ** 2 * k_c ** 2
    ax.scatter(g_list, V_list, alpha=0.5, label=f'{label} Measurement', color=color)
    sorted_indices = np.argsort(g_list)
    ax.plot(g_list[sorted_indices], y_fit[sorted_indices], linestyle='--', color=color,
            label=rf'Fit: $\sigma_{{read}}$={Sigma_read:.4f}, $\sigma_{{adc}}$={Sigma_adc:.4f}')
    return Sigma_read, Sigma_adc

ax2 = axs[1]
sigma_r_r, sigma_adc_r = plot_and_fit_noise(ax2, g_list, R_var_list, R_mean_list, k_r, 'red', 'Red')
sigma_r_g, sigma_adc_g = plot_and_fit_noise(ax2, g_list, G_var_list, G_mean_list, k_g, 'green', 'Green')
sigma_r_b, sigma_adc_b = plot_and_fit_noise(ax2, g_list, B_var_list, B_mean_list, k_b, 'blue', 'Blue')

ax2.set_xlabel('Gain $g$ = ISO / 100', fontsize=14)
ax2.set_ylabel('Variance of RAW Value $\sigma^2(I_c(p))$', fontsize=14)
ax2.legend()
ax2.grid(True)

parameter_json_dict = {
    'R_fit_params': {'Sigma_read': sigma_r_r, 'Sigma_adc': sigma_adc_r},
    'G_fit_params': {'Sigma_read': sigma_r_g, 'Sigma_adc': sigma_adc_g},
    'B_fit_params': {'Sigma_read': sigma_r_b, 'Sigma_adc': sigma_adc_b}
}
with open(f'Fit_parameters_dark.json', 'w') as f:
    json.dump(parameter_json_dict, f, indent=4)

plt.tight_layout(pad=0.5)
plt.show()
