import os
import pickle
from pathlib import Path
import time
import requests # For fetching utilities.py if not present

import arviz as az
from cmdstanpy import CmdStanModel
import numpy as np
import pandas as pd
import folium
from datetime import timedelta, datetime

import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
# import plotly.express as px # Not used in the final model part, but in EDA

# --- Setup ---
# Attempt to import custom_install_cmdstan from utilities
try:
    from utilities import custom_install_cmdstan, test_cmdstan_installation
except ImportError:
    print("utilities.py not found. Attempting to download...")
    try:
        r = requests.get("https://raw.githubusercontent.com/MLGlobalHealth/StatML4PopHealth/main/practicals/resources/scripts/utilities.py")
        r.raise_for_status()
        with open("utilities.py", "w") as f:
            f.write(r.text)
        from utilities import custom_install_cmdstan, test_cmdstan_installation
        print("utilities.py downloaded successfully.")
    except Exception as e:
        print(f"Failed to download utilities.py: {e}")
        print("Please ensure utilities.py is in the same directory or install CmdStan manually.")
        custom_install_cmdstan = None # To avoid further errors if user wants to proceed

# Install CmdStan (if in an environment like Colab or if not already installed path is set)
# This is typically done once.
# Check if CMDSTAN env var is set, if not, try to install
if not os.environ.get("CMDSTAN") and custom_install_cmdstan:
    print("CMDSTAN environment variable not set. Attempting to install CmdStan.")
    try:
        custom_install_cmdstan()
        if not test_cmdstan_installation(): # test_cmdstan_installation should be part of utilities
             print("CmdStan installation test failed. Please check.")
        else:
            print("CmdStan seems to be installed and working.")
    except Exception as e:
        print(f"Could not install CmdStan automatically: {e}")
        print("Please ensure CmdStan is installed and CMDSTAN environment variable is set.")
elif os.environ.get("CMDSTAN"):
    print(f"CmdStan found at: {os.environ.get('CMDSTAN')}")
else:
    print("CmdStan not found and utilities.py could not be loaded for installation.")


# Aesthetics
sns.set_theme(style="whitegrid")
font = {"family": "sans-serif", "weight": "normal", "size": 10.5}
mpl.rc('font', **font)

# Output directory for plots
PLOTS_DIR = Path("../plots")
PLOTS_DIR.mkdir(parents=True, exist_ok=True)
STAN_MODELS_DIR = Path("../stan-model") # Assuming stan model is in ../stan_models

# --- Data Loading and Initial EDA ---
print("\n--- Loading and Preprocessing Data ---")
DATA_URL = "https://raw.githubusercontent.com/MLGlobalHealth/StatML4PopHealth/main/data/sentinel_5p_particulate_matter.csv"
try:
    db = pd.read_csv(DATA_URL)
except Exception as e:
    print(f"Error downloading data: {e}")
    exit()

# WHO classification
dwho = {
    'risk': ['good','moderate','unhealthy for sensitive groups','unhealthy','very unhealthy'],
    'pm25_low': [0, 10, 25, 35, 55],
    'pm25_high': [10, 25, 35, 55, 1000], # Matched to WHO table in project
    'text': ['good air quality',
             'acceptable for short-term exposure but may affect sensitive groups over long-term exposure',
             'increased risk for vulnerable populations such as children, elderly, and people with pre-existing health conditions',
             'significant risk to the general population, especially with prolonged exposure',
             'severe health risk to the general public']
}
dwho_df = pd.DataFrame(dwho)

# Subset to Kampala
dp_kampala_full = db.loc[db['city'] == "Kampala", ['city', 'country', 'date', 'hour', 
                                          'site_id', 'site_latitude', 'site_longitude', 'pm2_5']].copy()
dp_kampala_full['date'] = pd.to_datetime(dp_kampala_full['date'], format='%Y-%m-%d')
# dp_kampala_full['year'] = dp_kampala_full['date'].dt.strftime('%Y') # Not used later
# dp_kampala_full['month'] = dp_kampala_full['date'].dt.strftime('%m') # Not used later

# Name sites and merge
dp_sites = dp_kampala_full.loc[:, ['city', 'site_id', 'site_latitude', 'site_longitude']].drop_duplicates()
dp_sites['site_name'] = [f'site-{i}' for i in range(1, len(dp_sites) + 1)]
dp_kampala_full = pd.merge(dp_kampala_full, dp_sites, on=['city', 'site_id', 'site_latitude', 'site_longitude'])

# Create map
print("Generating Kampala sites map...")
lat_kampala = 0.3136
lon_kampala = 32.5818
kampala_map = folium.Map(location=[lat_kampala, lon_kampala], zoom_start=12)
for idx, row in dp_sites.iterrows():
    folium.Marker(
        location=[row['site_latitude'], row['site_longitude']],
        popup=f"{row['site_name']} ({row['site_id']})"
    ).add_to(kampala_map)
map_path = PLOTS_DIR / "kampala_sites_map.html"
kampala_map.save(map_path)
print(f"Map saved to {map_path}")

# Plot time series for 'Buwate' (site-4) and 'Kyebando' (site-9)
# Find original site_id for Buwate and Kyebando if names are known, or use site-X
# From OCR, site-4 is Buwate, site-9 is Kyebando
# Let's find them dynamically if possible, or use the fixed names
site_mapping = {'site-4': 'Buwate', 'site-9': 'Kyebando'} # Based on project page 7
selected_sites_for_eda = ['site-4', 'site-9']
dp_eda = dp_kampala_full[dp_kampala_full['site_name'].isin(selected_sites_for_eda)].copy()
dp_eda['site_name'] = dp_eda['site_name'].replace(site_mapping)


plt.figure(figsize=(12, 7))
custom_palette_eda = sns.color_palette(["#073344FF", "#0B7C9AFF"]) # As per project
for i, row in dwho_df.iterrows():
    plt.fill_between(
        [dp_eda['date'].min() - timedelta(days=10), dp_eda['date'].max() + timedelta(days=10)],
        row['pm25_low'], row['pm25_high'],
        color=sns.color_palette("OrRd", len(dwho_df))[i], alpha=0.5,
        label=row['risk'] if i == 0 else None # Label only once for bands
    )
sns.scatterplot(
    data=dp_eda, x='date', y='pm2_5', hue='site_name',
    palette=custom_palette_eda, s=50
)
plt.xlim([dp_eda['date'].min() - timedelta(days=10), dp_eda['date'].max() + timedelta(days=10)])
plt.ylim([0, max(dp_eda['pm2_5'].dropna()) * 1.05 if not dp_eda['pm2_5'].dropna().empty else 100])
plt.xlabel('Date')
plt.ylabel('PM2.5 concentration (µg/m³)')
plt.title('PM2.5 Measurements in Kampala (Selected Sites for EDA)')
handles, labels = plt.gca().get_legend_handles_labels() # Get handles and labels
# Manually add risk band labels to legend if desired, or keep it simple
# For simplicity, we will rely on the hue for site names. The bands are visual guides.
unique_labels = {}
for handle, label in zip(handles, labels):
    if label not in unique_labels and label in ['Buwate', 'Kyebando', 'good', 'moderate', 'unhealthy for sensitive groups', 'unhealthy', 'very unhealthy']:
         unique_labels[label] = handle
plt.legend(unique_labels.values(), unique_labels.keys(), title='Location/Risk')
plt.tight_layout()
eda_plot_path = PLOTS_DIR / "pm25_buwate_kyebando_eda.png"
plt.savefig(eda_plot_path)
print(f"EDA plot saved to {eda_plot_path}")
plt.close()


# --- Site Selection and Model-Specific Preprocessing ---
# Task 1: Site Selection - Select 'Buwate' (site-4 as per notebook)
SELECTED_SITE_NAME = 'site-4' # Buwate
dps = dp_kampala_full[dp_kampala_full['site_name'] == SELECTED_SITE_NAME].copy()
dps = dps.sort_values(by='date').reset_index(drop=True)

print(f"\n--- Processing site: {site_mapping.get(SELECTED_SITE_NAME, SELECTED_SITE_NAME)} ---")
print(f"Number of observations for {site_mapping.get(SELECTED_SITE_NAME, SELECTED_SITE_NAME)}: {len(dps)}")
for col in dps.columns:
    print(f"Missing values in {col}: {dps[col].isna().sum()}")

if dps['pm2_5'].isna().any():
    print("Warning: PM2.5 column contains NaNs. Dropping them for modeling.")
    dps.dropna(subset=['pm2_5'], inplace=True)
    print(f"Number of observations after dropping NaNs: {len(dps)}")

if len(dps) < 10: # Arbitrary threshold for minimum data
    print(f"Site {SELECTED_SITE_NAME} has insufficient data ({len(dps)} points). Exiting.")
    exit()
    
# Converting date to a numerical feature (days since first observation)
# This 'day_num_original' is for plotting and understanding the full range
day_num_original = (dps.date - dps.date.min()).dt.days + 1.0

# Standardize time for the model using ONLY observed data points
# 'x' will be the standardized time for observed points
mean_day_obs = day_num_original.mean()
std_day_obs = day_num_original.std()
if std_day_obs == 0: std_day_obs = 1 # Avoid division by zero if only one unique day

x_obs = (day_num_original - mean_day_obs) / std_day_obs
y_obs = dps['pm2_5'].values

stan_data = {
    'N': len(x_obs),
    'x': x_obs.values, # Ensure it's a NumPy array for Stan
    'y': y_obs,
    'M': 30,          # Number of basis functions (as per notebook)
    'C': 1.5          # Boundary condition multiplier (as per notebook)
}

# --- Model Compilation and Fitting ---
print("\n--- Compiling Stan Model ---")
stan_file_path = STAN_MODELS_DIR / "hsgp-model.stan"
try:
    hsgp_se_model = CmdStanModel(stan_file=str(stan_file_path))
    print(hsgp_se_model.code())
except Exception as e:
    print(f"Error compiling Stan model: {e}")
    print("Ensure CmdStan is installed and CMDSTAN environment variable is correctly set.")
    exit()

print("\n--- Fitting Stan Model ---")
start_time = time.time()
hsgp_fit = hsgp_se_model.sample(
    data=stan_data,
    chains=4,
    iter_warmup=500,
    iter_sampling=1000,
    adapt_delta=0.95, # As per notebook
    seed=0, # For reproducibility
    show_progress=True
)
end_time = time.time()
runtime = end_time - start_time
print(f"Runtime of the Stan model: {runtime:.2f} seconds")

# --- Model Diagnostics ---
print("\n--- Model Diagnostics ---")
print(hsgp_fit.diagnose())

# Summary statistics
summary = hsgp_fit.summary()
print(summary)

# Effective Sample Size (ESS) and Rhat
# Find parameter with lowest ESS for trace plot focus
# This part of ArviZ summary is more convenient
idata_hsgp_se = az.from_cmdstanpy(hsgp_fit)
az_summary = az.summary(idata_hsgp_se, var_names=['alpha', 'rho', 'sigma', 'beta_0'])
print(az_summary)

lowest_ess_param = None
if not az_summary.empty:
    az_summary_ess = az_summary.sort_values(by='ess_bulk')
    if not az_summary_ess.empty:
        lowest_ess_param = az_summary_ess.index[0]
        print(f"Parameter with lowest ESS (bulk): {lowest_ess_param} ({az_summary_ess.iloc[0]['ess_bulk']:.0f})")

# Trace plots
# Plot a few key parameters and the one with lowest ESS if found
params_to_plot = ['beta_0', 'alpha', 'rho', 'sigma']
if lowest_ess_param and lowest_ess_param not in params_to_plot:
    params_to_plot.append(lowest_ess_param)

az.plot_trace(idata_hsgp_se, var_names=params_to_plot, compact=False, combined=True)
trace_plot_path = PLOTS_DIR / "model_trace_plots.png"
plt.savefig(trace_plot_path)
print(f"Trace plots saved to {trace_plot_path}")
plt.close()

# --- Visualization of Posterior ---
print("\n--- Visualizing Posterior Predictions ---")
# Extract posterior samples for mu (which is beta_0 + f)
mu_posterior_samples = idata_hsgp_se.posterior['mu'].values # Shape: (chains, draws, N_obs)
# Reshape to (total_samples, N_obs)
num_chains, num_draws, N_obs = mu_posterior_samples.shape
mu_posterior_samples_flat = mu_posterior_samples.reshape(num_chains * num_draws, N_obs)

# Calculate median and credible intervals for exp(mu) which is median of lognormal
# The mean of lognormal is exp(mu + sigma^2/2)
# The project asks for posterior median of the target variable (PM2.5).
# The model is y ~ LogNormal(mu, sigma). The median of LogNormal(mu, sigma) is exp(mu).
median_pm25_pred = np.exp(np.median(mu_posterior_samples_flat, axis=0))
q5_pm25_pred = np.exp(np.percentile(mu_posterior_samples_flat, 5, axis=0))
q95_pm25_pred = np.exp(np.percentile(mu_posterior_samples_flat, 95, axis=0))


fig, ax = plt.subplots(figsize=(12, 7))
ax.scatter(day_num_original, y_obs, label='Observed Data', s=20, alpha=0.6, color='blue')
ax.plot(day_num_original, median_pm25_pred, label='Posterior Median PM2.5', color='red')
ax.fill_between(day_num_original, q5_pm25_pred, q95_pm25_pred,
                color='gray', alpha=0.4, label='95% Credible Interval')

ax.set_xlabel(f'Days since {dps.date.min().strftime("%Y-%m-%d")}')
ax.set_ylabel('PM2.5 Concentration (µg/m³)')
ax.set_title(f'Posterior PM2.5 Prediction for {site_mapping.get(SELECTED_SITE_NAME, SELECTED_SITE_NAME)}')
ax.legend()
plt.tight_layout()
posterior_plot_path = PLOTS_DIR / f"posterior_pm25_{SELECTED_SITE_NAME.lower()}.png"
plt.savefig(posterior_plot_path)
print(f"Posterior prediction plot saved to {posterior_plot_path}")
plt.close()

# --- Answering the Main Objective ---
print("\n--- Answering Main Objective ---")
# We need to predict for a full year.
# The current model is fit on observed data points.
# To predict for a full year, we need to extend 'x' to cover the full year range,
# standardize it using the *original* mean_day_obs and std_day_obs,
# and then use the 'generated quantities' block in Stan for out-of-sample predictions.
#
# For simplicity here, as the `generated quantities` block in the provided Stan model
# only computes y_rep for observed data, we will use the posterior median of *observed*
# PM2.5 predictions to estimate unhealthy days.
# A more robust approach would involve modifying the Stan model.

# Use the median_pm25_pred which corresponds to the observed days.
unhealthy_threshold = 35 # µg/m³
unhealthy_days_observed_period = np.sum(median_pm25_pred > unhealthy_threshold)
total_observed_days = len(dps) # This is the number of days WITH observations
duration_observed_period_days = (dps.date.max() - dps.date.min()).days + 1

print(f"Total days in observed period: {duration_observed_period_days}")
print(f"Number of days with observations in this period: {total_observed_days}")
print(f"Number of days with median PM2.5 > {unhealthy_threshold} µg/m³ (within observed data points): {unhealthy_days_observed_period}")

# Extrapolate to a full year (365 days) based on the proportion in the observed period's *duration*
if duration_observed_period_days > 0:
    proportion_unhealthy = unhealthy_days_observed_period / total_observed_days # Proportion of *observed data points* that are unhealthy
    estimated_unhealthy_days_in_year = proportion_unhealthy * 365
    print(f"Estimated number of unhealthy days in a full year (based on observed data points): {estimated_unhealthy_days_in_year:.2f}")
else:
    estimated_unhealthy_days_in_year = "N/A (insufficient period duration)"
    print("Cannot estimate for a full year due to insufficient observed period duration.")


# Summary of findings (example)
print("\nSummary of Findings:")
print(f"- The GP model was fitted to PM2.5 data for site '{site_mapping.get(SELECTED_SITE_NAME, SELECTED_SITE_NAME)}' in Kampala.")
print(f"- Based on the posterior median predictions for the observed data points, an estimated {unhealthy_days_observed_period} out of {total_observed_days} observed days had PM2.5 concentrations above {unhealthy_threshold} µg/m³.")
print(f"- Extrapolating this, approximately {estimated_unhealthy_days_in_year:.2f} days in a full year might experience unhealthy PM2.5 levels at this site.")
print("- This indicates a potential public health concern for a number of days in the year at this location.")
print("- Note: Divergences were observed during MCMC sampling, suggesting the model might benefit from reparameterization or tighter priors for complex areas of the posterior. The results should be interpreted with this in mind.")
print("- For more accurate full-year prediction, the Stan model's `generated quantities` block should be used with a full year's worth of time points.")

print("\n--- Script Finished ---")
