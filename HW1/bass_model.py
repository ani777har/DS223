
# ============================================================================
# IMPORTS AND SETUP
# ============================================================================

import numpy as np
import pandas as pd
import matplotlib
# Use non-interactive backend for scripts so plt.show() doesn't block
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from sklearn.metrics import r2_score, mean_squared_error
import seaborn as sns
import warnings
import os
warnings.filterwarnings('ignore')

sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)
plt.rcParams['font.size'] = 11

print("✓ All libraries imported successfully!")
print("=" * 80)
print("BASS MODEL ANALYSIS: MASIMO W1 MEDICAL WATCH")
print("Predicting adoption based on Fitbit historical data (2010-2023)")
print("=" * 80)

os.makedirs('img', exist_ok=True)
os.makedirs('data', exist_ok=True)

csv_path = os.path.join('data', 'fitbit_sales.csv')
if os.path.exists(csv_path):
    try:
        df = pd.read_csv(csv_path)
        # Expect columns 'year' and 'sales'
        if not {'year', 'sales'}.issubset(df.columns):
            raise ValueError(f"CSV found at {csv_path} but required columns 'year' and 'sales' are missing. Columns: {list(df.columns)}")
        # Ensure correct dtypes and sort
        df = df[['year', 'sales']].copy()
        df['year'] = df['year'].astype(int)
        df['sales'] = pd.to_numeric(df['sales'], errors='coerce')
        if df['sales'].isnull().any():
            raise ValueError('Some sales values could not be converted to numeric in the CSV.')
        df = df.sort_values('year').reset_index(drop=True)
    except Exception as e:
        print(f"\n✗ Error reading CSV at {csv_path}: {e}")
        print("Falling back to embedded Fitbit sales data")
        df = None
else:
    df = None



# Convert to millions for easier interpretation
df['sales_millions'] = df['sales'] / 1000

# Create time variable (t = 0 for first year)
df['t'] = df['year'] - df['year'].min()

# Calculate cumulative sales
df['cumulative_sales'] = df['sales'].cumsum()
df['cumulative_millions'] = df['cumulative_sales'] / 1000

print("\n📊 FITBIT HISTORICAL DATA (2010-2023)")
print("=" * 80)
print(df[['year', 'sales', 'cumulative_sales']].to_string(index=False))
print(f"\nTotal cumulative sales: {df['cumulative_sales'].iloc[-1]:,.0f} thousand units")
print(f"                       = {df['cumulative_millions'].iloc[-1]:.2f} million units")


fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

# Annual sales
ax1.bar(df['year'], df['sales_millions'], color='steelblue', alpha=0.7, edgecolor='black')
ax1.set_xlabel('Year', fontsize=12, fontweight='bold')
ax1.set_ylabel('Annual Sales (millions)', fontsize=12, fontweight='bold')
ax1.set_title('Fitbit Annual Sales (2010-2023)', fontsize=14, fontweight='bold')
ax1.grid(axis='y', alpha=0.3)
ax1.set_xticks(df['year'])
ax1.tick_params(axis='x', rotation=45)

# Cumulative sales
ax2.plot(df['year'], df['cumulative_millions'], marker='o', linewidth=2.5, 
         markersize=8, color='darkgreen', label='Actual Cumulative Sales')
ax2.set_xlabel('Year', fontsize=12, fontweight='bold')
ax2.set_ylabel('Cumulative Sales (millions)', fontsize=12, fontweight='bold')
ax2.set_title('Fitbit Cumulative Sales (2010-2023)', fontsize=14, fontweight='bold')
ax2.grid(True, alpha=0.3)
ax2.legend()
ax2.set_xticks(df['year'])
ax2.tick_params(axis='x', rotation=45)

plt.tight_layout()
plt.savefig('img/fitbit_historical_data.png', dpi=300, bbox_inches='tight')
plt.close()

print("\n✓ Chart saved to: img/fitbit_historical_data.png")


def bass_model_cumulative(t, p, q, M):
    """
    Bass Model - Cumulative adoption function
    
    Parameters:
    -----------
    t : time period
    p : coefficient of innovation (external influence)
    q : coefficient of imitation (internal influence)
    M : market potential (total number of eventual adopters)
    
    Returns:
    --------
    N(t) : cumulative number of adopters at time t
    """
    try:
        numerator = 1 - np.exp(-(p + q) * t)
        denominator = 1 + (q / p) * np.exp(-(p + q) * t)
        return M * (numerator / denominator)
    except ZeroDivisionError:
        # If p == 0, fallback to midpoint; avoid silently catching other errors
        return M * 0.5

def bass_model_adoption_rate(t, p, q, M):
    """
    Bass Model - Adoption rate function (first derivative)
    
    Returns:
    --------
    n(t) : number of adopters at time t (not cumulative)
    """
    N_t = bass_model_cumulative(t, p, q, M)
    return p * M + (q - p) * N_t - (q / M) * N_t**2

print("\n✓ Bass Model functions defined successfully!")
print("\nBass Model Equations:")
print("  Cumulative: N(t) = M * [(1 - e^(-(p+q)t)) / (1 + (q/p) * e^(-(p+q)t))]")
print("  Rate:       n(t) = p*M + (q-p)*N(t) - (q/M)*N(t)²")

# ESTIMATE BASS MODEL PARAMETERS

def main():
    print("\n" + "=" * 80)
    print("PARAMETER ESTIMATION")
    print("=" * 80)

    t_data = df['t'].values
    N_data = df['cumulative_millions'].values
    p0 = [0.01, 0.4, N_data[-1] * 1.5]

    try:
        params, covariance = curve_fit(
            bass_model_cumulative, 
            t_data, 
            N_data, 
            p0=p0,
            maxfev=10000,
            bounds=([0.0001, 0.1, N_data[-1]], [0.1, 1.0, N_data[-1] * 5])
        )
        
        p_est, q_est, M_est = params
        
        # Calculate standard errors
        perr = np.sqrt(np.diag(covariance))
        p_err, q_err, M_err = perr
        
        print("\n✓ Parameter estimation successful!")
        print("\n📈 ESTIMATED PARAMETERS:")
        print(f"  p (innovation coefficient):  {p_est:.6f} ± {p_err:.6f}")
        print(f"  q (imitation coefficient):   {q_est:.6f} ± {q_err:.6f}")
        print(f"  M (market potential):        {M_est:.2f} ± {M_err:.2f} million units")
        
        print(f"\n📊 INTERPRETATION:")
        print(f"  - p/q ratio = {p_est/q_est:.4f} (q > p means imitation dominates)")
        print(f"  - Peak adoption at t* = {np.log(q_est/p_est)/(p_est + q_est):.1f} years")
        
        # Calculate fitted values
        N_fitted = bass_model_cumulative(t_data, p_est, q_est, M_est)
        
        # Calculate goodness of fit
        r2 = r2_score(N_data, N_fitted)
        rmse = np.sqrt(mean_squared_error(N_data, N_fitted))
        
        print(f"\n✓ MODEL FIT QUALITY:")
        print(f"  R² = {r2:.4f}")
        print(f"  RMSE = {rmse:.4f} million units")
        
        if r2 > 0.9:
            print("  ✓ Excellent fit (R² > 0.9)")
        elif r2 > 0.8:
            print("  ✓ Good fit (R² > 0.8)")
        else:
            print("  ⚠ Moderate fit (R² < 0.8)")
    except Exception as e:
        print(f"\n✗ Error in parameter estimation: {e}")
        print("Using default parameters...")
        p_est, q_est, M_est = 0.01, 0.4, 150
        p_err, q_err, M_err = np.nan, np.nan, np.nan
        r2, rmse = np.nan, np.nan
        N_fitted = bass_model_cumulative(df['t'].values, p_est, q_est, M_est)

    # VISUALIZE MODEL FIT

    # Generate smooth curve for plotting
    t_smooth = np.linspace(0, t_data[-1], 100)
    N_smooth = bass_model_cumulative(t_smooth, p_est, q_est, M_est)

    # Calculate annual adoption rates
    n_fitted = np.diff(N_fitted, prepend=0)
    n_actual = df['sales_millions'].values

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    # Cumulative adoption plot
    ax1.scatter(df['year'], N_data, s=100, color='blue', zorder=3, 
               label='Actual Data', edgecolors='black', linewidth=1.5)
    ax1.plot(df['year'].min() + t_smooth, N_smooth, 'r-', linewidth=2.5, 
            label=f'Bass Model Fit (R²={r2:.3f})', zorder=2)
    ax1.axhline(y=M_est, color='green', linestyle='--', linewidth=2, 
               label=f'Market Potential M = {M_est:.1f}M', alpha=0.7)
    ax1.set_xlabel('Year', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Cumulative Adopters (millions)', fontsize=12, fontweight='bold')
    ax1.set_title('Bass Model Fit: Fitbit Cumulative Adoption', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)

    # Annual adoption rate plot
    ax2.bar(df['year'], n_actual, alpha=0.6, label='Actual Annual Sales', 
           color='steelblue', edgecolor='black')
    ax2.plot(df['year'], n_fitted, 'ro-', linewidth=2, markersize=6, 
            label='Model Predicted', zorder=3)
    ax2.set_xlabel('Year', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Annual Adopters (millions)', fontsize=12, fontweight='bold')
    ax2.set_title('Annual Adoption Rate: Actual vs Predicted', fontsize=14, fontweight='bold')
    ax2.legend(fontsize=10)
    ax2.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig('img/bass_model_fit.png', dpi=300, bbox_inches='tight')
    plt.close()

    print("\n✓ Chart saved to: img/bass_model_fit.png")

    # PREDICT MASIMO W1 ADOPTION

    print("\n" + "=" * 80)
    print("PREDICTIONS FOR MASIMO W1 MEDICAL WATCH (2024-2038)")
    print("=" * 80)

    # Estimate market potential for Masimo W1
    # Method: Use Fermi estimation based on target market

    print("\n📊 MARKET POTENTIAL ESTIMATION FOR MASIMO W1:")
    print("=" * 80)

    # Global approach
    global_smartwatch_market = 500  # million units (2023)
    medical_grade_segment = 0.10  # 10% of smartwatch market
    M_masimo_global = global_smartwatch_market * medical_grade_segment

    print(f"\nGlobal Market Approach:")
    print(f"  - Global smartwatch market: ~{global_smartwatch_market}M units")
    print(f"  - Medical-grade segment: {medical_grade_segment*100}%")
    print(f"  - Estimated M for Masimo W1: {M_masimo_global:.1f}M units")

    # Use the estimated parameters from Fitbit
    print(f"\n📈 USING BASS MODEL PARAMETERS FROM FITBIT:")
    print(f"  - p = {p_est:.6f}")
    print(f"  - q = {q_est:.6f}")
    print(f"  - M = {M_masimo_global:.1f} million units (estimated for Masimo W1)")

    # Generate predictions for 15 years (2024-2038)
    prediction_years = 15
    t_predict = np.arange(0, prediction_years)
    years_predict = np.arange(2024, 2024 + prediction_years)

    # Calculate cumulative and annual adoption
    N_masimo = bass_model_cumulative(t_predict, p_est, q_est, M_masimo_global)
    n_masimo = np.diff(N_masimo, prepend=0)

    # Create prediction dataframe
    df_predict = pd.DataFrame({
        'Year': years_predict,
        't': t_predict,
        'Annual_Adopters_millions': n_masimo,
        'Cumulative_Adopters_millions': N_masimo,
        'Market_Penetration_%': (N_masimo / M_masimo_global) * 100
    })

    print("\n" + "=" * 80)
    print("MASIMO W1 ADOPTION FORECAST")
    print("=" * 80)
    print(df_predict.to_string(index=False))

    # Find peak year
    peak_year_idx = np.argmax(n_masimo)
    peak_year = years_predict[peak_year_idx]
    peak_sales = n_masimo[peak_year_idx]

    print(f"\n🎯 KEY MILESTONES:")
    print(f"  - Peak adoption year: {peak_year}")
    print(f"  - Peak annual sales: {peak_sales:.2f} million units")
    print(f"  - Time to 50% market penetration: ~{np.argmax(N_masimo >= M_masimo_global*0.5)} years")
    print(f"  - Cumulative sales by 2038: {N_masimo[-1]:.2f} million units")
    print(f"  - Market penetration by 2038: {(N_masimo[-1]/M_masimo_global)*100:.1f}%")

    # VISUALIZE MASIMO W1 PREDICTIONS

    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))

    # 1. Cumulative adoption curve
    ax1.plot(years_predict, N_masimo, linewidth=3, color='darkblue', marker='o', markersize=6)
    ax1.axhline(y=M_masimo_global, color='red', linestyle='--', linewidth=2, 
               label=f'Market Potential = {M_masimo_global:.0f}M', alpha=0.7)
    ax1.fill_between(years_predict, 0, N_masimo, alpha=0.2, color='blue')
    ax1.set_xlabel('Year', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Cumulative Adopters (millions)', fontsize=12, fontweight='bold')
    ax1.set_title('Masimo W1: Predicted Cumulative Adoption', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)

    # 2. Annual adoption rate
    ax2.bar(years_predict, n_masimo, color='steelblue', edgecolor='black', alpha=0.7)
    ax2.axvline(x=peak_year, color='red', linestyle='--', linewidth=2, 
               label=f'Peak Year: {peak_year}', alpha=0.7)
    ax2.set_xlabel('Year', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Annual Adopters (millions)', fontsize=12, fontweight='bold')
    ax2.set_title('Masimo W1: Annual Adoption Rate', fontsize=14, fontweight='bold')
    ax2.legend(fontsize=10)
    ax2.grid(axis='y', alpha=0.3)

    # 3. Market penetration
    ax3.plot(years_predict, (N_masimo/M_masimo_global)*100, linewidth=3, 
            color='darkgreen', marker='s', markersize=6)
    ax3.axhline(y=50, color='orange', linestyle='--', linewidth=2, 
               label='50% Penetration', alpha=0.7)
    ax3.fill_between(years_predict, 0, (N_masimo/M_masimo_global)*100, alpha=0.2, color='green')
    ax3.set_xlabel('Year', fontsize=12, fontweight='bold')
    ax3.set_ylabel('Market Penetration (%)', fontsize=12, fontweight='bold')
    ax3.set_title('Masimo W1: Market Penetration Over Time', fontsize=14, fontweight='bold')
    ax3.legend(fontsize=10)
    ax3.grid(True, alpha=0.3)
    ax3.set_ylim(0, 105)

    # 4. Comparison: Fitbit vs Masimo W1 (normalized)
    t_fitbit_extended = np.arange(0, 15)
    N_fitbit_extended = bass_model_cumulative(t_fitbit_extended, p_est, q_est, M_est)
    ax4.plot(t_fitbit_extended, (N_fitbit_extended/M_est)*100, linewidth=2.5, 
            label='Fitbit (Historical)', color='purple', marker='o', markersize=5)
    ax4.plot(t_predict, (N_masimo/M_masimo_global)*100, linewidth=2.5, 
            label='Masimo W1 (Predicted)', color='teal', marker='s', markersize=5)
    ax4.set_xlabel('Years Since Launch', fontsize=12, fontweight='bold')
    ax4.set_ylabel('Market Penetration (%)', fontsize=12, fontweight='bold')
    ax4.set_title('Adoption Curves: Fitbit vs Masimo W1 (Normalized)', fontsize=14, fontweight='bold')
    ax4.legend(fontsize=10)
    ax4.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('img/masimo_predictions.png', dpi=300, bbox_inches='tight')
    plt.close()

    print("\n✓ Chart saved to: img/masimo_predictions.png")

    # SENSITIVITY ANALYSIS

    print("\n" + "=" * 80)
    print("SENSITIVITY ANALYSIS")
    print("=" * 80)

    # Test different scenarios
    scenarios = {
        'Pessimistic': {'M': M_masimo_global * 0.5, 'p': p_est * 0.8, 'q': q_est * 0.8},
        'Base Case': {'M': M_masimo_global, 'p': p_est, 'q': q_est},
        'Optimistic': {'M': M_masimo_global * 1.5, 'p': p_est * 1.2, 'q': q_est * 1.2}
    }

    plt.figure(figsize=(14, 7))

    for scenario_name, params in scenarios.items():
        N_scenario = bass_model_cumulative(t_predict, params['p'], params['q'], params['M'])
        plt.plot(years_predict, N_scenario, linewidth=2.5, marker='o', 
                label=f"{scenario_name} (M={params['M']:.0f}M)", markersize=5)

    plt.xlabel('Year', fontsize=12, fontweight='bold')
    plt.ylabel('Cumulative Adopters (millions)', fontsize=12, fontweight='bold')
    plt.title('Sensitivity Analysis: Different Market Scenarios', fontsize=14, fontweight='bold')
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('img/sensitivity_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()

    print("\n✓ Chart saved to: img/sensitivity_analysis.png")

    # Print scenario comparison
    print("\n📊 SCENARIO COMPARISON (Year 2030):")
    print("=" * 80)
    for scenario_name, params in scenarios.items():
        N_2030 = bass_model_cumulative(6, params['p'], params['q'], params['M'])  # 2030 is t=6
        penetration = (N_2030 / params['M']) * 100
        print(f"{scenario_name:12s}: {N_2030:6.2f}M units ({penetration:5.1f}% penetration)")

    # EXPORT RESULTS

    print("\n" + "=" * 80)
    print("EXPORTING RESULTS")
    print("=" * 80)

    # Save predictions to CSV
    df_predict.to_csv('data/masimo_predictions.csv', index=False)
    print("✓ Predictions saved to: data/masimo_predictions.csv")

    # Save parameters to file
    with open('data/bass_model_parameters.txt', 'w') as f:
        f.write("BASS MODEL PARAMETERS\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"Estimated from: Fitbit sales data (2010-2023)\n")
        f.write(f"Applied to: Masimo W1 Medical Watch\n\n")
        f.write(f"p (innovation): {p_est:.6f} ± {p_err:.6f}\n")
        f.write(f"q (imitation):  {q_est:.6f} ± {q_err:.6f}\n")
        f.write(f"M (Fitbit):     {M_est:.2f} ± {M_err:.2f} million\n")
        f.write(f"M (Masimo W1):  {M_masimo_global:.2f} million\n\n")
        f.write(f"Model Fit Quality:\n")
        f.write(f"R² = {r2:.4f}\n")
        f.write(f"RMSE = {rmse:.4f} million units\n")

    print("✓ Parameters saved to: data/bass_model_parameters.txt")

    print("\n" + "=" * 80)
    print("✓ ANALYSIS COMPLETE!")
    print("=" * 80)
    print("\nGenerated files:")
    print("  📊 img/fitbit_historical_data.png")
    print("  📊 img/bass_model_fit.png")
    print("  📊 img/masimo_predictions.png")
    print("  📊 img/sensitivity_analysis.png")
    print("  📄 data/masimo_predictions.csv")
    print("  📄 data/bass_model_parameters.txt")
   

if __name__ == '__main__':
    main()