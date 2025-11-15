import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from lifelines import WeibullAFTFitter, LogNormalAFTFitter, LogLogisticAFTFitter
from sklearn.preprocessing import LabelEncoder
import warnings
warnings.filterwarnings('ignore')

plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

# load the data

def load_and_prepare_data(filepath='./data/telco.csv'):
    """Load and preprocess the telecom churn data"""
    df = pd.read_csv(filepath)
    
    print("=" * 80)
    print("DATA PREPARATION")
    print("=" * 80)
    print(f"Total records: {len(df)}")
    print(f"\nColumns: {df.columns.tolist()}")
    print(f"\nChurn distribution:\n{df['churn'].value_counts()}")
    print(f"Churn rate: {df['churn'].value_counts(normalize=True)['Yes']*100:.2f}%")
    
    df['churn_binary'] = (df['churn'] == 'Yes').astype(int)
    
    le = LabelEncoder()
    
    categorical_cols = ['region', 'marital', 'ed', 'retire', 'gender', 
                       'voice', 'internet', 'forward', 'custcat']
    
    for col in categorical_cols:
        df[f'{col}_encoded'] = le.fit_transform(df[col])
    
    print(f"\nTenure statistics:\n{df['tenure'].describe()}")
    print(f"\nAge statistics:\n{df['age'].describe()}")
    print(f"\nIncome statistics:\n{df['income'].describe()}")
    
    return df


# now we will fit AFT models

def fit_aft_models(df):
    """Fit AFT models with different distributions"""
    
    duration_col = 'tenure'
    event_col = 'churn_binary'
    
    feature_cols = ['age', 'income', 'address', 'region_encoded', 'marital_encoded', 
                    'ed_encoded', 'retire_encoded', 'gender_encoded', 'voice_encoded', 
                    'internet_encoded', 'forward_encoded', 'custcat_encoded']
    
    model_data = df[[duration_col, event_col] + feature_cols].copy()
    
    models = {
        'Weibull': WeibullAFTFitter(),
        'Log-Normal': LogNormalAFTFitter(),
        'Log-Logistic': LogLogisticAFTFitter()
    }
    
    fitted_models = {}
    model_summaries = {}
    
    print("\n" + "=" * 80)
    print("FITTING AFT MODELS")
    print("=" * 80)
    
    for name, model in models.items():
        print(f"\n{name} AFT Model:")
        print("-" * 40)
        
        try:
            model.fit(model_data, duration_col=duration_col, event_col=event_col)
            fitted_models[name] = model
            
            model_summaries[name] = {
                'AIC': model.AIC_,
                'BIC': model.BIC_,
                'concordance': model.concordance_index_,
                'log_likelihood': model.log_likelihood_
            }
            
            print(f"AIC: {model.AIC_:.2f}")
            print(f"BIC: {model.BIC_:.2f}")
            print(f"Concordance Index: {model.concordance_index_:.4f}")
            print(f"Log-Likelihood: {model.log_likelihood_:.2f}")
            
        except Exception as e:
            print(f"Error fitting {name}: {str(e)}")
    
    return fitted_models, model_summaries, feature_cols


# compare models

def compare_models(model_summaries):
    """Create comparison table"""
    comparison_df = pd.DataFrame(model_summaries).T
    comparison_df = comparison_df.sort_values('AIC')
    
    print("\n" + "=" * 80)
    print("MODEL COMPARISON")
    print("=" * 80)
    print(comparison_df.to_string())
    print(f"\nBest model (by AIC): {comparison_df.index[0]}")
    print(f"Best model (by concordance): {comparison_df['concordance'].idxmax()}")
    
    return comparison_df

# visualizations

def plot_survival_curves(fitted_models, df, feature_cols, save_path='./img/survival_curves.png'):
    """Plot all survival curves on one plot"""
    fig, ax = plt.subplots(figsize=(12, 7))
    
    colors = {'Weibull': '#2E86AB', 'Log-Normal': '#A23B72', 'Log-Logistic': '#F18F01'}
    
    median_individual = df[['tenure', 'churn_binary'] + feature_cols].median().to_frame().T
    
    for name, model in fitted_models.items():
        times = np.linspace(0, df['tenure'].max(), 100)
        survival_func = model.predict_survival_function(median_individual, times=times)
        
        ax.plot(times, survival_func.values.flatten(), 
                label=name, linewidth=2.5, color=colors.get(name, 'gray'))
    
    ax.set_xlabel('Time (Months)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Survival Probability', fontsize=12, fontweight='bold')
    ax.set_title('AFT Model Comparison: Survival Curves for Median Customer', 
                 fontsize=14, fontweight='bold')
    ax.legend(loc='best', fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 1)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"\nSurvival curves saved to {save_path}")
    plt.close()  




def select_features(model, df, feature_cols, significance_level=0.05):
    """Keep only significant features"""
    summary = model.summary
    
    print("\n" + "=" * 80)
    print(f"FEATURE SIGNIFICANCE ANALYSIS (p < {significance_level})")
    print("=" * 80)
    
    lambda_params = [col for col in summary.index if 'lambda_' in col and col != 'lambda_']
    significant_params = summary.loc[lambda_params][summary.loc[lambda_params]['p'] < significance_level]
    
    if len(significant_params) > 0:
        print("\nSignificant features:")
        print(significant_params[['coef', 'exp(coef)', 'p']].to_string())
    else:
        print("\nNo features meet the significance threshold.")
        print("\nTop 5 most significant features:")
        print(summary.loc[lambda_params].nsmallest(5, 'p')[['coef', 'exp(coef)', 'p']].to_string())
    
    return significant_params.index.tolist() if len(significant_params) > 0 else []


# calculate CLV

def calculate_clv(df, model, feature_cols, monthly_revenue=50):
    """Calculate Customer Lifetime Value"""
    
    model_data = df[['tenure', 'churn_binary'] + feature_cols].copy()
    
    expected_lifetimes = model.predict_expectation(model_data)
    
    df['expected_lifetime'] = expected_lifetimes
    df['CLV'] = monthly_revenue * expected_lifetimes
    
    print("\n" + "=" * 80)
    print("CUSTOMER LIFETIME VALUE STATISTICS")
    print("=" * 80)
    print(f"\nAssuming monthly revenue: ${monthly_revenue}")
    print(f"\n{df['CLV'].describe()}")
    print(f"\nTotal portfolio CLV: ${df['CLV'].sum():,.2f}")
    print(f"Average CLV per customer: ${df['CLV'].mean():,.2f}")
    
    return df


# semgment analysis

def segment_analysis(df):
    """Analyze CLV across different segments"""
    print("\n" + "=" * 80)
    print("CLV BY CUSTOMER SEGMENTS")
    print("=" * 80)
    
    segments = {
        'Customer Category': 'custcat',
        'Internet Service': 'internet',
        'Voice Service': 'voice',
        'Region': 'region',
        'Marital Status': 'marital'
    }
    
    segment_results = {}
    
    for seg_name, seg_col in segments.items():
        print(f"\n{seg_name}:")
        print("-" * 60)
        seg_clv = df.groupby(seg_col).agg({
            'CLV': ['mean', 'median', 'sum', 'count']
        }).round(2)
        seg_clv.columns = ['Mean CLV', 'Median CLV', 'Total CLV', 'Count']
        seg_clv = seg_clv.sort_values('Mean CLV', ascending=False)
        print(seg_clv.to_string())
        segment_results[seg_name] = seg_clv
    
    return segment_results

# retention budget calculation

def calculate_retention_budget(df, model, feature_cols, time_horizon=12):
    """Calculate annual retention budget"""
    
    model_data = df[['tenure', 'churn_binary'] + feature_cols].copy()
    
    survival_probs = model.predict_survival_function(model_data, times=[time_horizon])
    df['survival_prob_12m'] = survival_probs.T.values
    df['churn_prob_12m'] = 1 - df['survival_prob_12m']
    
    at_risk_threshold = 0.3
    df['at_risk'] = df['churn_prob_12m'] > at_risk_threshold
    
    at_risk_customers = df[df['at_risk']]
    total_at_risk = len(at_risk_customers)
    total_clv_at_risk = at_risk_customers['CLV'].sum()
    avg_clv_at_risk = at_risk_customers['CLV'].mean()
    
    retention_budget = total_clv_at_risk * 0.15
    
    print("\n" + "=" * 80)
    print("RETENTION BUDGET ANALYSIS")
    print("=" * 80)
    print(f"\nTotal customers: {len(df):,}")
    print(f"At-risk customers (churn prob > {at_risk_threshold:.0%} in {time_horizon} months): {total_at_risk:,}")
    print(f"At-risk rate: {total_at_risk/len(df)*100:.2f}%")
    print(f"\nAverage CLV of at-risk customers: ${avg_clv_at_risk:,.2f}")
    print(f"Total CLV at risk: ${total_clv_at_risk:,.2f}")
    print(f"\n{'='*60}")
    print(f"RECOMMENDED ANNUAL RETENTION BUDGET: ${retention_budget:,.2f}")
    print(f"{'='*60}")
    print(f"\nThis represents 15% of at-risk CLV")
    print(f"Budget per at-risk customer: ${retention_budget/total_at_risk:,.2f}")
    
    return {
        'total_customers': len(df),
        'at_risk_customers': total_at_risk,
        'at_risk_rate': total_at_risk/len(df),
        'total_clv_at_risk': total_clv_at_risk,
        'suggested_budget': retention_budget,
        'budget_per_customer': retention_budget/total_at_risk
    }


# again visualizations

def plot_clv_distribution(df, segment_results, save_path='./img/clv_analysis.png'):
    """Visualize CLV distribution and segments"""
    fig = plt.figure(figsize=(16, 10))
    gs = fig.add_gridspec(3, 2, hspace=0.3, wspace=0.3)
    
    # 1. CLV Distribution
    ax1 = fig.add_subplot(gs[0, :])
    ax1.hist(df['CLV'], bins=50, color='#2E86AB', alpha=0.7, edgecolor='black')
    ax1.axvline(df['CLV'].mean(), color='red', linestyle='--', linewidth=2, label=f"Mean: ${df['CLV'].mean():.2f}")
    ax1.axvline(df['CLV'].median(), color='green', linestyle='--', linewidth=2, label=f"Median: ${df['CLV'].median():.2f}")
    ax1.set_xlabel('Customer Lifetime Value ($)', fontweight='bold', fontsize=11)
    ax1.set_ylabel('Frequency', fontweight='bold', fontsize=11)
    ax1.set_title('CLV Distribution Across All Customers', fontweight='bold', fontsize=13)
    ax1.legend(fontsize=10)
    ax1.grid(alpha=0.3)
    
    # 2. CLV by Customer Category
    ax2 = fig.add_subplot(gs[1, 0])
    custcat_data = df.groupby('custcat')['CLV'].mean().sort_values(ascending=True)
    custcat_data.plot(kind='barh', ax=ax2, color='#A23B72')
    ax2.set_xlabel('Average CLV ($)', fontweight='bold')
    ax2.set_title('CLV by Customer Category', fontweight='bold', fontsize=12)
    ax2.grid(alpha=0.3, axis='x')
    
    # 3. CLV by Internet Service
    ax3 = fig.add_subplot(gs[1, 1])
    internet_data = df.groupby('internet')['CLV'].mean().sort_values(ascending=True)
    internet_data.plot(kind='barh', ax=ax3, color='#F18F01')
    ax3.set_xlabel('Average CLV ($)', fontweight='bold')
    ax3.set_title('CLV by Internet Service', fontweight='bold', fontsize=12)
    ax3.grid(alpha=0.3, axis='x')
    
    # 4. Churn Probability Distribution
    ax4 = fig.add_subplot(gs[2, 0])
    ax4.hist(df['churn_prob_12m'], bins=30, color='#E63946', alpha=0.7, edgecolor='black')
    ax4.axvline(0.3, color='darkred', linestyle='--', linewidth=2, label='At-risk threshold (30%)')
    ax4.set_xlabel('12-Month Churn Probability', fontweight='bold')
    ax4.set_ylabel('Frequency', fontweight='bold')
    ax4.set_title('Distribution of Churn Risk', fontweight='bold', fontsize=12)
    ax4.legend()
    ax4.grid(alpha=0.3)
    
    # 5. CLV vs Tenure (scatter)
    ax5 = fig.add_subplot(gs[2, 1])
    scatter = ax5.scatter(df['tenure'], df['CLV'], c=df['churn_prob_12m'], 
                         cmap='RdYlGn_r', alpha=0.6, s=20)
    ax5.set_xlabel('Tenure (Months)', fontweight='bold')
    ax5.set_ylabel('CLV ($)', fontweight='bold')
    ax5.set_title('CLV vs Tenure (colored by churn risk)', fontweight='bold', fontsize=12)
    plt.colorbar(scatter, ax=ax5, label='Churn Probability')
    ax5.grid(alpha=0.3)
    
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"\nCLV analysis plots saved to {save_path}")
    plt.close()  # Close instead of show to avoid blocking


def main(filepath='telco.csv'):
    """Main execution function"""
    
    print("\n" + "=" * 80)
    print("SURVIVAL ANALYSIS - CUSTOMER CHURN PREDICTION")
    print("=" * 80)
    
    df = load_and_prepare_data(filepath)
    
    fitted_models, model_summaries, feature_cols = fit_aft_models(df)
    
    comparison = compare_models(model_summaries)
    
    plot_survival_curves(fitted_models, df, feature_cols)
    
    best_model_name = comparison.index[0]
    best_model = fitted_models[best_model_name]
    print(f"\nSelected model for analysis: {best_model_name}")
    
    significant_features = select_features(best_model, df, feature_cols)
    
    df = calculate_clv(df, best_model, feature_cols, monthly_revenue=50)
    
    segment_results = segment_analysis(df)
    
    budget_info = calculate_retention_budget(df, best_model, feature_cols)
    
    plot_clv_distribution(df, segment_results)
    
  
    
    return df, fitted_models, best_model, budget_info


if __name__ == "__main__":
    df_results, models, final_model, budget = main('./data/telco.csv')