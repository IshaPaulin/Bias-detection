import pandas as pd
import numpy as np

def calculate_group_metrics(df: pd.DataFrame, outcome_col: str, proxy_var: str):
    """
    Calculate the approval rates for different groups of a proxy variable.
    Returns the privileged group, unprivileged group, and their respective approval rates.
    """
    if df[proxy_var].dtype in ['int64', 'float64']:
        # For numerical, split by median
        median_val = df[proxy_var].median()
        low_group = df[df[proxy_var] <= median_val]
        high_group = df[df[proxy_var] > median_val]
        
        low_approval = low_group[outcome_col].mean() if len(low_group) > 0 else 0
        high_approval = high_group[outcome_col].mean() if len(high_group) > 0 else 0
        
        if high_approval >= low_approval:
            return f"High (> {median_val:.1f})", f"Low (<= {median_val:.1f})", high_approval, low_approval
        else:
            return f"Low (<= {median_val:.1f})", f"High (> {median_val:.1f})", low_approval, high_approval
    else:
        # For categorical, find highest and lowest approval categories
        rates = df.groupby(proxy_var)[outcome_col].mean()
        if len(rates) == 0:
            return "Unknown", "Unknown", 0.0, 0.0
            
        privileged_group = rates.idxmax()
        privileged_rate = rates.max()
        
        unprivileged_group = rates.idxmin()
        unprivileged_rate = rates.min()
        
        return privileged_group, unprivileged_group, privileged_rate, unprivileged_rate

def get_fairness_metrics(df: pd.DataFrame, outcome_col: str, proxy_var: str):
    """
    Computes Disparate Impact Ratio (DIR) and Demographic Parity Difference.
    
    DIR = P(outcome=1 | unprivileged) / P(outcome=1 | privileged)
    DPD = P(outcome=1 | unprivileged) - P(outcome=1 | privileged)
    """
    priv_group, unpriv_group, priv_rate, unpriv_rate = calculate_group_metrics(df, outcome_col, proxy_var)
    
    # Avoid division by zero
    dir_score = unpriv_rate / priv_rate if priv_rate > 0 else 0.0
    dpd_score = unpriv_rate - priv_rate
    
    return {
        "proxy_variable": proxy_var,
        "privileged_group": priv_group,
        "unprivileged_group": unpriv_group,
        "privileged_approval_rate": round(priv_rate, 4),
        "unprivileged_approval_rate": round(unpriv_rate, 4),
        "disparate_impact_ratio": round(dir_score, 4),
        "demographic_parity_difference": round(dpd_score, 4)
    }
