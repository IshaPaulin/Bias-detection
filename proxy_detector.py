"""
PROXY DETECTOR
==============
Finds variables that are secretly acting as stand-ins for protected attributes.

THE PROBLEM IT SOLVES:
If you remove "Race" from a hiring dataset, the AI will just use:
- "Distance from downtown" (correlates with racial segregation)
- "High school name" (correlates with neighborhood demographics)
- "First name" (correlates with ethnicity)

This module uses statistical correlation to find those hidden proxies.

HOW IT FITS IN THE BIGGER PICTURE:
Before we can test counterfactuals, we need to know WHICH variables to test.
This is the reconnaissance phase.
"""

import pandas as pd
import numpy as np
from sklearn.metrics import mutual_info_score
from scipy.stats import chi2_contingency, pearsonr
from typing import Dict, List, Tuple

class ProxyDetector:
    """
    Identifies variables that correlate suspiciously with outcomes
    when grouped by a sensitive attribute.
    """
    
    def __init__(self, suspicious_threshold=0.3):
        """
        Args:
            suspicious_threshold: Correlation strength above which we flag a variable
                                 (0.3 = moderate correlation, good starting point)
        """
        self.suspicious_threshold = suspicious_threshold
        self.proxy_scores = {}
    
    def detect_proxies(self, 
                       df: pd.DataFrame, 
                       outcome_col: str,
                       sensitive_attr: str = None) -> Dict[str, float]:
        """
        Find variables that might be acting as proxies for discrimination.
        
        THE LOGIC:
        If changing zip_code changes approval rates more than changing credit_score,
        that's suspicious. Zip shouldn't matter more than creditworthiness.
        
        Args:
            df: Your dataset
            outcome_col: The decision being made (e.g., 'approved', 'hired')
            sensitive_attr: If you know the protected attribute (e.g., 'zip_code'),
                           we'll measure correlations directly. Otherwise, we look
                           for ANY variable that has outsized impact.
        
        Returns:
            Dictionary of {variable_name: suspiciousness_score}
        """
        
        results = {}
        
        for col in df.columns:
            # Skip the outcome and the sensitive attribute itself
            if col in [outcome_col, sensitive_attr]:
                continue
            
            # Skip ID columns
            if 'id' in col.lower():
                continue
            
            try:
                # Measure how strongly this variable predicts the outcome
                if df[col].dtype in ['int64', 'float64']:
                    # Numerical variable: use correlation
                    score = abs(pearsonr(df[col], df[outcome_col])[0])
                else:
                    # Categorical variable: use mutual information
                    score = mutual_info_score(df[col], df[outcome_col])
                
                # If we know the sensitive attribute, also check correlation with it
                if sensitive_attr and col != sensitive_attr:
                    if df[col].dtype in ['int64', 'float64'] and df[sensitive_attr].dtype in ['int64', 'float64']:
                        sensitive_correlation = abs(pearsonr(df[col], df[sensitive_attr])[0])
                        # Boost the score if it correlates with the sensitive attribute
                        score = max(score, sensitive_correlation * 1.5)
                
                results[col] = score
                
            except Exception as e:
                # Some columns might not be analyzable
                continue
        
        # Sort by suspiciousness
        self.proxy_scores = dict(sorted(results.items(), key=lambda x: x[1], reverse=True))
        
        return {k: v for k, v in self.proxy_scores.items() if v > self.suspicious_threshold}
    
    def get_proxy_explanation(self, proxy_var: str, df: pd.DataFrame, outcome_col: str) -> str:
        """
        Generate a human-readable explanation of why this variable is suspicious.
        
        HOW THIS FITS:
        The dashboard needs to show managers WHY something is a problem,
        not just that it is a problem.
        """
        
        # Calculate approval rates by this variable
        if df[proxy_var].dtype in ['int64', 'float64']:
            # For numerical: split into high/low
            median_val = df[proxy_var].median()
            low_group = df[df[proxy_var] <= median_val]
            high_group = df[df[proxy_var] > median_val]
            
            low_approval = low_group[outcome_col].mean()
            high_approval = high_group[outcome_col].mean()
            
            return f"""
⚠️  {proxy_var} shows suspicious patterns:
- Low {proxy_var} (≤{median_val:.0f}): {low_approval:.1%} approval rate
- High {proxy_var} (>{median_val:.0f}): {high_approval:.1%} approval rate
- Difference: {abs(high_approval - low_approval):.1%}

This suggests {proxy_var} is being used as a decision factor,
potentially as a proxy for a protected characteristic.
"""
        else:
            # For categorical: show approval rates by category
            rates = df.groupby(proxy_var)[outcome_col].mean()
            top_categories = rates.nlargest(3)
            bottom_categories = rates.nsmallest(3)
            
            explanation = f"\n⚠️  {proxy_var} shows suspicious patterns:\n\n"
            explanation += "Highest approval rates:\n"
            for cat, rate in top_categories.items():
                explanation += f"  - {cat}: {rate:.1%}\n"
            explanation += "\nLowest approval rates:\n"
            for cat, rate in bottom_categories.items():
                explanation += f"  - {cat}: {rate:.1%}\n"
            
            return explanation

# Example usage
if __name__ == "__main__":
    # Load our biased data
    df = pd.read_csv('/home/claude/sample_loan_data.csv')
    
    detector = ProxyDetector(suspicious_threshold=0.25)
    
    # Find proxies for the 'approved' decision
    proxies = detector.detect_proxies(df, outcome_col='approved', sensitive_attr='zip_code')
    
    print("🔍 PROXY VARIABLE DETECTION RESULTS\n")
    print("=" * 60)
    print(f"\nFound {len(proxies)} suspicious variables:\n")
    
    for var, score in list(proxies.items())[:5]:  # Show top 5
        print(f"📊 {var}: suspiciousness score = {score:.3f}")
        print(detector.get_proxy_explanation(var, df, 'approved'))
        print("-" * 60)
