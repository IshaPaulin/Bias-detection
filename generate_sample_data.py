"""
SAMPLE DATA GENERATOR
=====================
Creates a realistic "biased lending" dataset to test our auditor.

WHY THIS EXISTS:
- Real biased datasets are hard to get (companies hide them)
- We need controlled bias to validate our detection works
- Shows the "proxy variable" problem in action

THE BIAS WE'RE SIMULATING:
- Zip codes in certain areas get denied more often (redlining)
- Same credit score + income = different outcomes based on location
"""

import pandas as pd
import numpy as np

np.random.seed(42)

def generate_biased_loan_data(n_samples=1000):
    """
    Generate loan application data with realistic bias patterns
    
    The Hidden Bias:
    - Applications from zip codes 10001-10050 (let's call this "Area A") 
      get approved 30% of the time
    - Applications from zip codes 10051-10100 (let's call this "Area B")
      get approved 70% of the time
    - Even when credit scores and income are IDENTICAL
    
    This simulates real-world redlining where geographic proxies
    encode historical discrimination.
    """
    
    data = []
    
    for _ in range(n_samples):
        # Generate base applicant features
        age = np.random.randint(22, 70)
        income = np.random.randint(25000, 150000)
        credit_score = np.random.randint(550, 850)
        employment_years = min(age - 22, np.random.randint(0, 25))
        
        # Geographic assignment (this is our "protected attribute" proxy)
        # In real life, this correlates with historical discrimination
        zip_code = np.random.randint(10001, 10101)
        
        # Determine if they're in the "disadvantaged" area
        in_area_a = zip_code <= 10050
        
        # Calculate "legitimate" approval probability based on financials
        base_approval_prob = (
            0.2 * (credit_score / 850) +
            0.3 * (min(income, 100000) / 100000) +
            0.2 * (min(employment_years, 10) / 10) +
            0.3 * (age / 70)
        )
        
        # HERE'S THE BIAS: Reduce probability dramatically for Area A
        if in_area_a:
            biased_approval_prob = base_approval_prob * 0.4  # 60% penalty
        else:
            biased_approval_prob = base_approval_prob * 1.2  # 20% boost
        
        # Make the decision
        approved = np.random.random() < biased_approval_prob
        
        # Add some correlated features (these will show up as proxies)
        # People in Area A tend to have different browser usage patterns
        browser = "Safari" if in_area_a and np.random.random() < 0.7 else "Chrome"
        
        # People in Area A apply at different times
        application_hour = np.random.randint(9, 17) if in_area_a else np.random.randint(18, 23)
        
        data.append({
            'applicant_id': f'APP_{_:05d}',
            'age': age,
            'income': income,
            'credit_score': credit_score,
            'employment_years': employment_years,
            'zip_code': zip_code,
            'browser': browser,
            'application_hour': application_hour,
            'loan_amount_requested': np.random.randint(5000, 50000),
            'approved': int(approved)
        })
    
    return pd.DataFrame(data)

if __name__ == "__main__":
    # Generate the dataset
    df = generate_biased_loan_data(1000)
    
    # Save it
    df.to_csv('data/sample_loan_data.csv', index=False)
    
    print("✅ Generated biased loan dataset")
    print(f"Total applications: {len(df)}")
    print(f"Approval rate: {df['approved'].mean():.1%}")
    print("\nApproval by zip code region:")
    df['region'] = df['zip_code'].apply(lambda x: 'Area A (10001-10050)' if x <= 10050 else 'Area B (10051-10100)')
    print(df.groupby('region')['approved'].agg(['mean', 'count']))
    print("\n⚠️  Notice: Same applicants, different outcomes based on geography")
