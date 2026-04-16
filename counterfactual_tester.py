"""
COUNTERFACTUAL FAIRNESS TESTER
===============================
The core innovation of your fairness auditor.

THE INSIGHT:
If an AI denies John (zip 10025) but would approve Jane (zip 10075),
AND they have identical credit scores, income, employment history...
Then the AI is discriminating based on zip code.

HOW IT WORKS:
1. Take a rejected application
2. Create a "twin" application with ONLY the suspicious variable changed
3. Run both through the model
4. If the decision flips → FLAG IT as biased

WHY THIS IS POWERFUL:
- Courts understand "what-if" scenarios
- Shows CAUSAL bias, not just correlation
- Generates concrete examples for auditors
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from typing import List, Dict, Tuple

class CounterfactualTester:
    """
    Tests if changing protected/proxy attributes flips decisions.
    """
    
    def __init__(self, model=None):
        """
        Args:
            model: The decision-making model we're auditing.
                   If None, we'll train a simple one for demo purposes.
        """
        self.model = model
        self.flagged_cases = []
        
    def train_biased_model(self, df: pd.DataFrame, outcome_col: str):
        """
        Train a model on the biased data (simulating the real-world AI)
        
        IN THE REAL PRODUCT:
        Users would upload their existing model, but for the demo,
        we train one to show how it inherits bias from data.
        """
        
        # Prepare features
        feature_cols = [col for col in df.columns 
                       if col not in [outcome_col, 'applicant_id', 'region']]
        
        # Handle categorical variables
        X = pd.get_dummies(df[feature_cols], drop_first=True)
        y = df[outcome_col]
        
        # Train/test split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        # Train a simple random forest
        self.model = RandomForestClassifier(n_estimators=100, random_state=42)
        self.model.fit(X_train, y_train)
        
        # Store feature names for later
        self.feature_names = X.columns.tolist()
        
        accuracy = self.model.score(X_test, y_test)
        print(f"✅ Trained model with {accuracy:.1%} accuracy")
        print(f"   (High accuracy doesn't mean fair - that's the problem!)")
        
        return X_test, y_test
    
    def generate_counterfactual(self, 
                                original_row: pd.Series, 
                                variable_to_change: str,
                                df: pd.DataFrame) -> pd.Series:
        """
        Create a "twin" applicant with only ONE variable changed.
        
        THE LOGIC:
        - If testing zip_code: pick a zip from the opposite region
        - If testing application_hour: pick the opposite time window
        - Keep everything else identical
        
        Args:
            original_row: The applicant we're testing
            variable_to_change: Which variable to flip
            df: Full dataset (to sample realistic alternative values)
        
        Returns:
            Modified version of the applicant
        """
        
        counterfactual = original_row.copy()
        
        if variable_to_change == 'zip_code':
            # Flip to opposite region
            if original_row['zip_code'] <= 10050:
                # Move to Area B
                counterfactual['zip_code'] = np.random.randint(10051, 10101)
            else:
                # Move to Area A
                counterfactual['zip_code'] = np.random.randint(10001, 10051)
        
        elif variable_to_change == 'application_hour':
            # Flip time window
            if original_row['application_hour'] <= 18:
                counterfactual['application_hour'] = np.random.randint(18, 23)
            else:
                counterfactual['application_hour'] = np.random.randint(9, 17)
        
        elif variable_to_change == 'browser':
            # Flip browser
            counterfactual['browser'] = 'Chrome' if original_row['browser'] == 'Safari' else 'Safari'
        
        else:
            # For numerical variables, swap with median from opposite outcome group
            approved_group = df[df['approved'] == 1]
            rejected_group = df[df['approved'] == 0]
            
            if original_row['approved'] == 0:
                counterfactual[variable_to_change] = approved_group[variable_to_change].median()
            else:
                counterfactual[variable_to_change] = rejected_group[variable_to_change].median()
        
        return counterfactual
    
    def test_counterfactual_fairness(self, 
                                     df: pd.DataFrame, 
                                     outcome_col: str,
                                     variables_to_test: List[str],
                                     sample_size: int = 100) -> pd.DataFrame:
        """
        Run the full counterfactual audit.
        
        THE WORKFLOW:
        1. For each rejected applicant
        2. For each suspicious variable
        3. Create a counterfactual twin
        4. Check if the decision flips
        5. If it flips → document this as evidence of bias
        
        Returns:
            DataFrame of all cases where decisions flipped
        """
        
        # Get rejected applications to test
        rejected = df[df[outcome_col] == 0].sample(n=min(sample_size, len(df[df[outcome_col] == 0])))
        
        flagged_cases = []
        
        print(f"\n🔬 Testing {len(rejected)} rejected applications...\n")
        
        for idx, original_row in rejected.iterrows():
            
            # Prepare original for prediction
            feature_cols = [col for col in df.columns 
                           if col not in [outcome_col, 'applicant_id', 'region']]
            original_df = pd.DataFrame([original_row[feature_cols]])
            original_encoded = pd.get_dummies(original_df, drop_first=True)
            
            # Ensure all expected columns exist
            for col in self.feature_names:
                if col not in original_encoded.columns:
                    original_encoded[col] = 0
            original_encoded = original_encoded[self.feature_names]
            
            original_prediction = self.model.predict(original_encoded)[0]
            
            # Test each suspicious variable
            for var in variables_to_test:
                if var not in original_row.index:
                    continue
                
                # Create counterfactual
                counterfactual_row = self.generate_counterfactual(original_row, var, df)
                counterfactual_df = pd.DataFrame([counterfactual_row[feature_cols]])
                counterfactual_encoded = pd.get_dummies(counterfactual_df, drop_first=True)
                
                # Ensure all expected columns exist
                for col in self.feature_names:
                    if col not in counterfactual_encoded.columns:
                        counterfactual_encoded[col] = 0
                counterfactual_encoded = counterfactual_encoded[self.feature_names]
                
                counterfactual_prediction = self.model.predict(counterfactual_encoded)[0]
                
                # Did the decision flip?
                if original_prediction != counterfactual_prediction:
                    flagged_cases.append({
                        'applicant_id': original_row['applicant_id'],
                        'variable_changed': var,
                        'original_value': original_row[var],
                        'counterfactual_value': counterfactual_row[var],
                        'original_decision': 'Rejected',
                        'counterfactual_decision': 'Approved',
                        'credit_score': original_row['credit_score'],
                        'income': original_row['income'],
                        'employment_years': original_row['employment_years']
                    })
        
        self.flagged_cases = pd.DataFrame(flagged_cases)
        
        if len(self.flagged_cases) > 0:
            print(f"🚨 Found {len(self.flagged_cases)} cases where decision flipped!\n")
        else:
            print("✅ No clear counterfactual bias detected\n")
        
        return self.flagged_cases

# Example usage
if __name__ == "__main__":
    # Load data
    df = pd.read_csv('/home/claude/sample_loan_data.csv')
    
    # Initialize tester
    tester = CounterfactualTester()
    
    # Train model on biased data
    print("📊 Training AI on biased historical data...")
    tester.train_biased_model(df, outcome_col='approved')
    
    # Run counterfactual tests
    print("\n" + "="*60)
    flagged = tester.test_counterfactual_fairness(
        df, 
        outcome_col='approved',
        variables_to_test=['zip_code', 'application_hour', 'browser'],
        sample_size=50
    )
    
    if len(flagged) > 0:
        print("📋 SAMPLE BIASED DECISIONS:\n")
        print(flagged.head(10).to_string(index=False))
        print(f"\n💡 These applicants were rejected, but changing ONLY their")
        print(f"   {flagged['variable_changed'].value_counts().index[0]} would have gotten them approved.")
        print(f"   Their credit score and income stayed the same!")
