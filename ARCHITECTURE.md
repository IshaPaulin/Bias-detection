# 🏗️ ARCHITECTURE DEEP DIVE

## The Product Journey: From Upload to Report

```
┌─────────────────────────────────────────────────────────────────────┐
│ STEP 1: USER UPLOADS DATA                                           │
│ "Here's my historical loan decisions from 2020-2024"                │
└─────────────────────────────────────────────────────────────────────┘
                              ⬇️
┌─────────────────────────────────────────────────────────────────────┐
│ STEP 2: DATA PROFILING (ProxyDetector.detect_proxies)              │
│                                                                      │
│ For each variable in the dataset:                                   │
│   1. Calculate correlation with outcome                             │
│   2. Calculate correlation with known sensitive attributes          │
│   3. Rank by "suspiciousness score"                                 │
│                                                                      │
│ Output: ['zip_code': 0.85, 'application_hour': 0.72, ...]          │
└─────────────────────────────────────────────────────────────────────┘
                              ⬇️
┌─────────────────────────────────────────────────────────────────────┐
│ STEP 3: MODEL TRAINING (CounterfactualTester.train_biased_model)   │
│                                                                      │
│ Why we do this:                                                      │
│ - In real deployments, you'd audit an EXISTING model                │
│ - For demos, we train one to show how bias propagates               │
│                                                                      │
│ What happens:                                                        │
│   RandomForest trains on historical data                            │
│   → Learns "zip_code 10025 → high rejection rate"                  │
│   → Even though creditworthiness is the same!                       │
│                                                                      │
│ This simulates the REAL PROBLEM: AI automates human bias            │
└─────────────────────────────────────────────────────────────────────┘
                              ⬇️
┌─────────────────────────────────────────────────────────────────────┐
│ STEP 4: COUNTERFACTUAL TESTING                                      │
│ (CounterfactualTester.test_counterfactual_fairness)                │
│                                                                      │
│ FOR EACH rejected application:                                      │
│   FOR EACH suspicious variable:                                     │
│                                                                      │
│     ┌─────────────────────────────────────────┐                    │
│     │ Original Applicant                      │                    │
│     │ ─────────────────────                   │                    │
│     │ Name: John                              │                    │
│     │ Zip: 10025                              │                    │
│     │ Credit: 750                             │                    │
│     │ Income: $75k                            │                    │
│     │                                         │                    │
│     │ Model says: REJECT                      │                    │
│     └─────────────────────────────────────────┘                    │
│                     ⬇️ CHANGE ONLY ZIP                               │
│     ┌─────────────────────────────────────────┐                    │
│     │ Counterfactual Twin                     │                    │
│     │ ─────────────────────                   │                    │
│     │ Name: John                              │                    │
│     │ Zip: 10075 ← CHANGED                    │                    │
│     │ Credit: 750 ← SAME                      │                    │
│     │ Income: $75k ← SAME                     │                    │
│     │                                         │                    │
│     │ Model says: APPROVE                     │                    │
│     └─────────────────────────────────────────┘                    │
│                     ⬇️                                               │
│             🚨 BIAS DETECTED!                                        │
│     Flag this case in the report                                    │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
                              ⬇️
┌─────────────────────────────────────────────────────────────────────┐
│ STEP 5: SCORING & VISUALIZATION (app.py)                           │
│                                                                      │
│ Calculate:                                                           │
│   - Fairness score (% of decisions that passed counterfactual test) │
│   - Proxy variable rankings                                         │
│   - Specific examples of biased decisions                           │
│                                                                      │
│ Generate:                                                            │
│   - Interactive charts (Plotly)                                     │
│   - Downloadable CSV report                                         │
│   - Plain-English explanations                                      │
└─────────────────────────────────────────────────────────────────────┘
                              ⬇️
┌─────────────────────────────────────────────────────────────────────┐
│ STEP 6: USER GETS REPORT                                           │
│                                                                      │
│ "Your AI rejected 63 people based on zip code alone.                │
│  Here's the list. Here's how to fix it."                            │
└─────────────────────────────────────────────────────────────────────┘
```

---

## The Three Core Algorithms (Explained Like You're 5)

### 1. Proxy Detection Algorithm

**What it does:**  
Finds variables that predict outcomes too well (suspiciously)

**Analogy:**  
Imagine you're a teacher grading exams. You notice that students who use blue pens get better grades than students who use black pens. That's suspicious! Pen color shouldn't matter. → Blue pen is a PROXY for something else (maybe blue-pen students studied together).

**The Math:**
```python
# For numerical variables like income
correlation = pearsonr(variable, outcome)
# Returns -1 to 1, where 1 = perfect correlation

# For categorical variables like browser type
mutual_info = mutual_info_score(variable, outcome)
# Returns 0 to ∞, where higher = more dependency

if correlation > threshold:
    flag_as_suspicious(variable)
```

**Example Output:**
```
Suspicious Variables:
1. zip_code (score: 0.85) ← Very suspicious
2. application_hour (score: 0.72) ← Moderately suspicious
3. browser (score: 0.31) ← Slightly suspicious
4. credit_score (score: 0.68) ← Expected! This SHOULD matter
```

---

### 2. Counterfactual Generation Algorithm

**What it does:**  
Creates a "parallel universe" version of an applicant with ONE thing changed

**Analogy:**  
In movie "Sliding Doors," Gwyneth Paltrow's character catches/misses a train, and we see both timelines. We're doing the same thing: showing what would've happened if ONE variable was different.

**The Algorithm:**
```python
def generate_counterfactual(applicant, variable):
    twin = applicant.copy()  # Clone the applicant
    
    if variable == 'zip_code':
        # Flip to opposite demographic area
        if applicant.zip in area_A:
            twin.zip = random_zip_from(area_B)
    
    elif variable == 'application_hour':
        # Flip time window
        if applicant.hour < 18:
            twin.hour = random(18, 23)
        else:
            twin.hour = random(9, 17)
    
    return twin
```

**Why this works:**  
By changing ONLY the suspicious variable, we isolate its effect. If the decision flips, we KNOW that variable caused it (ceteris paribus - all else equal).

---

### 3. Fairness Scoring Algorithm

**What it does:**  
Converts complex bias patterns into a single number (0-100)

**The Formula:**
```python
total_rejected = count(applicants where outcome = 0)
biased_rejections = count(applicants where counterfactual flipped)

fairness_score = 100 - (biased_rejections / total_rejected * 100)
```

**Example:**
```
Total rejections: 420
Biased rejections: 63

fairness_score = 100 - (63/420 * 100)
               = 100 - 15
               = 85/100

Interpretation: 85% of rejections were "fair" (wouldn't flip if 
we changed proxy variables). 15% were biased.
```

---

## How the Files Work Together (Code Flow)

### Scenario: User Clicks "Run Audit"

**app.py (Dashboard):**
```python
# 1. User clicks button
if st.button("Run Audit"):
    
    # 2. First, find suspicious variables
    detector = ProxyDetector()
    proxies = detector.detect_proxies(df, 'approved')
    # Returns: {'zip_code': 0.85, 'app_hour': 0.72}
    
    # 3. Train model (or load existing)
    tester = CounterfactualTester()
    tester.train_biased_model(df, 'approved')
    
    # 4. Test counterfactuals
    flagged = tester.test_counterfactual_fairness(
        df,
        variables_to_test=['zip_code', 'app_hour']
    )
    # Returns: DataFrame of biased cases
    
    # 5. Display results
    show_results(proxies, flagged)
```

**proxy_detector.py:**
```python
class ProxyDetector:
    def detect_proxies(self, df, outcome_col):
        results = {}
        
        for column in df.columns:
            # Calculate suspiciousness
            score = calculate_correlation(df[column], df[outcome_col])
            
            if score > threshold:
                results[column] = score
        
        return sorted_by_score(results)
```

**counterfactual_tester.py:**
```python
class CounterfactualTester:
    def test_counterfactual_fairness(self, df, variables_to_test):
        flagged_cases = []
        
        for applicant in rejected_applicants:
            for variable in variables_to_test:
                
                # Create twin
                twin = generate_counterfactual(applicant, variable)
                
                # Compare decisions
                original_decision = model.predict(applicant)
                twin_decision = model.predict(twin)
                
                if original_decision != twin_decision:
                    flagged_cases.append({
                        'applicant': applicant.id,
                        'variable': variable,
                        'flipped': True
                    })
        
        return flagged_cases
```

---

## Data Flow Diagram

```
CSV Upload
    │
    ▼
┌─────────────────────┐
│ Pandas DataFrame    │
│ (1000 rows × 9 cols)│
└─────────────────────┘
    │
    ├──────────────────────────┐
    │                          │
    ▼                          ▼
┌─────────────┐    ┌──────────────────┐
│Proxy        │    │ Model Training   │
│Detector     │    │                  │
│             │    │ RandomForest     │
│Scans all    │    │ learns from      │
│columns      │    │ biased data      │
└─────────────┘    └──────────────────┘
    │                          │
    │ Returns                  │ Returns
    │ suspicious vars          │ trained model
    │                          │
    ▼                          ▼
┌──────────────────────────────────────┐
│    Counterfactual Tester             │
│                                      │
│  For each rejected applicant:        │
│    For each suspicious var:          │
│      Create twin                     │
│      Compare predictions             │
│      Flag if different               │
└──────────────────────────────────────┘
    │
    │ Returns flagged cases
    │
    ▼
┌──────────────────────┐
│ Streamlit Dashboard  │
│                      │
│ - Charts             │
│ - Tables             │
│ - Download button    │
└──────────────────────┘
```

---

## Why This Architecture Matters

### Separation of Concerns

Each file has ONE job:

1. **generate_sample_data.py** → Creates test data  
   (In production, this would be replaced by user uploads)

2. **proxy_detector.py** → Statistical analysis  
   (Pure math, no ML, no UI)

3. **counterfactual_tester.py** → ML + causal inference  
   (Can be tested independently)

4. **app.py** → Presentation layer  
   (Could swap Streamlit for Flask/FastAPI/etc.)

**Benefit:** You can improve one piece without breaking others

---

### Testability

Each component can be unit tested:

```python
# Test proxy detector
def test_proxy_detection():
    df = create_biased_dataset()
    detector = ProxyDetector()
    proxies = detector.detect_proxies(df, 'approved')
    assert 'zip_code' in proxies  # Should find the bias we planted

# Test counterfactual generator
def test_counterfactual_generation():
    applicant = {'zip_code': 10025, 'income': 50000}
    twin = generate_counterfactual(applicant, 'zip_code')
    assert twin['zip_code'] != 10025  # Should change
    assert twin['income'] == 50000    # Should NOT change
```

---

### Extensibility

**Adding a new fairness metric?**  
→ Add method to ProxyDetector, don't touch CounterfactualTester

**Want to support more ML models?**  
→ Modify CounterfactualTester, don't touch app.py

**Want a better UI?**  
→ Replace app.py entirely, keep the analysis engines

---

## The "Why" Behind Design Decisions

### Q: Why use RandomForest instead of a simple logistic regression?

**A:** RandomForest is more realistic:
- Real-world ML systems use complex models
- Shows that even "accurate" models can be biased
- Demonstrates that high accuracy ≠ fairness

### Q: Why generate sample data instead of using real datasets?

**A:** 
- Real biased data is hard to find (companies hide it)
- We can control exactly what bias to inject
- No privacy concerns for demos

### Q: Why Streamlit instead of React + FastAPI?

**A:**
- Faster to build (100 lines vs 1000)
- Good enough for MVP/internal tools
- Can always migrate to production stack later

### Q: Why test counterfactuals instead of just checking approval rates?

**A:**
- Approval rates only show CORRELATION
- Counterfactuals prove CAUSATION
- Courts/regulators understand "what-if" scenarios better

---

## Scaling to Production

### Current Architecture (MVP):
```
User uploads CSV → Streamlit processes locally → Returns report
```

**Good for:** Internal audits, demos, research

**Limitations:**
- Runs on one machine
- Handles datasets up to ~100K rows
- No persistent storage
- No user accounts

---

### Production Architecture (If You Scale This):

```
┌─────────────┐
│ React App   │
│ (Frontend)  │
└─────────────┘
       ↓ API calls
┌─────────────────────┐
│ FastAPI Backend     │
│                     │
│ - Auth (JWT)        │
│ - Rate limiting     │
│ - Job queue         │
└─────────────────────┘
       ↓
┌─────────────────────┐
│ Worker Pool         │
│ (Celery + Redis)    │
│                     │
│ - ProxyDetector     │
│ - Counterfactual    │
│   Tester            │
└─────────────────────┘
       ↓
┌─────────────────────┐
│ PostgreSQL          │
│                     │
│ - Audit history     │
│ - User datasets     │
│ - Flagged cases     │
└─────────────────────┘
```

**What changes:**
- API layer for multiple clients
- Async job processing for large datasets
- Database for persistence
- Caching for repeated audits
- Monitoring/logging

**What stays the same:**
- Core algorithms (ProxyDetector, CounterfactualTester)
- The underlying math
- The fairness definitions

---

This architecture lets you start simple and scale complexity only when needed.
