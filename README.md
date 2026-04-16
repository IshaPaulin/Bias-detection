# 🎯 AI Fairness Auditor

**Detect algorithmic bias before it causes harm**

This tool analyzes AI decision-making systems to find hidden discrimination patterns using counterfactual testing.

---

## 🏗️ The Big Picture: How Everything Fits Together

```
┌─────────────────────────────────────────────────────────────┐
│                    THE USER EXPERIENCE                       │
│  Upload CSV → Click Button → Get Fairness Report            │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                   STREAMLIT DASHBOARD                        │
│              (app.py - The Orchestrator)                     │
│                                                              │
│  What it does:                                               │
│  • Accepts user uploads                                      │
│  • Routes data through analysis pipeline                     │
│  • Visualizes results                                        │
│  • Generates downloadable reports                            │
└─────────────────────────────────────────────────────────────┘
         │                           │
         │                           │
         ▼                           ▼
┌──────────────────┐      ┌──────────────────────┐
│  PROXY DETECTOR  │      │ COUNTERFACTUAL       │
│                  │      │ TESTER               │
│ (proxy_detector  │      │                      │
│     .py)         │      │ (counterfactual_     │
│                  │      │      tester.py)      │
│                  │      │                      │
│ Finds variables  │      │ Proves bias by       │
│ that shouldn't   │──────│ creating "what-if"   │
│ matter but do    │      │ scenarios            │
│                  │      │                      │
│ Examples:        │      │ Tests: "If John      │
│ - Zip codes      │      │ had Jane's zip,      │
│ - Browser type   │      │ would he get         │
│ - App time       │      │ approved?"           │
└──────────────────┘      └──────────────────────┘
         │                           │
         │                           │
         ▼                           ▼
┌─────────────────────────────────────────────────────────────┐
│                    SAMPLE DATA GENERATOR                     │
│              (generate_sample_data.py)                       │
│                                                              │
│  Creates realistic biased datasets for testing               │
│  Shows the "redlining" problem in action                     │
└─────────────────────────────────────────────────────────────┘
```

---

## 🧠 The Core Innovation: Counterfactual Testing

**The Problem We're Solving:**

Traditional fairness metrics ask: *"Are outcomes equal across groups?"*

That's not enough. An AI could reject everyone equally and pass that test.

**Our Approach:**

We ask: *"If we changed ONLY the suspicious variable, would the decision flip?"*

**Example:**
```
Original Applicant:
  Name: John
  Zip Code: 10025 (Area A)
  Credit Score: 750
  Income: $75,000
  Decision: REJECTED

Counterfactual Twin:
  Name: John
  Zip Code: 10075 (Area B) ← ONLY THIS CHANGED
  Credit Score: 750
  Income: $75,000
  Decision: APPROVED

Result: 🚨 ZIP CODE IS CAUSING BIAS
```

---

## 📂 File Structure & What Each Does

```
fairness-auditor/
│
├── app.py                        # Main dashboard (START HERE)
│   └── What: Streamlit web interface
│   └── Why: Makes the tool accessible to non-technical users
│   └── How it fits: Orchestrates all other components
│
├── proxy_detector.py             # Statistical analyzer
│   └── What: Finds variables correlated with outcomes
│   └── Why: Can't test counterfactuals without knowing what to test
│   └── How it fits: Feeds suspicious variables to counterfactual tester
│
├── counterfactual_tester.py      # The core algorithm
│   └── What: Creates "twin" applicants and compares decisions
│   └── Why: Proves CAUSAL bias, not just correlation
│   └── How it fits: Generates concrete evidence for auditors
│
├── generate_sample_data.py       # Demo data creator
│   └── What: Simulates a biased lending dataset
│   └── Why: Shows the tool working without sensitive real data
│   └── How it fits: Lets you test the system immediately
│
└── requirements.txt              # Dependencies
```

---

## 🚀 Quick Start

### 1️⃣ Installation

```bash
# Install dependencies
pip install -r requirements.txt
```

### 2️⃣ Generate Sample Data (Optional)

```bash
python generate_sample_data.py
```

This creates `sample_loan_data.csv` with realistic bias patterns.

### 3️⃣ Run the Dashboard

```bash
streamlit run app.py
```

The app will open in your browser at `http://localhost:8501`

### 4️⃣ Try It Out

1. Check "Use sample biased loan data"
2. Click "Run Fairness Audit"
3. Explore the results:
   - Fairness Score
   - Proxy Variables
   - Biased Decisions

---

## 🎯 How to Use This With Your Own Data

### Data Requirements

Your CSV should have:
- **One row per decision** (loan application, job application, etc.)
- **Multiple feature columns** (age, income, zip code, etc.)
- **One binary outcome column** (1 = approved, 0 = rejected)

### Example Structure

```csv
applicant_id,age,income,zip_code,credit_score,approved
APP_001,34,65000,10025,720,0
APP_002,28,82000,10075,750,1
APP_003,45,55000,10032,680,0
```

### Steps

1. Upload your CSV in the dashboard
2. Select the outcome column (e.g., "approved")
3. Optionally select a known sensitive attribute (e.g., "zip_code")
4. Click "Run Fairness Audit"

---

## 🔬 The Three-Layer Analysis

### Layer 1: Proxy Detection
**What it does:** Finds variables that correlate suspiciously with outcomes

**Example output:**
```
🚨 application_hour: suspiciousness score = 0.85
   Low hours (9-17): 31% approval
   High hours (18-23): 84% approval
```

**Why it matters:** These are the variables we'll test counterfactually

---

### Layer 2: Counterfactual Testing
**What it does:** Creates "twin" applicants with only proxy variables changed

**Example output:**
```
🚨 Found 63 biased decisions

Sample case:
  Applicant: APP_00388
  Credit Score: 839
  Income: $70,734
  Original Zip: 10028 → REJECTED
  Changed Zip: 10084 → APPROVED
```

**Why it matters:** Proves the AI is discriminating based on zip code

---

### Layer 3: Fairness Scoring
**What it does:** Calculates an overall fairness metric

**Formula:**
```python
fairness_score = 100 - (biased_rejections / total_rejections * 100)
```

**Interpretation:**
- **90-100:** Excellent fairness
- **70-89:** Moderate bias detected
- **Below 70:** Significant bias, take action

---

## 🛠️ Technical Architecture Deep Dive

### The Proxy Detector

**Statistical Method:** Mutual Information + Pearson Correlation

```python
# For numerical features
correlation = abs(pearsonr(feature, outcome)[0])

# For categorical features
mutual_info_score(feature, outcome)
```

**Why both?**
- Pearson catches linear relationships (income ↔ approval)
- Mutual info catches nonlinear patterns (zip code ↔ approval)

---

### The Counterfactual Generator

**Algorithm:**

```python
def generate_counterfactual(original_applicant, variable_to_change):
    twin = original_applicant.copy()
    
    if variable == 'zip_code':
        # Flip to opposite demographic region
        if original_zip in area_A:
            twin['zip_code'] = random_zip_from(area_B)
        else:
            twin['zip_code'] = random_zip_from(area_A)
    
    return twin
```

**Key insight:** We're testing INTERVENTIONS, not just observations

---

### The ML Model

**Why we train our own:**
- In real deployments, you'd audit an existing model
- For demos, we train a RandomForest on the biased data
- This simulates real-world "AI learns from biased history" problem

---

## 📊 Real-World Use Cases

### 1. Lending (Current Demo)
**Problem:** Historical redlining encoded in zip codes  
**Solution:** Flag loans rejected due to geography

### 2. Hiring
**Problem:** "Culture fit" encodes gender/race bias  
**Solution:** Test if changing name/school changes hiring outcome

### 3. College Admissions
**Problem:** "Legacy" status correlates with race/class  
**Solution:** Detect if legacy status overrides merit

### 4. Healthcare
**Problem:** Insurance denials vary by neighborhood  
**Solution:** Flag health-based denials driven by demographics

---

## 🚨 Important Limitations

### What This Tool CAN Do:
✅ Detect statistical bias patterns  
✅ Generate evidence for auditors  
✅ Identify problematic proxy variables  
✅ Prove causation via counterfactuals  

### What This Tool CANNOT Do:
❌ Replace legal advice  
❌ Automatically fix biased models  
❌ Determine if bias is illegal (context-dependent)  
❌ Solve societal discrimination  

---

## 🔮 Next Steps / Extensions

### For a Production System:

1. **Add more fairness metrics**
   - Demographic parity
   - Equal opportunity
   - Calibration across groups

2. **Integrate with MLOps**
   - Monitor models in production
   - Alert when fairness degrades
   - Version fairness scores alongside accuracy

3. **Build explainability layer**
   - Use Gemini API to generate plain-English explanations
   - Create regulatory compliance reports
   - Generate bias mitigation recommendations

4. **Add mitigation tools**
   - Reweighting training data
   - Adversarial debiasing
   - Fairness constraints in optimization

---

## 🤝 Contributing

This is an educational/demo project showing how to build fairness auditing tools.

**Ideas for improvement:**
- Add more sophisticated counterfactual generation
- Implement additional fairness metrics
- Create industry-specific templates (hiring, lending, etc.)
- Build compliance report generators

---

## 📚 Further Reading

**Academic Papers:**
- [Counterfactual Fairness](https://arxiv.org/abs/1703.06856) - Kusner et al.
- [Fairness Through Awareness](https://arxiv.org/abs/1104.3913) - Dwork et al.

**Real-World Cases:**
- ProPublica's COMPAS investigation
- Amazon's hiring algorithm case
- Apple Card credit limit controversy

**Regulations:**
- EU AI Act
- NYC AI Hiring Law (Local Law 144)
- ECOA (Equal Credit Opportunity Act)

---

## ⚖️ License & Disclaimer

This is an educational tool for demonstrating bias detection techniques.

**Not legal advice.** Consult compliance experts before using in production.

**Not a guarantee.** Passing this audit doesn't mean your AI is legally compliant.

---

## 🎓 Learning Resources

### If you want to understand the math:

**ProxyDetector.py:**
- Study `scipy.stats.pearsonr` documentation
- Learn about mutual information theory
- Understand correlation vs causation

**CounterfactualTester.py:**
- Read about causal inference
- Study Pearl's "Book of Why"
- Learn scikit-learn's RandomForest internals

**App.py:**
- Streamlit tutorial: https://docs.streamlit.io
- Plotly visualization: https://plotly.com/python

---

## 💬 Questions?

**"Why not just remove sensitive attributes from training data?"**  
→ The AI will use proxies (zip code, names, etc.)

**"Why counterfactuals instead of just checking approval rates?"**  
→ Counterfactuals prove CAUSATION, not just correlation

**"Can this tool fix biased models?"**  
→ No, it only DETECTS bias. Fixing requires retraining or architectural changes

**"Is 100% fairness possible?"**  
→ No. Tradeoffs exist between different fairness definitions

---

Built to expose algorithmic bias • Consult legal experts • Use responsibly
