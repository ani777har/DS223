# Survival Analysis: Customer Churn Prediction

## Overview
This project applies Accelerated Failure Time (AFT) models to predict customer churn and estimate Customer Lifetime Value (CLV) using a telecom dataset of 1,000 subscribers. The analysis includes model comparison, survival curve evaluation, CLV calculation, churn-risk segmentation, and retention budget recommendations.

## Dataset
The dataset contains:
- **Survival variables**: tenure (months), churn event  
- **Demographics**: age, marital status, education, income, gender  
- **Service usage**: voice, internet, call forwarding  
- **Other attributes**: region, address stability, customer category (Basic, Plus, E-service, Total service)

## Project Structure 
```
├── survival_analysis.py     
├── requirements.txt          
├── README.md                 
├── report.md                 
├── data/
│   └── telecom_churn.csv    
└── outputs/
    ├── survival_curves.png   
    └── clv_distribution.png  
```

## Installation & Running the Analysis
```bash
source venv/bin/activate
pip install -r requirements.txt
python survival_analysis.py
```