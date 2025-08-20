# 🌍 Global Economic Insights

An interactive project combining **Power BI dashboards** and **Machine Learning models** to analyze, visualize, and forecast global economic indicators such as GDP, Inflation, Public Debt, and Fiscal Balance.  

---

## 📊 Dashboard Preview

### India's Macroeconomic Trends (Power BI)

<img width="1151" height="644" alt="image" src="https://github.com/user-attachments/assets/6eda32f8-1111-47e9-87cc-2e789247fcb9" />


> Example: Interactive Power BI dashboard showing GDP, Inflation, Fiscal Indicators, and Debt trends for India (2010–2025).  

---

## 🧑‍💻 Model Overview

The machine learning pipeline focuses on predicting **GDP (Current USD)** using multiple regression models (Linear Regression, Random Forest, Gradient Boosting).  
It also includes **feature engineering** with lag variables, rolling averages, and fiscal data.  

### Key Features
- Data cleaning & preprocessing for missing values  
- Feature engineering (lags, rolling averages, ratios)  
- Model training with `scikit-learn` pipelines  
- Evaluation using **RMSE, R², and MAE**  
- Visualization of feature importance and model performance  

---

## 📈 Results

- **Best Model:** Gradient Boosting (highest R² score)  
- Achieved accurate **GDP predictions** for the latest available year  
- Feature importance shows strong contributions from:  
  - GDP Growth (% Annual)  
  - Inflation (CPI %)  
  - Public Debt (% of GDP)  
  - Lagged GDP values  
<img width="713" height="547" alt="image" src="https://github.com/user-attachments/assets/b3787c52-0862-4871-88c3-068302f87ce9" />

---

## 🚀 How to Run

1. Clone this repo:  
   ```bash
   git clone https://github.com/mondal-paushali03/global-economic-insights.git
   cd global-economic-insights
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
3. Run the model:
   ```bash
   python scripts/economic_model.py
4. Open the Power BI file (.pbix) to explore the dashboard.

## ✉️ Contact
For questions or analysis, contact Paushali Mondal at mondal.paushali384@gmail.com

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details
