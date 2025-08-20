# 1. Import Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_selection import mutual_info_regression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# 2. Load Dataset
df = pd.read_csv(r'C:\Users\PAUSHALI MONDAL\Desktop\economic\world_bank_data_2025.csv')

print("Initial Shape:", df.shape)
print("\nMissing Values:\n", df.isnull().sum())

# 3. Preprocessing Missing Values

df = df.sort_values(['country_name', 'year'])

time_sensitive_cols = [
    'GDP (Current USD)', 'GDP per Capita (Current USD)', 'GDP Growth (% Annual)',
    'Inflation (CPI %)', 'Unemployment Rate (%)'
]
for col in time_sensitive_cols:
    df[col] = df.groupby('country_name')[col].ffill().bfill()
    df[col] = df.groupby('country_name')[col].transform(
        lambda x: x.interpolate(method='linear', limit_direction='both')
    )

df = df.drop(columns=['Interest Rate (Real, %)', 'Public Debt (% of GDP)'], errors='ignore')

fiscal_cols = ['Government Expense (% of GDP)', 'Government Revenue (% of GDP)', 'Tax Revenue (% of GDP)']
for col in fiscal_cols:
    df[col] = df.groupby('country_name')[col].transform(
        lambda x: x.fillna(x.median()) if x.notna().any() else x
    )
    df[col] = df[col].fillna(df[col].median())

df['Current Account Balance (% GDP)'] = df.groupby('country_name')['Current Account Balance (% GDP)'].ffill().bfill()
df['Current Account Balance (% GDP)'] = df['Current Account Balance (% GDP)'].fillna(df['Current Account Balance (% GDP)'].median())

if 'Gross National Income (USD)' in df.columns:
    gni_ratio = (df['Gross National Income (USD)'] / df['GDP (Current USD)']).median()
    df['Gross National Income (USD)'] = df['Gross National Income (USD)'].fillna(
        df['GDP (Current USD)'] * gni_ratio
    )

print("\nAfter Cleaning Missing Values:\n", df.isnull().sum())

# 4. Exploratory Visualizations
heatmap_data = df.pivot_table(index='country_name', columns='year', values='Inflation (CPI %)')
plt.figure(figsize=(15, 20))
sns.heatmap(heatmap_data, cmap='coolwarm', center=0, cbar_kws={'label': 'Inflation Rate (%)'})
plt.title('Global Inflation Trends (2010-2025)')
plt.xlabel('Year')
plt.ylabel('Country')
plt.tight_layout()
plt.show()

gdp_growth = df.groupby('country_name')['GDP Growth (% Annual)'].mean().sort_values()
plt.figure(figsize=(12, 8))
sns.barplot(x=gdp_growth.tail(10).values, y=gdp_growth.tail(10).index, color='green', label='Top 10')
sns.barplot(x=gdp_growth.head(10).values, y=gdp_growth.head(10).index, color='red', label='Bottom 10')
plt.axvline(0, color='black', linestyle='--')
plt.title('Top/Bottom 10 Countries by Avg GDP Growth (2010-2025)')
plt.xlabel('Average GDP Growth (%)')
plt.ylabel('Country')
plt.legend()
plt.show()

plt.figure(figsize=(10, 5))
sns.lineplot(data=df, x='year', y='Current Account Balance (% GDP)', estimator='median', errorbar=None)
plt.axhline(0, color='black')
plt.title('Global Trade Balance Trend')
plt.show()

plt.figure(figsize=(12, 8))
sns.heatmap(df.corr(numeric_only=True), annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Correlation Heatmap of Numerical Columns')
plt.show()

# 5. Feature Engineering for ML
TARGET = 'GDP (Current USD)'
df.dropna(subset=[TARGET], inplace=True)
df = df.sort_values(['country_name', 'year']).copy()

numerical_features_for_mi = df.drop(columns=[TARGET, 'country_name', 'year'], errors='ignore').select_dtypes(include=np.number)
df_for_mi = pd.concat([numerical_features_for_mi, df[TARGET]], axis=1).dropna()
X_mi = df_for_mi.drop(columns=[TARGET])
y_mi = df_for_mi[TARGET]
mi_scores = mutual_info_regression(X_mi, y_mi)
mi_scores = pd.Series(mi_scores, name="MI Score", index=X_mi.columns).sort_values(ascending=False)
selected_numeric_features_mi = mi_scores[mi_scores > 0.05].index.tolist()

final_features = selected_numeric_features_mi.copy()
if 'country_name' not in final_features: final_features.append('country_name')
if 'year' not in final_features: final_features.append('year')

# Lag & rolling features
df[f'{TARGET}_lag1'] = df.groupby('country_name')[TARGET].shift(1)
final_features.append(f'{TARGET}_lag1')

lag_cols = ['GDP per Capita (Current USD)', 'Population (Total)',
            'GDP Growth (% Annual)', 'Inflation (CPI %)', 'Unemployment Rate (%)']
for col in lag_cols:
    if col in df.columns:
        df[f'{col}_lag1'] = df.groupby('country_name')[col].shift(1)
        final_features.append(f'{col}_lag1')

df['GDP Growth (% Annual)_roll3'] = df.groupby('country_name')['GDP Growth (% Annual)'].transform(
    lambda x: x.rolling(window=3, min_periods=1).mean().shift(1)
)
final_features.append('GDP Growth (% Annual)_roll3')

df.dropna(subset=[col for col in final_features if col not in ['country_name','year']], inplace=True)

# 6. Train-Test Split
X = df[final_features].copy()
y = df[TARGET].copy()
last_year = df['year'].max()
X_train = X[X['year'] < last_year].drop(columns=['year'])
y_train = y[X['year'] < last_year]
X_test = X[X['year'] == last_year].drop(columns=['year'])
y_test = y[X['year'] == last_year]

categorical_features = ['country_name']
numerical_features = [col for col in X_train.columns if col not in categorical_features]
preprocessor = ColumnTransformer([
    ('num', StandardScaler(), numerical_features),
    ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
])

# 7. Model Training
models = {
    "Linear Regression": LinearRegression(),
    "Random Forest": RandomForestRegressor(n_estimators=200, random_state=42),
    "Gradient Boosting": GradientBoostingRegressor(n_estimators=300, learning_rate=0.05, random_state=42)
}
results = {}

for name, model in models.items():
    pipeline = Pipeline([('preprocessor', preprocessor), ('regressor', model)])
    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    results[name] = {"RMSE": rmse, "R2": r2, "MAE": mae, "predictions": y_pred}
    print(f"\n--- {name} Performance ---")
    print(f"RMSE: {rmse:,.2f}, R²: {r2:.4f}, MAE: {mae:,.2f}")

# 8. Compare Model Performance
results_df = pd.DataFrame(results).T[['RMSE','R2','MAE']]
print("\nModel Comparison:\n", results_df)

plt.figure(figsize=(8,5))
sns.barplot(x=results_df.index, y=results_df['R2'])
plt.title("Model R² Comparison")
plt.ylabel("R² Score")
plt.show()

# 9. Actual vs Predicted GDP (Best Model)
best_model_name = max(results, key=lambda m: results[m]['R2'])
print(f"\nBest Model Selected: {best_model_name}")
y_pred_best = results[best_model_name]["predictions"]

plt.figure(figsize=(8, 6))
sns.scatterplot(x=y_test/1e9, y=y_pred_best/1e9)
plt.plot([y_test.min()/1e9, y_test.max()/1e9], [y_test.min()/1e9, y_test.max()/1e9], color="red", linestyle="--")
plt.title(f"Actual vs Predicted GDP ({best_model_name}) - Test Year {last_year}")
plt.xlabel("Actual GDP (Billion USD)")
plt.ylabel("Predicted GDP (Billion USD)")
plt.show()

# 10. Feature Importance (Tree Models Only)
if best_model_name in ["Random Forest", "Gradient Boosting"]:
    best_pipeline = Pipeline([('preprocessor', preprocessor), ('regressor', models[best_model_name])])
    best_pipeline.fit(X_train, y_train)
    ohe_features = best_pipeline.named_steps['preprocessor'].transformers_[1][1].get_feature_names_out(['country_name'])
    feature_names = numerical_features + list(ohe_features)
    importances = best_pipeline.named_steps['regressor'].feature_importances_
    feat_imp = pd.Series(importances, index=feature_names).sort_values(ascending=False)[:20]
    plt.figure(figsize=(10,6))
    sns.barplot(x=feat_imp.values, y=feat_imp.index)
    plt.title(f"Top 20 Feature Importances - {best_model_name}")
    plt.show()
