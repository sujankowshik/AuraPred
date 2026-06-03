import json

def create_notebook():
    notebook = {
        "cells": [
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "# Real Estate Price Prediction - Machine Learning Pipeline\n",
                    "\n",
                    "This notebook demonstrates the end-to-end Machine Learning pipeline used to predict housing prices. We will perform:\n",
                    "1. **Exploratory Data Analysis (EDA)**\n",
                    "2. **Data Cleaning & Preprocessing** (Imputing missing values, removing duplicates, and detecting outliers using IQR)\n",
                    "3. **Feature Encoding & Scaling** (One-hot encoding and standard scaling)\n",
                    "4. **Model Training & Comparison** (Linear Regression, Random Forest, Gradient Boosting, and XGBoost Regressor)\n",
                    "5. **Evaluation Metrics** ($R^2$ Score, MAE, MSE, RMSE)\n",
                    "6. **Model Saving & Visualization** (Feature Importance, Actual vs Predicted, Model Comparison)"
                ]
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "## Step 1: Import Libraries and Load Dataset"
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "import os\n",
                    "import pickle\n",
                    "import numpy as np\n",
                    "import pandas as pd\n",
                    "import matplotlib.pyplot as plt\n",
                    "import seaborn as sns\n",
                    "\n",
                    "from sklearn.model_selection import train_test_split\n",
                    "from sklearn.preprocessing import StandardScaler, OneHotEncoder\n",
                    "from sklearn.linear_model import LinearRegression\n",
                    "from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor\n",
                    "from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error\n",
                    "\n",
                    "# Try to import xgboost, fallback if not available\n",
                    "XGB_AVAILABLE = False\n",
                    "try:\n",
                    "    from xgboost import XGBRegressor\n",
                    "    XGB_AVAILABLE = True\n",
                    "except (ImportError, Exception) as e:\n",
                    "    print(f'WARNING: XGBoost could not be loaded: {str(e)}')\n",
                    "\n",
                    "# Styling setups\n",
                    "sns.set_theme(style=\"darkgrid\", palette=\"muted\")\n",
                    "plt.rcParams['figure.figsize'] = (10, 6)"
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "# Load the dataset\n",
                    "df = pd.read_csv('../dataset/housing.csv')\n",
                    "print(f\"Dataset shape: {df.shape}\")\n",
                    "df.head()"
                ]
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "## Step 2: Exploratory Data Analysis & Preprocessing\n",
                    "\n",
                    "### 2.1 Handling Duplicates\n",
                    "Removing exact duplicate entries."
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "duplicates_count = df.duplicated().sum()\n",
                    "print(f\"Number of duplicate rows: {duplicates_count}\")\n",
                    "if duplicates_count > 0:\n",
                    "    df.drop_duplicates(inplace=True)\n",
                    "    print(f\"Removed duplicates. New shape: {df.shape}\")"
                ]
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "### 2.2 Imputing Missing Values\n",
                    "Impute missing values using numerical medians and categorical modes."
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "print(\"Missing values count per column:\")\n",
                    "print(df.isnull().sum())\n",
                    "\n",
                    "# Compute values\n",
                    "area_median = df['Area_SqFt'].median()\n",
                    "year_median = df['Year_Built'].median()\n",
                    "amenities_mode = df['Amenities'].mode()[0]\n",
                    "\n",
                    "# Impute\n",
                    "df['Area_SqFt'] = df['Area_SqFt'].fillna(area_median)\n",
                    "df['Year_Built'] = df['Year_Built'].fillna(year_median)\n",
                    "df['Amenities'] = df['Amenities'].fillna(amenities_mode)\n",
                    "\n",
                    "print(\"\\nMissing values count after imputation:\")\n",
                    "print(df.isnull().sum())"
                ]
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "### 2.3 Outlier Detection & Capping (IQR Method)\n",
                    "Identify and remove severe outliers in `Area_SqFt` and `Price` using the $1.5 \\times IQR$ rule."
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "cleaned_df = df.copy()\n",
                    "\n",
                    "for col in ['Area_SqFt', 'Price']:\n",
                    "    Q1 = cleaned_df[col].quantile(0.25)\n",
                    "    Q3 = cleaned_df[col].quantile(0.75)\n",
                    "    IQR = Q3 - Q1\n",
                    "    lower_bound = Q1 - 1.5 * IQR\n",
                    "    upper_bound = Q3 + 1.5 * IQR\n",
                    "    \n",
                    "    outliers = cleaned_df[(cleaned_df[col] < lower_bound) | (cleaned_df[col] > upper_bound)]\n",
                    "    print(f\"Column '{col}': found {len(outliers)} outliers outside bounds [{lower_bound:.2f}, {upper_bound:.2f}]\")\n",
                    "    \n",
                    "    # Filter outliers\n",
                    "    cleaned_df = cleaned_df[(cleaned_df[col] >= lower_bound) & (cleaned_df[col] <= upper_bound)]\n",
                    "    \n",
                    "print(f\"Shape after removing outliers: {cleaned_df.shape} (Removed {len(df) - len(cleaned_df)} rows)\")\n",
                    "df = cleaned_df"
                ]
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "## Step 3: Feature Encoding & Scaling\n",
                    "\n",
                    "Separate numerical and categorical variables, scale numerical inputs, and one-hot encode categorical classes."
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "categorical_cols = ['Location', 'Parking', 'Amenities']\n",
                    "numerical_cols = ['Area_SqFt', 'Bedrooms', 'Bathrooms', 'Year_Built', 'Floors']\n",
                    "\n",
                    "# One-Hot Encoding\n",
                    "encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')\n",
                    "encoded_features = encoder.fit_transform(df[categorical_cols])\n",
                    "encoded_feature_names = encoder.get_feature_names_out(categorical_cols)\n",
                    "encoded_df = pd.DataFrame(encoded_features, columns=encoded_feature_names, index=df.index)\n",
                    "\n",
                    "# Scaling\n",
                    "scaler = StandardScaler()\n",
                    "scaled_features = scaler.fit_transform(df[numerical_cols])\n",
                    "scaled_df = pd.DataFrame(scaled_features, columns=numerical_cols, index=df.index)\n",
                    "\n",
                    "# Combine\n",
                    "X = pd.concat([scaled_df, encoded_df], axis=1)\n",
                    "y = df['Price']\n",
                    "\n",
                    "feature_columns = list(X.columns)\n",
                    "print(f\"Features shape: {X.shape}\")\n",
                    "X.head()"
                ]
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "## Step 4: Model Training, Evaluation & Comparison\n",
                    "\n",
                    "Train four regression models and evaluate using standard ML metrics."
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "# Train-Test Split\n",
                    "X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)\n",
                    "print(f\"Train size: {X_train.shape[0]}, Test size: {X_test.shape[0]}\")"
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "models = {\n",
                    "    'Linear Regression': LinearRegression(),\n",
                    "    'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42),\n",
                    "    'Gradient Boosting': GradientBoostingRegressor(n_estimators=150, learning_rate=0.1, random_state=42)\n",
                    "}\n",
                    "if XGB_AVAILABLE:\n",
                    "    models['XGBoost'] = XGBRegressor(n_estimators=150, learning_rate=0.1, random_state=42)\n",
                    "\n",
                    "results = {}\n",
                    "best_r2 = -1\n",
                    "best_model_name = None\n",
                    "best_model = None\n",
                    "\n",
                    "for name, model in models.items():\n",
                    "    model.fit(X_train, y_train)\n",
                    "    preds = model.predict(X_test)\n",
                    "    \n",
                    "    r2 = r2_score(y_test, preds)\n",
                    "    mae = mean_absolute_error(y_test, preds)\n",
                    "    mse = mean_squared_error(y_test, preds)\n",
                    "    rmse = np.sqrt(mse)\n",
                    "    \n",
                    "    results[name] = {\n",
                    "        'R2': r2,\n",
                    "        'MAE': mae,\n",
                    "        'RMSE': rmse\n",
                    "    }\n",
                    "    print(f\"{name:18} | R2: {r2:.4f} | MAE: ₹{mae:,.2f} | RMSE: ₹{rmse:,.2f}\")\n",
                    "    \n",
                    "    if r2 > best_r2:\n",
                    "        best_r2 = r2\n",
                    "        best_model_name = name\n",
                    "        best_model = model"
                ]
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "## Step 5: Visualizations & Analytics\n",
                    "\n",
                    "Plot model benchmarks, actual-vs-predicted scatters, and feature importances."
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "# 5.1 Model Comparison R2 Scores\n",
                    "names = list(results.keys())\n",
                    "r2_scores = [results[n]['R2'] * 100 for n in names]\n",
                    "\n",
                    "plt.figure(figsize=(10, 6))\n",
                    "bars = plt.bar(names, r2_scores, color=['#3498db', '#9b59b6', '#f1c40f', '#2ecc71'], width=0.5)\n",
                    "plt.title('Algorithm R² Accuracy Comparison (%)', fontsize=14, pad=15, fontweight='bold')\n",
                    "plt.ylabel('R² Score (%)', fontsize=11)\n",
                    "plt.ylim(80, 100)\n",
                    "for bar in bars:\n",
                    "    h = bar.get_height()\n",
                    "    plt.text(bar.get_x() + bar.get_width()/2.0, h + 0.5, f'{h:.2f}%', ha='center', va='bottom', fontweight='bold')\n",
                    "plt.show()"
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "# 5.2 Actual vs Predicted Price Scatter Plot\n",
                    "best_preds = best_model.predict(X_test)\n",
                    "plt.figure(figsize=(10, 7))\n",
                    "sns.scatterplot(x=y_test, y=best_preds, alpha=0.5, color='#8e44ad')\n",
                    "ideal_min = min(y_test.min(), best_preds.min())\n",
                    "ideal_max = max(y_test.max(), best_preds.max())\n",
                    "plt.plot([ideal_min, ideal_max], [ideal_min, ideal_max], color='#e74c3c', linestyle='--', linewidth=2, label='Optimal Prediction')\n",
                    "plt.title(f'Actual vs Predicted House Prices ({best_model_name})', fontsize=14, pad=15, fontweight='bold')\n",
                    "plt.xlabel('Actual Price (₹)', fontsize=11)\n",
                    "plt.ylabel('Predicted Price (₹)', fontsize=11)\n",
                    "plt.gca().xaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f\"₹{x:,.0f}\"))\n",
                    "plt.gca().yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f\"₹{x:,.0f}\"))\n",
                    "plt.legend()\n",
                    "plt.show()"
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "# 5.3 Feature Importance Chart\n",
                    "if hasattr(best_model, 'feature_importances_'):\n",
                    "    importances = best_model.feature_importances_\n",
                    "    indices = np.argsort(importances)[::-1]\n",
                    "    \n",
                    "    top_n = min(12, len(feature_columns))\n",
                    "    top_importances = importances[indices[:top_n]]\n",
                    "    top_features = [feature_columns[i] for i in indices[:top_n]]\n",
                    "    \n",
                    "    plt.figure(figsize=(10, 7))\n",
                    "    sns.barplot(x=top_importances, y=top_features, hue=top_features, palette='viridis', legend=False)\n",
                    "    plt.title(f'Top {top_n} Feature Importance ({best_model_name})', fontsize=14, pad=15, fontweight='bold')\n",
                    "    plt.xlabel('Relative Importance Score', fontsize=11)\n",
                    "    plt.ylabel('Features', fontsize=11)\n",
                    "    plt.show()\n",
                    "else:\n",
                    "    print(\"Feature importance is not available for this model.\")"
                ]
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "## Step 6: Export Models & Preprocessing Artifacts\n",
                    "\n",
                    "Save the fitted model, scaler, and key metadata so the Flask application can predict house prices instantly from incoming client forms."
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "os.makedirs('../model', exist_ok=True)\n",
                    "\n",
                    "with open('../model/model.pkl', 'wb') as f:\n",
                    "    pickle.dump(best_model, f)\n",
                    "\n",
                    "with open('../model/scaler.pkl', 'wb') as f:\n",
                    "    pickle.dump(scaler, f)\n",
                    "\n",
                    "preprocessing_meta = {\n",
                    "    'area_median': area_median,\n",
                    "    'year_median': year_median,\n",
                    "    'amenities_mode': amenities_mode,\n",
                    "    'encoder': encoder,\n",
                    "    'categorical_cols': categorical_cols,\n",
                    "    'numerical_cols': numerical_cols,\n",
                    "    'feature_columns': feature_columns,\n",
                    "    'best_model_name': best_model_name,\n",
                    "    'best_r2': best_r2,\n",
                    "    'metrics': results\n",
                    "}\n",
                    "\n",
                    "with open('../model/preprocessing_meta.pkl', 'wb') as f:\n",
                    "    pickle.dump(preprocessing_meta, f)\n",
                    "\n",
                    "print(\"All pipeline artifacts successfully exported!\")"
                ]
            }
        ],
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3"
            },
            "language_info": {
                "name": "python"
            }
        },
        "nbformat": 4,
        "nbformat_minor": 2
    }

    with open('notebooks/model_training.ipynb', 'w') as f:
        json.dump(notebook, f, indent=1)
    print("Jupyter Notebook generated successfully at 'notebooks/model_training.ipynb'!")

if __name__ == '__main__':
    create_notebook()
