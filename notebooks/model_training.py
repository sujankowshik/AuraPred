import os
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
# Try to import xgboost regressor, fallback if not available
XGB_AVAILABLE = False
try:
    from xgboost import XGBRegressor
    XGB_AVAILABLE = True
except (ImportError, Exception) as e:
    print(f"WARNING: XGBoost could not be loaded ({str(e)}). Falling back to other regressors.")

# Set Seaborn theme for beautiful plots
sns.set_theme(style="darkgrid", palette="muted")
plt.rcParams['figure.figsize'] = (10, 6)
plt.rcParams['font.family'] = 'sans-serif'

def run_ml_pipeline(dataset_path='dataset/housing.csv', output_dir='model', plots_dir='static/images'):
    print("--- STEP 1: LOADING DATA ---")
    df = pd.read_csv(dataset_path)
    print(f"Loaded dataset from {dataset_path} with shape: {df.shape}")
    
    print("\n--- STEP 2: DATA PREPROCESSING ---")
    
    # 2.1 Handle Duplicates
    duplicates_count = df.duplicated().sum()
    print(f"Number of duplicate rows: {duplicates_count}")
    if duplicates_count > 0:
        df.drop_duplicates(inplace=True)
        print(f"Removed duplicates. New shape: {df.shape}")
        
    # 2.2 Handle Missing Values
    print("Null values before imputation:")
    print(df.isnull().sum()[df.isnull().sum() > 0])
    
    # Compute medians/modes on the training distribution (or total dataset here since it's a script, 
    # but we'll store them for Flask deployment!)
    area_median = df['Area_SqFt'].median()
    year_median = df['Year_Built'].median()
    amenities_mode = df['Amenities'].mode()[0]
    
    df['Area_SqFt'] = df['Area_SqFt'].fillna(area_median)
    df['Year_Built'] = df['Year_Built'].fillna(year_median)
    df['Amenities'] = df['Amenities'].fillna(amenities_mode)
    print("Missing values after imputation:")
    print(df.isnull().sum())
    
    # 2.3 Detect and Handle Outliers (IQR Method)
    # Detect outliers in Area_SqFt and Price
    print("\nDetecting outliers using IQR...")
    cleaned_df = df.copy()
    
    for col in ['Area_SqFt', 'Price']:
        Q1 = cleaned_df[col].quantile(0.25)
        Q3 = cleaned_df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        outliers = cleaned_df[(cleaned_df[col] < lower_bound) | (cleaned_df[col] > upper_bound)]
        print(f"Column '{col}': found {len(outliers)} outliers outside bounds [{lower_bound:.2f}, {upper_bound:.2f}]")
        
        # Filter outliers
        cleaned_df = cleaned_df[(cleaned_df[col] >= lower_bound) & (cleaned_df[col] <= upper_bound)]
        
    print(f"Shape after removing outliers: {cleaned_df.shape} (Removed {len(df) - len(cleaned_df)} rows)")
    df = cleaned_df
    
    # 2.4 Encode Categorical Features
    print("\n--- STEP 3: ENCODING AND SCALING ---")
    categorical_cols = ['Location', 'Parking', 'Amenities']
    numerical_cols = ['Area_SqFt', 'Bedrooms', 'Bathrooms', 'Year_Built', 'Floors']
    
    # Set up OneHotEncoder
    encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
    encoded_features = encoder.fit_transform(df[categorical_cols])
    encoded_feature_names = encoder.get_feature_names_out(categorical_cols)
    
    encoded_df = pd.DataFrame(encoded_features, columns=encoded_feature_names, index=df.index)
    
    # Scale Numerical Features
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(df[numerical_cols])
    scaled_df = pd.DataFrame(scaled_features, columns=numerical_cols, index=df.index)
    
    # Combine processed features and target
    X = pd.concat([scaled_df, encoded_df], axis=1)
    y = df['Price']
    
    print(f"Features dimension: {X.shape}")
    
    # Save the feature column order for prediction consistency
    feature_columns = list(X.columns)
    
    # 2.5 Split dataset
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    print(f"Train size: {X_train.shape[0]}, Test size: {X_test.shape[0]}")
    
    print("\n--- STEP 4: MODEL TRAINING & COMPARISON ---")
    models = {
        'Linear Regression': LinearRegression(),
        'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42),
        'Gradient Boosting': GradientBoostingRegressor(n_estimators=150, learning_rate=0.1, random_state=42)
    }
    if XGB_AVAILABLE:
        models['XGBoost'] = XGBRegressor(n_estimators=150, learning_rate=0.1, random_state=42)
    
    results = {}
    best_r2 = -1
    best_model_name = None
    best_model = None
    
    for name, model in models.items():
        print(f"Training {name}...")
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        
        # Calculate Metrics
        r2 = r2_score(y_test, preds)
        mae = mean_absolute_error(y_test, preds)
        mse = mean_squared_error(y_test, preds)
        rmse = np.sqrt(mse)
        
        results[name] = {
            'R2': r2,
            'MAE': mae,
            'MSE': mse,
            'RMSE': rmse
        }
        print(f"{name} Results -> R²: {r2:.4f} | MAE: ₹{mae:,.2f} | RMSE: ₹{rmse:,.2f}")
        
        if r2 > best_r2:
            best_r2 = r2
            best_model_name = name
            best_model = model
            
    print(f"\nBest Model: {best_model_name} with R² Score: {best_r2:.4f}")
    
    # Ensure folders exist
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(plots_dir, exist_ok=True)
    
    # Save best model and scaler
    model_path = os.path.join(output_dir, 'model.pkl')
    scaler_path = os.path.join(output_dir, 'scaler.pkl')
    meta_path = os.path.join(output_dir, 'preprocessing_meta.pkl')
    
    models_to_save = {
        'best_model': best_model,
        'Gradient Boosting': models.get('Gradient Boosting'),
        'Linear Regression': models.get('Linear Regression'),
        'Random Forest': models.get('Random Forest')
    }
    with open(model_path, 'wb') as f:
        pickle.dump(models_to_save, f)
        
    with open(scaler_path, 'wb') as f:
        pickle.dump(scaler, f)
        
    # Save encoder and other preprocessing meta data
    preprocessing_meta = {
        'area_median': area_median,
        'year_median': year_median,
        'amenities_mode': amenities_mode,
        'encoder': encoder,
        'categorical_cols': categorical_cols,
        'numerical_cols': numerical_cols,
        'feature_columns': feature_columns,
        'best_model_name': best_model_name,
        'best_r2': best_r2,
        'metrics': results
    }
    with open(meta_path, 'wb') as f:
        pickle.dump(preprocessing_meta, f)
        
    print(f"Saved best model, scaler, and preprocessing metadata to '{output_dir}/'")
    
    # --- STEP 5: VISUALIZATIONS ---
    print("\n--- STEP 5: GENERATING CHARTS ---")
    
    # 5.1 Model Comparison Chart
    names = list(results.keys())
    r2_scores = [results[n]['R2'] * 100 for n in names] # Percentage
    
    plt.figure(figsize=(10, 6))
    bars = plt.bar(names, r2_scores, color=['#3498db', '#9b59b6', '#f1c40f', '#2ecc71'], width=0.6)
    plt.title('Model R² Score Comparison (%)', fontsize=15, pad=15, fontweight='bold')
    plt.ylabel('R² Score (%)', fontsize=12)
    plt.ylim(80, 100)
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2.0, height + 0.5, f'{height:.2f}%', ha='center', va='bottom', fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, 'model_comparison.png'), dpi=150)
    plt.close()
    
    # 5.2 Actual vs Predicted Chart
    best_preds = best_model.predict(X_test)
    plt.figure(figsize=(10, 7))
    sns.scatterplot(x=y_test, y=best_preds, alpha=0.5, color='#8e44ad')
    # Ideal prediction line
    ideal_min = min(y_test.min(), best_preds.min())
    ideal_max = max(y_test.max(), best_preds.max())
    plt.plot([ideal_min, ideal_max], [ideal_min, ideal_max], color='#e74c3c', linestyle='--', linewidth=2.5, label='Perfect Prediction')
    
    plt.title(f'Actual vs Predicted House Prices ({best_model_name})', fontsize=15, pad=15, fontweight='bold')
    plt.xlabel('Actual Price (₹)', fontsize=12)
    plt.ylabel('Predicted Price (₹)', fontsize=12)
    plt.gca().xaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f"₹{x:,.0f}"))
    plt.gca().yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f"₹{x:,.0f}"))
    plt.legend(fontsize=11)
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, 'actual_vs_predicted.png'), dpi=150)
    plt.close()
    
    # 5.3 Feature Importance Chart (using Random Forest/Gradient Boosting/XGBoost)
    if hasattr(best_model, 'feature_importances_'):
        importances = best_model.feature_importances_
        indices = np.argsort(importances)[::-1]
        
        plt.figure(figsize=(10, 7))
        # Take top 10 features if too many
        top_n = min(12, len(feature_columns))
        top_importances = importances[indices[:top_n]]
        top_features = [feature_columns[i] for i in indices[:top_n]]
        
        sns.barplot(x=top_importances, y=top_features, hue=top_features, palette="viridis", legend=False)
        plt.title(f'Feature Importance Chart ({best_model_name})', fontsize=15, pad=15, fontweight='bold')
        plt.xlabel('Relative Importance Score', fontsize=12)
        plt.ylabel('Features', fontsize=12)
        plt.tight_layout()
        plt.savefig(os.path.join(plots_dir, 'feature_importance.png'), dpi=150)
        plt.close()
        print("Feature importance chart saved successfully!")
    else:
        print("Model does not support feature importances (e.g., Linear Regression was selected). Skipping feature importance plot.")
        
    # Write dataset stats for Flask app to load easily
    stats = {
        'count': int(df.shape[0]),
        'mean_price': float(df['Price'].mean()),
        'median_price': float(df['Price'].median()),
        'min_price': float(df['Price'].min()),
        'max_price': float(df['Price'].max()),
        'mean_area': float(df['Area_SqFt'].mean()),
        'avg_bedrooms': float(df['Bedrooms'].mean()),
        'avg_bathrooms': float(df['Bathrooms'].mean()),
        'top_location': str(df['Location'].mode()[0])
    }
    with open(os.path.join(output_dir, 'dataset_stats.pkl'), 'wb') as f:
        pickle.dump(stats, f)
    print("Dataset stats saved successfully!")
    
    # Create comparison results dataframe printout
    print("\nFinal Model Comparison Summary Table:")
    results_df = pd.DataFrame(results).T
    print(results_df)
    
if __name__ == '__main__':
    run_ml_pipeline()
