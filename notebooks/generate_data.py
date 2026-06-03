import os
import numpy as np
import pandas as pd

def generate_dataset(output_path, num_samples=5000, seed=42):
    np.random.seed(seed)
    
    # 1. Generate features
    area = np.random.randint(500, 8000, size=num_samples)
    bedrooms = np.random.randint(1, 7, size=num_samples)
    
    # Bathrooms are correlated with bedrooms, mostly (e.g., bedrooms + [-1, 0, 1] capped between 1 and 5)
    bathrooms = np.zeros(num_samples)
    for i in range(num_samples):
        b = bedrooms[i] + np.random.choice([-1.0, -0.5, 0.0, 0.5, 1.0], p=[0.1, 0.2, 0.4, 0.2, 0.1])
        bathrooms[i] = np.clip(b, 1.0, 5.0)
        
    locations = ['Downtown', 'Suburbs', 'Uptown', 'Rural', 'Waterfront']
    location = np.random.choice(locations, size=num_samples, p=[0.25, 0.35, 0.20, 0.15, 0.05])
    
    year_built = np.random.randint(1950, 2027, size=num_samples)
    parking = np.random.choice(['Yes', 'No'], size=num_samples, p=[0.7, 0.3])
    floors = np.random.randint(1, 5, size=num_samples)
    
    amenities_opts = ['Basic', 'Standard', 'Luxury']
    amenities = np.random.choice(amenities_opts, size=num_samples, p=[0.4, 0.45, 0.15])
    
    # 2. Generative formula for Price (with non-linearities and interaction effects)
    base_price = 50000
    price_area = area * 165
    price_beds = bedrooms * 28000
    price_baths = bathrooms * 35000
    
    # Location factors
    loc_map = {
        'Downtown': 135000,
        'Suburbs': 45000,
        'Uptown': 95000,
        'Rural': -35000,
        'Waterfront': 280000
    }
    price_loc = np.array([loc_map[loc] for loc in location])
    
    # Age factor (newer houses are more expensive, depreciation of older houses)
    price_age = (year_built - 1950) * 1600
    
    # Parking & Floors
    price_parking = np.array([22000 if p == 'Yes' else 0 for p in parking])
    price_floors = floors * 18000
    
    # Amenities factor
    amenities_map = {
        'Basic': 0,
        'Standard': 38000,
        'Luxury': 120000
    }
    price_amenities = np.array([amenities_map[a] for a in amenities])
    
    # Interaction Effects (making it non-linear so tree models excel)
    # Effect 1: Waterfront + Luxury is extremely premium (+150k)
    interaction_waterfront_luxury = np.zeros(num_samples)
    for i in range(num_samples):
        if location[i] == 'Waterfront' and amenities[i] == 'Luxury':
            interaction_waterfront_luxury[i] = 160000
            
    # Effect 2: Large area in Downtown or Waterfront gets premium pricing
    interaction_large_premium = np.zeros(num_samples)
    for i in range(num_samples):
        if (location[i] == 'Downtown' or location[i] == 'Waterfront') and area[i] > 4000:
            interaction_large_premium[i] = (area[i] - 4000) * 45
            
    # Random Gaussian Noise (low noise to target high R² around 96%)
    noise = np.random.normal(0, 18000, size=num_samples)
    
    # Calculate Price
    price = (base_price + price_area + price_beds + price_baths + price_loc + 
             price_age + price_parking + price_floors + price_amenities + 
             interaction_waterfront_luxury + interaction_large_premium + noise)
    
    # Scale to INR (1 USD ≈ 80 INR multiplier for realistic Indian housing market prices)
    price = price * 80.0
    
    # Clip price to ensure no negative prices (minimum price ₹2,400,000 / 24 Lakhs)
    price = np.clip(price, 2400000, None)
    
    # Create DataFrame
    df = pd.DataFrame({
        'Area_SqFt': area,
        'Bedrooms': bedrooms,
        'Bathrooms': bathrooms,
        'Location': location,
        'Year_Built': year_built,
        'Parking': parking,
        'Floors': floors,
        'Amenities': amenities,
        'Price': price
    })
    
    # 3. Inject imperfections to show preprocessing pipeline strength
    print("Injecting missing values...")
    # Inject ~1% missing values in Area_SqFt, Year_Built, and Amenities
    missing_idx_area = np.random.choice(num_samples, size=int(num_samples * 0.012), replace=False)
    missing_idx_year = np.random.choice(num_samples, size=int(num_samples * 0.008), replace=False)
    missing_idx_amenities = np.random.choice(num_samples, size=int(num_samples * 0.01), replace=False)
    
    df.loc[missing_idx_area, 'Area_SqFt'] = np.nan
    df.loc[missing_idx_year, 'Year_Built'] = np.nan
    df.loc[missing_idx_amenities, 'Amenities'] = None
    
    # Inject 12 duplicate rows
    print("Injecting duplicates...")
    dup_indices = np.random.choice(num_samples, size=12, replace=False)
    dup_rows = df.iloc[dup_indices].copy()
    df = pd.concat([df, dup_rows], ignore_index=True)
    
    # Inject 15 extreme outliers
    print("Injecting outliers...")
    outliers_idx = np.random.choice(len(df), size=15, replace=False)
    
    for idx in outliers_idx[:5]:
        # Huge area, very low price (underpriced luxury mansion)
        df.loc[idx, 'Area_SqFt'] = 22000.0
        df.loc[idx, 'Price'] = 85000.0 * 80.0
        
    for idx in outliers_idx[5:10]:
        # Tiny area, extremely high price (overpriced tiny home)
        df.loc[idx, 'Area_SqFt'] = 350.0
        df.loc[idx, 'Price'] = 2800000.0 * 80.0
        
    for idx in outliers_idx[10:15]:
        # Massive price multiplier outlier
        df.loc[idx, 'Price'] = df.loc[idx, 'Price'] * 8.5
        
    # Shuffle dataframe
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    
    # Ensure dataset directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"Dataset saved successfully to {output_path}!")
    print(f"Total samples: {len(df)}")
    print(f"Duplicate rows count: {df.duplicated().sum()}")
    print(f"Nulls count per feature:\n{df.isnull().sum()}")

if __name__ == '__main__':
    generate_dataset('dataset/housing.csv')
