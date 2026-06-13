import os
import pickle
import numpy as np
import pandas as pd
from flask import Flask, render_template, request, jsonify, session, redirect, url_for, flash
from flask_sqlalchemy import SQLAlchemy
from flask_bcrypt import Bcrypt
from datetime import datetime

app = Flask(__name__)

# Database and Security Configurations
db_url = os.environ.get('DATABASE_URL')
if db_url:
    if db_url.startswith("postgres://"):
        db_url = db_url.replace("postgres://", "postgresql+pg8000://", 1)
    elif db_url.startswith("postgresql://"):
        db_url = db_url.replace("postgresql://", "postgresql+pg8000://", 1)
    app.config['SQLALCHEMY_DATABASE_URI'] = db_url
else:
    app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///housing_app.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
app.config['SECRET_KEY'] = os.environ.get('SECRET_KEY', 'aurapred-secret-key-3918-auth')

db = SQLAlchemy(app)
bcrypt = Bcrypt(app)

# Database Models
class User(db.Model):
    __tablename__ = 'users'
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), unique=True, nullable=False)
    email = db.Column(db.String(120), unique=True, nullable=False)
    password_hash = db.Column(db.String(128), nullable=False)
    is_admin = db.Column(db.Boolean, default=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    predictions = db.relationship('PredictionLog', backref='user', lazy=True)

class PredictionLog(db.Model):
    __tablename__ = 'prediction_logs'
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('users.id'), nullable=True)
    area = db.Column(db.Float, nullable=False)
    bedrooms = db.Column(db.Integer, nullable=False)
    bathrooms = db.Column(db.Float, nullable=False)
    location = db.Column(db.String(80), nullable=False)
    year_built = db.Column(db.Integer, nullable=False)
    parking = db.Column(db.String(10), nullable=False)
    floors = db.Column(db.Integer, nullable=False)
    amenities = db.Column(db.String(40), nullable=False)
    predicted_price = db.Column(db.Float, nullable=False)
    timestamp = db.Column(db.DateTime, default=datetime.utcnow)

# Constants
MODEL_DIR = 'model'
MODEL_PATH = os.path.join(MODEL_DIR, 'model.pkl')
SCALER_PATH = os.path.join(MODEL_DIR, 'scaler.pkl')
META_PATH = os.path.join(MODEL_DIR, 'preprocessing_meta.pkl')
STATS_PATH = os.path.join(MODEL_DIR, 'dataset_stats.pkl')

# Global variables for ML artifacts
model = None
scaler = None
meta = None
stats = None
models_dict = None
df_housing = None

def load_ml_artifacts():
    global model, scaler, meta, stats, models_dict, df_housing
    try:
        if os.path.exists(MODEL_PATH):
            with open(MODEL_PATH, 'rb') as f:
                loaded = pickle.load(f)
                if isinstance(loaded, dict):
                    models_dict = loaded
                    model = models_dict['best_model']
                else:
                    model = loaded
                    models_dict = {'best_model': model, 'Gradient Boosting': model}
        if os.path.exists(SCALER_PATH):
            with open(SCALER_PATH, 'rb') as f:
                scaler = pickle.load(f)
        if os.path.exists(META_PATH):
            with open(META_PATH, 'rb') as f:
                meta = pickle.load(f)
        if os.path.exists(STATS_PATH):
            with open(STATS_PATH, 'rb') as f:
                stats = pickle.load(f)
        if os.path.exists('dataset/housing.csv'):
            df_housing = pd.read_csv('dataset/housing.csv')
        print("ML Pipeline artifacts loaded successfully!")
    except Exception as e:
        print(f"Error loading ML artifacts: {str(e)}")

# Load artifacts on startup
load_ml_artifacts()

# Create database and seed admin on startup
with app.app_context():
    try:
        db.create_all()
        admin_user = User.query.filter_by(username='admin').first()
        if not admin_user:
            hashed_pw = bcrypt.generate_password_hash('adminpassword').decode('utf-8')
            default_admin = User(
                username='admin',
                email='admin@aurapred.com',
                password_hash=hashed_pw,
                is_admin=True
            )
            db.session.add(default_admin)
            db.session.commit()
            print("Seeded default admin user (admin / adminpassword)")
    except Exception as db_init_err:
        print(f"Error initializing or seeding SQLite database: {str(db_init_err)}")

def get_default_stats():
    return {
        'count': 4983,
        'mean_price': 43700000.0,
        'median_price': 42240000.0,
        'min_price': 2400000.0,
        'max_price': 198400000.0,
        'mean_area': 4250.0,
        'avg_bedrooms': 3.5,
        'avg_bathrooms': 2.8,
        'top_location': 'Suburbs'
    }

@app.route('/')
def home():
    # Pass dataset stats to home page for interactive cards
    current_stats = stats if stats else get_default_stats()
    return render_template('index.html', stats=current_stats, meta=meta)

@app.route('/about')
def about():
    # Pass model comparison metrics to about page
    current_meta = meta if meta else {
        'best_model_name': 'Gradient Boosting',
        'best_r2': 0.9844,
        'metrics': {
            'Linear Regression': {'R2': 0.9822, 'MAE': 2469259, 'RMSE': 4310188},
            'Random Forest': {'R2': 0.9742, 'MAE': 3143158, 'RMSE': 5145430},
            'Gradient Boosting': {'R2': 0.9844, 'MAE': 2116150, 'RMSE': 4028599}
        }
    }
    return render_template('about.html', meta=current_meta)

@app.route('/predict', methods=['POST'])
def predict():
    # Check if artifacts are loaded
    if model is None or scaler is None or meta is None:
        load_ml_artifacts()
        if model is None:
            return jsonify({
                'success': False,
                'error': 'ML model artifacts are not loaded on server. Please run training pipeline first.'
            }), 500

    # Parse and validate inputs
    is_ajax = request.is_json or request.headers.get('X-Requested-With') == 'XMLHttpRequest'
    
    try:
        if request.is_json:
            data = request.get_json()
        else:
            data = request.form

        # Extract features
        area = data.get('Area_SqFt')
        bedrooms = data.get('Bedrooms')
        bathrooms = data.get('Bathrooms')
        location = data.get('Location')
        year_built = data.get('Year_Built')
        parking = data.get('Parking')
        floors = data.get('Floors')
        amenities = data.get('Amenities')

        # Input Validation
        errors = []
        try:
            area = float(area) if area else None
            if area is None or area <= 0 or area > 30000:
                errors.append("Area must be a positive number up to 30,000 sq ft.")
        except ValueError:
            errors.append("Area must be a valid number.")

        try:
            bedrooms = int(bedrooms) if bedrooms else None
            if bedrooms is None or bedrooms < 1 or bedrooms > 10:
                errors.append("Bedrooms must be an integer between 1 and 10.")
        except ValueError:
            errors.append("Bedrooms must be a valid integer.")

        try:
            bathrooms = float(bathrooms) if bathrooms else None
            if bathrooms is None or bathrooms < 1.0 or bathrooms > 8.0:
                errors.append("Bathrooms must be between 1.0 and 8.0.")
        except ValueError:
            errors.append("Bathrooms must be a valid number.")

        try:
            year_built = int(year_built) if year_built else None
            if year_built is None or year_built < 1800 or year_built > 2026:
                errors.append("Year Built must be between 1800 and 2026.")
        except ValueError:
            errors.append("Year Built must be a valid integer.")

        try:
            floors = int(floors) if floors else None
            if floors is None or floors < 1 or floors > 5:
                errors.append("Floors must be between 1 and 5.")
        except ValueError:
            errors.append("Floors must be a valid integer.")

        if not location or location not in ['Downtown', 'Suburbs', 'Uptown', 'Rural', 'Waterfront']:
            errors.append("Location must be one of: Downtown, Suburbs, Uptown, Rural, Waterfront.")

        if not parking or parking not in ['Yes', 'No']:
            errors.append("Parking must be 'Yes' or 'No'.")

        if not amenities or amenities not in ['Basic', 'Standard', 'Luxury']:
            errors.append("Amenities must be 'Basic', 'Standard' or 'Luxury'.")

        if errors:
            if is_ajax:
                return jsonify({'success': False, 'errors': errors}), 400
            else:
                return render_template('result.html', error="<br>".join(errors), inputs=data)

        # 2. Impute default values if any optional field is missing
        # (Though we validated and forced them, it's good practice for API endpoints!)
        area = area if area is not None else meta['area_median']
        year_built = year_built if year_built is not None else meta['year_median']
        amenities = amenities if amenities is not None else meta['amenities_mode']

        # 3. Create DataFrame for preprocessing to match feature orders
        # Numerical DF
        num_data = pd.DataFrame([{
            'Area_SqFt': area,
            'Bedrooms': bedrooms,
            'Bathrooms': bathrooms,
            'Year_Built': year_built,
            'Floors': floors
        }])
        
        # Categorical DF
        cat_data = pd.DataFrame([{
            'Location': location,
            'Parking': parking,
            'Amenities': amenities
        }])

        # Scale Numerical Features
        scaled_num = scaler.transform(num_data[meta['numerical_cols']])
        scaled_df = pd.DataFrame(scaled_num, columns=meta['numerical_cols'])

        # Encode Categorical Features
        encoded_cat = meta['encoder'].transform(cat_data[meta['categorical_cols']])
        encoded_names = meta['encoder'].get_feature_names_out(meta['categorical_cols'])
        encoded_df = pd.DataFrame(encoded_cat, columns=encoded_names)

        # Combine Features
        X_input = pd.concat([scaled_df, encoded_df], axis=1)

        # Reindex Columns to match the exact training column order
        X_input = X_input.reindex(columns=meta['feature_columns'], fill_value=0.0)

        # 4. Predict using standard loaded model
        predicted_price = model.predict(X_input)[0]
        predicted_price = max(10000.0, float(predicted_price))
        formatted_price = f"₹{predicted_price:,.0f}"

        # 4.05 Save prediction log in database if user is authenticated (or save always)
        user_id = session.get('user_id')
        new_log = PredictionLog(
            user_id=user_id,
            area=float(area),
            bedrooms=int(bedrooms),
            bathrooms=float(bathrooms),
            location=location,
            year_built=int(year_built),
            parking=parking,
            floors=int(floors),
            amenities=amenities,
            predicted_price=float(predicted_price)
        )
        try:
            db.session.add(new_log)
            db.session.commit()
        except Exception as db_err:
            db.session.rollback()
            print(f"Error saving prediction log to database: {str(db_err)}")

        # 4.1 Compute Multi-Model pricing comparison
        gb_price = predicted_price
        lr_price = predicted_price
        rf_price = predicted_price
        
        if models_dict:
            try:
                if models_dict.get('Gradient Boosting'):
                    gb_price = max(10000.0, float(models_dict['Gradient Boosting'].predict(X_input)[0]))
                if models_dict.get('Linear Regression'):
                    lr_price = max(10000.0, float(models_dict['Linear Regression'].predict(X_input)[0]))
                if models_dict.get('Random Forest'):
                    rf_price = max(10000.0, float(models_dict['Random Forest'].predict(X_input)[0]))
            except Exception as ex:
                print(f"Multi-model regression failed: {str(ex)}")

        # 4.2 Compute Localized explainability contributions based on linear coefficients
        explainability = []
        if models_dict and models_dict.get('Linear Regression') and scaler:
            try:
                lr = models_dict['Linear Regression']
                coefs = lr.coef_
                scaled_vals = X_input.values[0]
                col_indices = {col: i for i, col in enumerate(meta['feature_columns'])}
                
                # Coef * Scaled Value
                area_contrib = coefs[col_indices['Area_SqFt']] * scaled_vals[col_indices['Area_SqFt']]
                beds_contrib = coefs[col_indices['Bedrooms']] * scaled_vals[col_indices['Bedrooms']]
                baths_contrib = coefs[col_indices['Bathrooms']] * scaled_vals[col_indices['Bathrooms']]
                year_contrib = coefs[col_indices['Year_Built']] * scaled_vals[col_indices['Year_Built']]
                floor_contrib = coefs[col_indices['Floors']] * scaled_vals[col_indices['Floors']]
                
                # Sum categorized one-hot columns
                loc_contrib = 0.0
                for cat in ['Downtown', 'Suburbs', 'Uptown', 'Rural', 'Waterfront']:
                    c_name = f"Location_{cat}"
                    if c_name in col_indices:
                        loc_contrib += coefs[col_indices[c_name]] * scaled_vals[col_indices[c_name]]
                
                park_contrib = 0.0
                for cat in ['Yes', 'No']:
                    c_name = f"Parking_{cat}"
                    if c_name in col_indices:
                        park_contrib += coefs[col_indices[c_name]] * scaled_vals[col_indices[c_name]]
                        
                amenities_contrib = 0.0
                for cat in ['Basic', 'Standard', 'Luxury']:
                    c_name = f"Amenities_{cat}"
                    if c_name in col_indices:
                        amenities_contrib += coefs[col_indices[c_name]] * scaled_vals[col_indices[c_name]]
                
                explainability = [
                    {"feature": "Property Area Size", "weight": round(float(area_contrib), 2)},
                    {"feature": "Bedrooms Count", "weight": round(float(beds_contrib), 2)},
                    {"feature": "Bathrooms Count", "weight": round(float(baths_contrib), 2)},
                    {"feature": "Location Premium", "weight": round(float(loc_contrib), 2)},
                    {"feature": "Building History (Age)", "weight": round(float(year_contrib), 2)},
                    {"feature": "Garage Space (Parking)", "weight": round(float(park_contrib), 2)},
                    {"feature": "Floors Count", "weight": round(float(floor_contrib), 2)},
                    {"feature": "Amenities Quality Grade", "weight": round(float(amenities_contrib), 2)}
                ]
            except Exception as ex:
                print(f"Explainability calculation skipped: {str(ex)}")

        # 4.3 Property Matchmaker Search (top 3 similar properties)
        matches = []
        if df_housing is not None:
            try:
                similar_df = df_housing[df_housing['Location'] == location].copy()
                if len(similar_df) < 3:
                    similar_df = df_housing.copy()
                
                # Normalized distance metrics
                similar_df['distance'] = (
                    ((similar_df['Area_SqFt'] - area) / 4000).pow(2) + 
                    ((similar_df['Price'] - predicted_price) / 20000000).pow(2)
                ).pow(0.5)
                
                top_matches = similar_df.sort_values(by='distance').head(3)
                for _, row in top_matches.iterrows():
                    matches.append({
                        'area': int(row['Area_SqFt']) if not pd.isna(row['Area_SqFt']) else 2500,
                        'beds': int(row['Bedrooms']),
                        'baths': float(row['Bathrooms']),
                        'location': str(row['Location']),
                        'year': int(row['Year_Built']) if not pd.isna(row['Year_Built']) else 2005,
                        'parking': str(row['Parking']),
                        'floors': int(row['Floors']),
                        'amenities': str(row['Amenities']),
                        'price': float(row['Price']),
                        'formatted_price': f"₹{float(row['Price']):,.0f}"
                    })
            except Exception as ex:
                print(f"Matchmaker search failed: {str(ex)}")

        # 4.4 Dynamic Historical Appreciation Indices (10 years)
        trends = []
        try:
            rate = {'Waterfront': 0.085, 'Downtown': 0.07, 'Uptown': 0.06, 'Suburbs': 0.045, 'Rural': 0.025}.get(location, 0.05)
            for yr in range(2017, 2027):
                yr_price = predicted_price * ((1 + rate) ** (yr - 2026))
                trends.append({
                    'year': yr,
                    'price': round(yr_price, 2),
                    'formatted_price': f"₹{yr_price:,.0f}"
                })
        except Exception as ex:
            print(f"Trend generation failed: {str(ex)}")

        response_data = {
            'success': True,
            'price': round(predicted_price, 2),
            'formatted_price': formatted_price,
            'model_used': meta.get('best_model_name', 'Gradient Boosting Regressor'),
            'r2_score': f"{meta.get('best_r2', 0.9844) * 100:.2f}%",
            'multi_model_prices': {
                'Gradient Boosting': f"₹{gb_price:,.0f}",
                'Linear Regression': f"₹{lr_price:,.0f}",
                'Random Forest': f"₹{rf_price:,.0f}"
            },
            'explainability_contributions': explainability,
            'recommended_matches': matches,
            'historical_appreciation_trends': trends,
            'inputs': {
                'Area_SqFt': int(area),
                'Bedrooms': int(bedrooms),
                'Bathrooms': float(bathrooms),
                'Location': location,
                'Year_Built': int(year_built),
                'Parking': parking,
                'Floors': int(floors),
                'Amenities': amenities
            }
        }

        if is_ajax:
            return jsonify(response_data)
        else:
            return render_template('result.html', result=response_data)

    except Exception as e:
        error_msg = f"An error occurred during prediction: {str(e)}"
        if is_ajax:
            return jsonify({'success': False, 'error': error_msg}), 500
        else:
            return render_template('result.html', error=error_msg, inputs=request.form)

@app.route('/register', methods=['GET', 'POST'])
def register():
    if session.get('user_id'):
        return redirect(url_for('dashboard'))
        
    if request.method == 'POST':
        username = request.form.get('username', '').strip()
        email = request.form.get('email', '').strip()
        password = request.form.get('password', '')
        confirm_password = request.form.get('confirm_password', '')
        
        errors = []
        if not username or len(username) < 3:
            errors.append("Username must be at least 3 characters long.")
        if not email or '@' not in email:
            errors.append("Please enter a valid email address.")
        if not password or len(password) < 6:
            errors.append("Password must be at least 6 characters long.")
        if password != confirm_password:
            errors.append("Passwords do not match.")
            
        if not errors:
            existing_user = User.query.filter((User.username == username) | (User.email == email)).first()
            if existing_user:
                errors.append("Username or Email already registered.")
                
        if errors:
            return render_template('register.html', errors=errors, inputs=request.form)
            
        hashed_pw = bcrypt.generate_password_hash(password).decode('utf-8')
        new_user = User(username=username, email=email, password_hash=hashed_pw)
        db.session.add(new_user)
        db.session.commit()
        
        flash("Registration successful! Please login.", "success")
        return redirect(url_for('login'))
        
    return render_template('register.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if session.get('user_id'):
        return redirect(url_for('dashboard'))
        
    if request.method == 'POST':
        username_or_email = request.form.get('username_or_email', '').strip()
        password = request.form.get('password', '')
        
        user = User.query.filter((User.username == username_or_email) | (User.email == username_or_email)).first()
        if user and bcrypt.check_password_hash(user.password_hash, password):
            session['user_id'] = user.id
            session['username'] = user.username
            session['is_admin'] = user.is_admin
            flash("Logged in successfully!", "success")
            return redirect(url_for('dashboard'))
        else:
            return render_template('login.html', error="Invalid username/email or password.", inputs=request.form)
            
    return render_template('login.html')

@app.route('/logout')
def logout():
    session.clear()
    flash("You have logged out successfully.", "info")
    return redirect(url_for('home'))

@app.route('/dashboard')
def dashboard():
    if not session.get('user_id'):
        flash("Please login to access the dashboard.", "warning")
        return redirect(url_for('login'))
        
    user = db.session.get(User, session['user_id'])
    if not user:
        session.clear()
        return redirect(url_for('login'))
        
    user_logs = PredictionLog.query.filter_by(user_id=user.id).order_by(PredictionLog.timestamp.desc()).all()
    
    # User stats
    log_count = len(user_logs)
    avg_price = 0.0
    top_loc = "N/A"
    if log_count > 0:
        prices = [log.predicted_price for log in user_logs]
        avg_price = sum(prices) / log_count
        
        locations = [log.location for log in user_logs]
        top_loc = max(set(locations), key=locations.count)
        
    stats = {
        'count': log_count,
        'avg_price': avg_price,
        'top_location': top_loc
    }
    
    return render_template('dashboard.html', user=user, logs=user_logs, stats=stats)

@app.route('/admin')
def admin():
    if not session.get('user_id') or not session.get('is_admin'):
        flash("Unauthorized access. Admin privileges required.", "danger")
        return redirect(url_for('home'))
        
    all_users = User.query.order_by(User.created_at.desc()).all()
    all_logs = PredictionLog.query.order_by(PredictionLog.timestamp.desc()).all()
    
    # Global stats
    users_count = len(all_users)
    logs_count = len(all_logs)
    avg_price = 0.0
    if logs_count > 0:
        prices = [log.predicted_price for log in all_logs]
        avg_price = sum(prices) / logs_count
        
    stats = {
        'users_count': users_count,
        'logs_count': logs_count,
        'avg_price': avg_price
    }
    
    # Dynamic metrics grouped by location for Chart.js distribution chart in admin panel
    loc_data = db.session.query(
        PredictionLog.location,
        db.func.count(PredictionLog.id),
        db.func.avg(PredictionLog.predicted_price)
    ).group_by(PredictionLog.location).all()
    
    chart_data = {
        'labels': [row[0] for row in loc_data],
        'counts': [row[1] for row in loc_data],
        'avg_prices': [round(float(row[2]), 2) if row[2] else 0.0 for row in loc_data]
    }
    
    return render_template('admin_dashboard.html', users=all_users, logs=all_logs, stats=stats, chart_data=chart_data)

if __name__ == '__main__':
    # Enable debug mode on local development port 5001
    port = int(os.environ.get('PORT', 5001))
    app.run(host='0.0.0.0', port=port, debug=True)
