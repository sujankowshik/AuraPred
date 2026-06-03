# AuraPred - Premium ML House Price Prediction Web Application

AuraPred is a state-of-the-art Real Estate Property Price Prediction Web Application built using Python Flask, Scikit-Learn, and modern vanilla CSS frontend assets. The application allows users to submit property details (such as Area, Bedrooms, Bathrooms, Location, Year built, Parking availability, Floors, and Amenities) and predicts the estimated house price instantaneously using a trained machine learning model.

Designed with a sleek **Glassmorphism Theme**, fluid responsive grids, dynamic sliders, real-time input validation, a visual metrics dashboard, prediction history, and a toggling Dark/Light mode, AuraPred represents a professional, production-level, deployment-ready software product.

---

## 🏗️ Architecture & Preprocessing Pipeline

Our ML pipeline implements a comprehensive, robust preprocessing script to handle noisy real-world property listing files:

1. **Deduplication**: Drops exact duplicates in listing databases using Pandas.
2. **Imputation of Nulls**: Dynamically replaces null variables. Imputes numerical properties (Area Sq Ft, Construction Year) with training distribution medians, and categorical features (Amenities) with training modes.
3. **Outlier Filtering**: Removes outliers using the **Interquartile Range (IQR)** method. Listings with prices or sizes that exceed $1.5 \times IQR$ are capped and filtered.
4. **One-Hot Encoding**: Converts qualitative inputs (Location, Parking, Amenities) to numerical features using `OneHotEncoder`.
5. **Feature Scaling**: Standardizes high-variance numerical features using a `StandardScaler`.

---

## 📊 Machine Learning Model Comparison

We trained multiple regression models on a realistic housing dataset containing 5,000 listings. Gradient Boosting Regressor performed the best and was selected for production:

| Algorithm Model | R² Score (Accuracy) | Mean Absolute Error (MAE) | Root Mean Squared Error (RMSE) | State |
| :--- | :---: | :---: | :---: | :---: |
| **Gradient Boosting Regressor** | **98.44%** | **$26,451.88** | **$50,357.49** | 🥇 Selected |
| **Linear Regression** | **98.22%** | **$30,865.74** | **$53,877.35** | 🥈 Fast Benchmark |
| **Random Forest Regressor** | **97.42%** | **$39,527.02** | **$64,842.84** | 🥉 Ensemble |

---

## 📂 Project Structure

```
house-price-prediction/
├── app.py                      # Flask backend API & routing
├── Dockerfile                  # Multi-stage production container build
├── Procfile                    # Web service process configuration
├── requirements.txt            # Python dependencies specification
├── README.md                   # Project documentation
│
├── dataset/
│   └── housing.csv             # 5,000-listing housing dataset
│
├── model/                      # ML pipeline exports
│   ├── model.pkl               # Pickled Gradient Boosting Model
│   ├── scaler.pkl              # Fitted StandardScaler artifact
│   ├── preprocessing_meta.pkl  # Imputers, encoders, and columns meta
│   └── dataset_stats.pkl       # Descriptive metrics for analytics tab
│
├── notebooks/                  # Training notebooks and helper scripts
│   ├── generate_data.py        # Generates realistic synthetic dataset
│   ├── generate_notebook.py    # Generates Jupyter Notebook from cells
│   ├── model_training.py       # ML Pipeline training execution script
│   └── model_training.ipynb    # Documented pipeline research notebook
│
├── static/                     # Frontend static assets
│   ├── css/
│   │   └── style.css           # Premium theme variables & layout stylesheet
│   ├── js/
│   │   └── script.js           # AJAX controllers, counters, theme switches, history caches
│   └── images/                 # Exported model evaluation PNGs
│       ├── actual_vs_predicted.png
│       ├── feature_importance.png
│       └── model_comparison.png
│
└── templates/                  # Frontend HTML views
    ├── index.html              # Main dashboard (Prediction, Charts, History tabs)
    ├── result.html             # Standalone prediction result page (Traditional Fallback)
    └── about.html              # Technical model documentation page
```

---

## ⚡ Quick Start & Installation

### Local Development (Pip & Virtual Environment)

1. **Clone the repository and go to the directory:**
   ```bash
   cd house-price-prediction
   ```

2. **Create a Python virtual environment and activate it:**
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   ```

3. **Install python dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Run the dataset generator & training pipeline (fits models, outputs pickled files & charts):**
   ```bash
   python3 notebooks/generate_data.py
   python3 notebooks/model_training.py
   python3 notebooks/generate_notebook.py
   ```

5. **Start the local Flask server:**
   ```bash
   python3 app.py
   ```

   Open your browser and navigate to: `http://localhost:5001`

---

## 🐳 Containerized Running (Docker)

Ensure Docker is installed on your machine.

1. **Build the production Docker image:**
   ```bash
   docker build -t aura-pred .
   ```

2. **Spin up the container on port 5001:**
   ```bash
   docker run -d -p 5001:5001 --name aurapred-app aura-pred
   ```

   Visit the valuation dashboard in your browser: `http://localhost:5001`

---

## ☁️ Deployment Setups

### Render or Railway (PaaS)
1. **GitHub Sync**: Push the repository to GitHub.
2. **Create Web Service**: Link your GitHub repository to Render/Railway.
3. **Configure Environment**:
   - Runtime: `Python` or `Docker` (Render automatically detects `Dockerfile` or uses the `Procfile` / `requirements.txt`).
   - Start Command (if using Gunicorn directly): `gunicorn app:app` or `web: gunicorn app:app`.
   - Port environment: `PORT=5001` (or leaves it dynamic as default).

---

## 📈 Future Enhancements
- **Dynamic Geo-Coordinates mapping**: Integrate Leaflet.js to pinpoint home location on active map grids.
- **Inflation adjustments API**: Fetch dynamic mortgage rates and inflation indices using open APIs to adjust pricing.
- **PDF Report Downloads**: Expose a Flask route to generate fully stylized, downloadable PDF property valuation reports.
