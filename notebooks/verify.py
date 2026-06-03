import sys
import unittest
import json
import pickle
import os

# Append project root to path to ensure app import works
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from app import app, load_ml_artifacts

class TestHousePriceValuation(unittest.TestCase):
    
    @classmethod
    def setUpClass(cls):
        # Force load artifacts
        load_ml_artifacts()
        cls.client = app.test_client()
        cls.client.testing = True
        
        # Ensure testuser is deleted so tests can run repeatedly
        from app import db, User
        with app.app_context():
            try:
                User.query.filter_by(username='testuser').delete()
                db.session.commit()
            except Exception as e:
                print(f"Test setup clean skipped: {str(e)}")

    def test_home_route(self):
        """Test home index page rendering"""
        response = self.client.get('/')
        self.assertEqual(response.status_code, 200)
        self.assertIn(b'AURA', response.data)
        self.assertIn(b'Valuation', response.data)

    def test_about_route(self):
        """Test about page documentation rendering"""
        response = self.client.get('/about')
        self.assertEqual(response.status_code, 200)
        self.assertIn(b'Trained Models Comparison', response.data)
        self.assertIn(b'OneHotEncoder', response.data)

    def test_ajax_prediction_success(self):
        """Test AJAX prediction request returns valid price estimation"""
        payload = {
            'Area_SqFt': 2800,
            'Bedrooms': 3,
            'Bathrooms': 2.5,
            'Location': 'Uptown',
            'Year_Built': 2012,
            'Parking': 'Yes',
            'Floors': 2,
            'Amenities': 'Standard'
        }
        
        response = self.client.post('/predict', 
                                    data=json.dumps(payload),
                                    content_type='application/json',
                                    headers={'X-Requested-With': 'XMLHttpRequest'})
        
        self.assertEqual(response.status_code, 200)
        data = json.loads(response.data.decode('utf-8'))
        
        self.assertTrue(data['success'])
        self.assertIn('price', data)
        self.assertIn('formatted_price', data)
        self.assertTrue(data['price'] > 4000000.0)
        self.assertEqual(data['inputs']['Location'], 'Uptown')
        
        # Verify Advanced PropTech keys
        self.assertIn('multi_model_prices', data)
        self.assertIn('Gradient Boosting', data['multi_model_prices'])
        self.assertIn('Linear Regression', data['multi_model_prices'])
        self.assertIn('Random Forest', data['multi_model_prices'])
        
        self.assertIn('explainability_contributions', data)
        self.assertTrue(len(data['explainability_contributions']) > 0)
        self.assertIn('feature', data['explainability_contributions'][0])
        self.assertIn('weight', data['explainability_contributions'][0])
        
        self.assertIn('recommended_matches', data)
        self.assertTrue(len(data['recommended_matches']) > 0)
        self.assertIn('formatted_price', data['recommended_matches'][0])
        
        self.assertIn('historical_appreciation_trends', data)
        self.assertTrue(len(data['historical_appreciation_trends']) > 0)
        self.assertIn('year', data['historical_appreciation_trends'][0])
        self.assertIn('price', data['historical_appreciation_trends'][0])
        
        print(f"Verified AJAX prediction response price: {data['formatted_price']}")

    def test_standard_form_prediction_success(self):
        """Test traditional Form POST prediction request returns rendered success card"""
        form_payload = {
            'Area_SqFt': '3400',
            'Bedrooms': '4',
            'Bathrooms': '3.0',
            'Location': 'Downtown',
            'Year_Built': '2018',
            'Parking': 'No',
            'Floors': '3',
            'Amenities': 'Luxury'
        }
        
        response = self.client.post('/predict', data=form_payload)
        self.assertEqual(response.status_code, 200)
        self.assertIn(b'Estimated Home Market Value', response.data)
        self.assertIn(b'Downtown', response.data)
        self.assertIn(b'Luxury', response.data)

    def test_prediction_input_validation_failure(self):
        """Test input validations check ranges and formats correctly"""
        # Bad payload (extremely massive Area and negative Bedrooms)
        bad_payload = {
            'Area_SqFt': 80000,
            'Bedrooms': -2,
            'Bathrooms': 2.5,
            'Location': 'Downtown',
            'Year_Built': 2012,
            'Parking': 'Yes',
            'Floors': 2,
            'Amenities': 'Standard'
        }
        
        response = self.client.post('/predict', 
                                    data=json.dumps(bad_payload),
                                    content_type='application/json',
                                    headers={'X-Requested-With': 'XMLHttpRequest'})
        
        self.assertEqual(response.status_code, 400)
        data = json.loads(response.data.decode('utf-8'))
        self.assertFalse(data['success'])
        self.assertTrue(len(data['errors']) >= 2)
        print(f"Verified validation failures check correctly! Errors: {data['errors']}")

    def test_user_registration_and_login_flow(self):
        """Test complete user registration, login, and authorization validation"""
        # 1. Register a new user
        reg_payload = {
            'username': 'testuser',
            'email': 'testuser@aurapred.com',
            'password': 'testpassword',
            'confirm_password': 'testpassword'
        }
        response = self.client.post('/register', data=reg_payload)
        self.assertEqual(response.status_code, 302) # redirect to login
        
        # 2. Login with registered credentials
        login_payload = {
            'username_or_email': 'testuser',
            'password': 'testpassword'
        }
        response = self.client.post('/login', data=login_payload, follow_redirects=True)
        self.assertEqual(response.status_code, 200)
        self.assertIn(b'testuser', response.data) # Check username appears on dashboard
        
        # 3. Access User Dashboard while logged in
        response = self.client.get('/dashboard')
        self.assertEqual(response.status_code, 200)
        self.assertIn(b'testuser@aurapred.com', response.data)
        
        # 4. Try to access Admin Dashboard (should fail as normal user)
        response = self.client.get('/admin', follow_redirects=True)
        self.assertIn(b'Unauthorized access', response.data) # Flash alert error
        
        # 5. Logout
        response = self.client.get('/logout', follow_redirects=True)
        self.assertEqual(response.status_code, 200)
        self.assertIn(b'logged out successfully', response.data)

    def test_admin_seeding_and_access(self):
        """Test default admin user is seeded and can successfully access admin panel"""
        # 1. Login as default admin
        admin_login = {
            'username_or_email': 'admin',
            'password': 'adminpassword'
        }
        response = self.client.post('/login', data=admin_login, follow_redirects=True)
        self.assertEqual(response.status_code, 200)
        self.assertIn(b'admin', response.data)
        
        # 2. Open Admin Dashboard successfully
        response = self.client.get('/admin')
        self.assertEqual(response.status_code, 200)
        self.assertIn(b'Administrative Board', response.data)
        self.assertIn(b'Platform Users', response.data)
        
        # 3. Logout
        self.client.get('/logout')

if __name__ == '__main__':
    unittest.main()
