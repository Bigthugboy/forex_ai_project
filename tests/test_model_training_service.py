import unittest
import pandas as pd
import numpy as np
import os
from models.model_training_service import ModelTrainingService
# import pytest  # Remove this line

class TestModelTrainingService(unittest.TestCase):
    def setUp(self):
        self.service = ModelTrainingService(model_dir='models/test_models/', min_samples=10, min_classes=2)
        # Clean up test model dir
        if os.path.exists('models/test_models/'):
            for f in os.listdir('models/test_models/'):
                os.remove(os.path.join('models/test_models/', f))
        else:
            os.makedirs('models/test_models/', exist_ok=True)

    def make_df(self, n=20, imbalance=False, single_class=False):
        np.random.seed(42)
        df = pd.DataFrame({
            'Close': np.linspace(100, 120, n) + np.random.randn(n),
            'Open': np.linspace(99, 119, n) + np.random.randn(n),
            'High': np.linspace(101, 121, n) + np.random.randn(n),
            'Low': np.linspace(98, 118, n) + np.random.randn(n),
            'Volume': np.random.randint(1000, 2000, n),
            'feature1': np.random.randn(n),
            'feature2': np.random.randn(n),
        })
        df['future_return'] = np.random.randn(n) * 0.01
        if single_class:
            df['target'] = 1
        elif imbalance:
            df['target'] = [1] * 2 + [0] * (n - 2)
        else:
            df['target'] = [0, 1] * (n // 2)
        return df

    def test_normal_training(self):
        df = self.make_df(n=20)
        result = self.service.train(df, 'TEST', save=False)
        self.assertIn('model', result)
        self.assertIn('scaler', result)
        self.assertIn('feature_cols', result)
        self.assertGreaterEqual(result['accuracy'], 0)

    def test_upsampling(self):
        df = self.make_df(n=20, imbalance=True)
        result = self.service.train(df, 'TEST', save=False)
        self.assertIn('model', result)
        self.assertIn('scaler', result)
        self.assertIn('feature_cols', result)

    def test_insufficient_data(self):
        # Create a DataFrame of length 5 and assign target directly
        df = pd.DataFrame({
            'Close': np.linspace(100, 105, 5),
            'Open': np.linspace(99, 104, 5),
            'High': np.linspace(101, 106, 5),
            'Low': np.linspace(98, 103, 5),
            'Volume': np.linspace(1000, 2000, 5),
            'future_return': np.linspace(0, 1, 5)
        })
        df['target'] = [0, 1, 0, 1, 0]
        with self.assertRaises(ValueError):
            self.service.train(df, 'TEST', save=False)

    def test_single_class(self):
        n = 20
        df = pd.DataFrame({
            'Close': np.linspace(100, 120, n),
            'Open': np.linspace(99, 119, n),
            'High': np.linspace(101, 121, n),
            'Low': np.linspace(98, 118, n),
            'Volume': np.linspace(1000, 2000, n),
            'future_return': np.full(n, 0.1)
        })
        df = self.service.prepare_target(df)
        df = df.dropna(subset=['target'])
        # Forcibly set all targets to 1 to guarantee single-class
        df['target'] = 1
        self.assertEqual(df['target'].nunique(), 1)
        with self.assertRaises(ValueError):
            self.service.train(df, 'TEST', save=False)

    def test_missing_target(self):
        df = self.make_df(n=20)
        # Remove all features so feature_cols will be empty
        features = ['feature1', 'feature2']
        df = df.drop(features, axis=1)
        with self.assertRaises(ValueError):
            self.service.train(df, 'TEST', required_features=features, save=False)

    def test_feature_selection(self):
        df = self.make_df(n=20)
        features = ['feature1', 'feature2']
        result = self.service.train(df, 'TEST', required_features=features, save=False)
        self.assertEqual(set(result['feature_cols']), set(features))

    def test_model_save_and_load(self):
        df = self.make_df(n=20)
        self.service.train(df, 'TEST', save=True)
        model, scaler, feature_cols = self.service.load('TEST')
        self.assertIsNotNone(model)
        self.assertIsNotNone(scaler)
        self.assertIsInstance(feature_cols, list)

    def test_error_handling_non_dataframe(self):
        with self.assertRaises(Exception):
            self.service.train('not_a_df', 'TEST', save=False)

    def test_train_returns_dict(self):
        df = self.make_df(n=20)
        result = self.service.train(df, 'TEST', save=False)
        assert isinstance(result, dict)
        assert 'model' in result
        assert 'scaler' in result
        assert 'feature_cols' in result
        assert 'accuracy' in result

    def test_train_with_required_features(self):
        df = self.make_df(n=20)
        features = ['feature1', 'feature2']
        result = self.service.train(df, 'TEST', required_features=features, save=False)
        assert set(result['feature_cols']) == set(features)

    def test_train_upsampling(self):
        df = self.make_df(n=20, imbalance=True)
        self.service.train(df, 'TEST', save=False)
        # No assertion needed, just ensure no error

    def test_train_with_save(self):
        df = self.make_df(n=20)
        self.service.train(df, 'TEST', save=True)
        # No assertion needed, just ensure no error

    def test_train_with_invalid_df(self):
        with self.assertRaises(Exception):
            self.service.train('not_a_df', 'TEST', save=False)

if __name__ == '__main__':
    unittest.main() 