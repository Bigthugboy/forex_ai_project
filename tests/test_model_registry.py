import unittest
import os
from pymongo import MongoClient
import random
import string
from models.model_registry import ModelRegistry
import tempfile
import joblib
from unittest.mock import patch
import sys

class TestModelRegistry(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Use a test database/collection
        cls.test_db = "forex_ai_test"
        cls.test_collection = "model_registry_test"
        cls.registry = ModelRegistry(db_name=cls.test_db, collection_name=cls.test_collection)
        # Clean up before
        cls.registry.collection.delete_many({})

    @classmethod
    def tearDownClass(cls):
        # Clean up after
        cls.registry.collection.delete_many({})
        cls.registry.close()
        # Optionally drop the test database
        client = MongoClient(cls.registry.mongo_uri)
        client.drop_database(cls.test_db)

    def random_version(self):
        return ''.join(random.choices(string.ascii_lowercase + string.digits, k=8))

    def test_register_and_get_model(self):
        version = self.random_version()
        model_type = "test_model"
        features = ["f1", "f2"]
        scaler_path = "/tmp/scaler.pkl"
        model_path = "/tmp/model.pkl"
        metrics = {"accuracy": 0.99}
        data_range = "2020-01-01 to 2020-12-31"
        notes = "unit test"
        retrain_source = "test"
        status = "active"
        inserted_id = self.registry.register_model(
            model_type, version, features, scaler_path, model_path, metrics, data_range, notes, retrain_source, status
        )
        self.assertIsNotNone(inserted_id)
        # Fetch by version
        doc = self.registry.get_model_by_version(model_type, version)
        self.assertIsNotNone(doc)
        if doc is not None:
            self.assertEqual(doc["version"], version)
        # Fetch latest
        latest = self.registry.get_latest_model(model_type)
        self.assertIsNotNone(latest)
        if latest is not None:
            self.assertEqual(latest["version"], version)

    def test_list_models(self):
        model_type = "list_test_model"
        versions = [self.random_version() for _ in range(3)]
        for v in versions:
            self.registry.register_model(
                model_type, v, ["f1"], "/tmp/scaler.pkl", "/tmp/model.pkl", {"accuracy": 0.9}, "range", None, None, "active"
            )
        models = self.registry.list_models(model_type)
        self.assertGreaterEqual(len(models), 3)
        # Only active
        active_models = self.registry.list_models(model_type, only_active=True)
        for m in active_models:
            self.assertEqual(m["status"], "active")

    def test_deprecate_model(self):
        model_type = "deprecate_test_model"
        version = self.random_version()
        self.registry.register_model(
            model_type, version, ["f1"], "/tmp/scaler.pkl", "/tmp/model.pkl", {"accuracy": 0.8}, "range", None, None, "active"
        )
        result = self.registry.deprecate_model(model_type, version)
        self.assertTrue(result)
        doc = self.registry.get_model_by_version(model_type, version)
        self.assertIsNotNone(doc)
        if doc is not None:
            self.assertEqual(doc["status"], "deprecated")

class TestRegistryAwareLoading(unittest.TestCase):
    def setUp(self):
        self.registry = ModelRegistry(db_name="forex_ai_test", collection_name="model_registry_test")
        self.registry.collection.delete_many({})
        # Create a dummy model and scaler
        self.model = {'dummy': 'model'}
        self.scaler = {'dummy': 'scaler'}
        self.features = ['f1', 'f2']
        self.temp_model_file = tempfile.NamedTemporaryFile(delete=False)
        self.temp_scaler_file = tempfile.NamedTemporaryFile(delete=False)
        joblib.dump(self.model, self.temp_model_file.name)
        joblib.dump(self.scaler, self.temp_scaler_file.name)
        self.version = 'testver123'
        self.model_type = 'signal_model_TEST'
        self.registry.register_model(
            self.model_type, self.version, self.features, self.temp_scaler_file.name, self.temp_model_file.name,
            {'accuracy': 1.0}, 'range', None, None, 'active'
        )

    def tearDown(self):
        self.registry.collection.delete_many({})
        self.registry.close()
        os.unlink(self.temp_model_file.name)
        os.unlink(self.temp_scaler_file.name)

    def test_registry_aware_load(self):
        # Patch ModelRegistry in load_model_registry_aware to use the test db/collection
        from models import predict
        with patch.object(predict, 'ModelRegistry', lambda: ModelRegistry(db_name="forex_ai_test", collection_name="model_registry_test")):
            model, scaler, features = predict.load_model_registry_aware('TEST')
            self.assertEqual(model, self.model)
            self.assertEqual(scaler, self.scaler)
            self.assertEqual(features, self.features)

class TestRegistryAwareRetrain(unittest.TestCase):
    def setUp(self):
        self.registry = ModelRegistry(db_name="forex_ai_test", collection_name="model_registry_test")
        self.registry.collection.delete_many({})
        self.pair = 'TEST'
        self.version = 'abc123'
        self.registry.register_model(
            f'signal_model_{self.pair}', self.version, ['f1'], '/tmp/scaler.pkl', '/tmp/model.pkl', {'accuracy': 0.9}, 'range', None, None, 'active'
        )

    def tearDown(self):
        self.registry.collection.delete_many({})
        self.registry.close()

    def test_should_retrain_pair(self):
        # Patch ModelRegistry in ContinuousLearning to use the test db/collection
        from models import continuous_learning
        with patch.object(continuous_learning, 'ModelRegistry', lambda: ModelRegistry(db_name="forex_ai_test", collection_name="model_registry_test")):
            cl = continuous_learning.ContinuousLearning()
            # Should not retrain if version matches
            self.assertFalse(cl.should_retrain_pair(self.pair, self.version))
            # Should retrain if version does not match
            self.assertTrue(cl.should_retrain_pair(self.pair, 'differentver'))

if __name__ == "__main__":
    unittest.main() 