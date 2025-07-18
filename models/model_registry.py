import os
from datetime import datetime
from typing import Optional, Dict, Any, List
from pymongo import MongoClient, ASCENDING, DESCENDING
from pymongo.collection import Collection

class ModelRegistry:
    """
    Handles model version tracking and analytics in MongoDB.
    """
    def __init__(self, mongo_uri: Optional[str] = None, db_name: str = "forex_ai", collection_name: str = "model_registry"):
        self.mongo_uri = mongo_uri or os.getenv("MONGODB_URI", "mongodb://localhost:27017/")
        self.client = MongoClient(self.mongo_uri)
        self.db = self.client[db_name]
        self.collection: Collection = self.db[collection_name]
        self.collection.create_index([("model_type", ASCENDING), ("version", DESCENDING)], unique=True)

    def register_model(self, model_type: str, version: str, features: List[str], scaler_path: str, model_path: str,
                      metrics: Dict[str, Any], data_range: str, notes: Optional[str] = None,
                      retrain_source: Optional[str] = None, status: str = "active") -> str:
        """
        Register a new model version with metadata.
        """
        doc = {
            "model_type": model_type,
            "version": version,
            "features": features,
            "scaler_path": scaler_path,
            "model_path": model_path,
            "metrics": metrics,
            "data_range": data_range,
            "notes": notes,
            "retrain_source": retrain_source,
            "status": status,
            "created_at": datetime.utcnow(),
        }
        try:
            result = self.collection.insert_one(doc)
            return str(result.inserted_id)
        except Exception as e:
            raise RuntimeError(f"Failed to register model: {e}")

    def get_latest_model(self, model_type: str, only_active: bool = True) -> Optional[Dict[str, Any]]:
        """
        Fetch the latest (by version) model for a given type.
        """
        query = {"model_type": model_type}
        if only_active:
            query["status"] = "active"
        return self.collection.find_one(query, sort=[("created_at", DESCENDING)])

    def get_model_by_version(self, model_type: str, version: str) -> Optional[Dict[str, Any]]:
        """
        Fetch a specific model version.
        """
        return self.collection.find_one({"model_type": model_type, "version": version})

    def list_models(self, model_type: Optional[str] = None, only_active: bool = False) -> List[Dict[str, Any]]:
        """
        List all models, optionally filtered by type and status.
        """
        query = {}
        if model_type:
            query["model_type"] = model_type
        if only_active:
            query["status"] = "active"
        return list(self.collection.find(query).sort("created_at", DESCENDING))

    def deprecate_model(self, model_type: str, version: str) -> bool:
        """
        Mark a model version as deprecated.
        """
        result = self.collection.update_one(
            {"model_type": model_type, "version": version},
            {"$set": {"status": "deprecated", "deprecated_at": datetime.utcnow()}}
        )
        return result.modified_count > 0

    def close(self):
        self.client.close()

# Unit tests for ModelRegistry (to be placed in tests/test_model_registry.py)
if __name__ == "__main__":
    import unittest
    import os
    from pymongo import MongoClient
    import random
    import string

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

    unittest.main() 