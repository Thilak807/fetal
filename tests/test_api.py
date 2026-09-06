import sys
import unittest
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from backend.app import create_app
from backend.database.db import db
from backend.database.models import Patient

class TestAPI(unittest.TestCase):
    def setUp(self):
        self.app = create_app()
        self.app.config["TESTING"] = True
        self.app.config["SQLALCHEMY_DATABASE_URI"] = "sqlite:///:memory:"
        self.client = self.app.test_client()

        with self.app.app_context():
            db.create_all()

    def tearDown(self):
        with self.app.app_context():
            db.session.remove()
            db.drop_all()

    def test_get_root(self):
        """Tests that index endpoint serves HTML workstation."""
        res = self.client.get("/")
        self.assertEqual(res.status_code, 200)
        self.assertIn(b"Multi-Modal Fetal Risk Assessment", res.data)

    def test_create_and_list_patient(self):
        """Tests patient registration and listing."""
        payload = {
            "patient_id": "PAT-TEST-001",
            "name": "Sarah Connor",
            "age": 28,
            "gestational_week": 34.0,
            "notes": "Test case registration"
        }
        post_res = self.client.post("/api/patient", json=payload)
        self.assertEqual(post_res.status_code, 201)
        data = post_res.get_json()
        self.assertEqual(data["patient"]["name"], "Sarah Connor")

        list_res = self.client.get("/api/patients")
        self.assertEqual(list_res.status_code, 200)
        patients = list_res.get_json()
        self.assertGreaterEqual(len(patients), 1)

    def test_evaluation_endpoint(self):
        """Tests evaluation endpoint returns valid JSON structure."""
        res = self.client.get("/api/evaluation")
        self.assertEqual(res.status_code, 200)
        data = res.get_json()
        self.assertTrue("status" in data or "accuracy" in data)

if __name__ == "__main__":
    unittest.main()
