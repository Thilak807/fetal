import sys
import unittest
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
from backend.atlas.deformable_registration import (
    generate_synthetic_fetal_brain_template,
    ElasticDeformableRegistrar,
)

class TestAtlasRegistration(unittest.TestCase):
    def setUp(self):
        self.template = generate_synthetic_fetal_brain_template(size=(128, 128))
        self.registrar = ElasticDeformableRegistrar(grid_spacing=16, elasticity=1.5)

    def test_atlas_template_properties(self):
        """Verifies canonical fetal atlas shape and intensity bounds."""
        self.assertEqual(self.template.shape, (128, 128))
        self.assertGreaterEqual(float(np.min(self.template)), 0.0)
        self.assertLessEqual(float(np.max(self.template)), 1.0)
        # Verify cranial tissue presence
        self.assertGreater(float(np.mean(self.template)), 0.05)

    def test_elastic_registration(self):
        """Verifies deformable registration aligns slightly rotated scan to atlas."""
        from scipy import ndimage
        moving = ndimage.rotate(self.template, angle=5.0, reshape=False, mode='nearest')
        
        mapped, fields, metrics = self.registrar.register(moving, self.template)
        
        # Verify mapped shape
        self.assertEqual(mapped.shape, (128, 128))
        self.assertFalse(np.isnan(mapped).any())
        self.assertFalse(np.isinf(mapped).any())
        
        # Verify displacement fields
        self.assertIn("u_y", fields)
        self.assertIn("u_x", fields)
        self.assertEqual(fields["u_y"].shape, (128, 128))
        
        # Verify metrics
        self.assertIn("normalized_cross_correlation", metrics)
        self.assertGreater(metrics["normalized_cross_correlation"], 0.7)
        self.assertIn("mean_displacement_pixels", metrics)

if __name__ == "__main__":
    unittest.main()
