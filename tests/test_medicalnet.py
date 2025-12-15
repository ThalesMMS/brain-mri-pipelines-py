import unittest
import torch
import torch.nn as nn
import sys
import os

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from brain_mri.ml.medicalnet_models import resnet10_2d, resnet18_2d, convert_3d_to_2d_weights, ResNet2D
from brain_mri.ml.training_utils import build_medicalnet

class TestMedicalNet(unittest.TestCase):
    def test_resnet2d_forward(self):
        """Testa se o forward pass funciona com dimensões corretas."""
        model = resnet10_2d(pretrained=False)
        x = torch.randn(2, 3, 224, 224)
        out = model(x)
        self.assertEqual(out.shape[0], 2)
        # ResNet2D que implementamos tem num_classes default=1000
        self.assertEqual(out.shape[1], 1000)
    
    def test_build_medicalnet_classification(self):
        """Testa construção do modelo para classificação."""
        model = build_medicalnet(mode='classification', depth=18, pretrained=False)
        self.assertIsInstance(model, ResNet2D)
        # Verifica se última linear tem 2 saídas
        # model.fc é Sequential(Dropout, Linear)
        last_linear = list(model.fc.children())[-1]
        self.assertEqual(last_linear.out_features, 2)
        
    def test_build_medicalnet_regression(self):
        """Testa construção do modelo para regressão."""
        model = build_medicalnet(mode='regression', depth=18, pretrained=False)
        last_linear = list(model.fc.children())[-1]
        self.assertEqual(last_linear.out_features, 1)

    def test_weight_conversion(self):
        """Testa lógica de conversão de pesos 3D -> 2D."""
        chkpt_3d = {
            'conv1.weight': torch.ones(64, 3, 7, 7, 7),
            'bn1.weight': torch.ones(64),
            'layer1.0.conv1.weight': torch.ones(64, 64, 3, 3, 3),
            'module.layer2.0.conv1.weight': torch.ones(128, 64, 3, 3, 3) # Teste remove prefix module.
        }
        converted = convert_3d_to_2d_weights(chkpt_3d)
        
        # Check shapes
        self.assertEqual(converted['conv1.weight'].shape, (64, 3, 7, 7))
        self.assertEqual(converted['layer2.0.conv1.weight'].shape, (128, 64, 3, 3))
        
        # Check values: mean over dim 2 (depth=7) -> permanece 1
        self.assertTrue(torch.allclose(converted['conv1.weight'], torch.ones(64, 3, 7, 7)))

if __name__ == '__main__':
    unittest.main()
