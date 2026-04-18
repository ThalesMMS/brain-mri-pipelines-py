import math
import sys
import types
import unittest


def _install_tkinter_stub() -> None:
    if 'tkinter' in sys.modules:
        return
    tk = types.ModuleType('tkinter')
    messagebox = types.ModuleType('tkinter.messagebox')
    messagebox.showinfo = lambda *args, **kwargs: None
    messagebox.showwarning = lambda *args, **kwargs: None
    messagebox.showerror = lambda *args, **kwargs: None
    messagebox.askyesno = lambda *args, **kwargs: True
    tk.messagebox = messagebox
    sys.modules['tkinter'] = tk
    sys.modules['tkinter.messagebox'] = messagebox


_install_tkinter_stub()

import pandas as pd

from brain_mri.ui.segmentation import SegmentationMixin


class _App(SegmentationMixin):
    pass


class SegmentationLongitudinalTests(unittest.TestCase):
    def test_calc_longitudinal_respects_subject_order_and_viability_breaks(self):
        app = _App()
        df = pd.DataFrame(
            [
                {
                    'MRI_ID': 'OAS2_0001_MR2',
                    'Subject_ID': 'OAS2_0001',
                    'viable': True,
                    'ventricle_area': 110.0,
                    'ventricle_perimeter': 55.0,
                },
                {
                    'MRI_ID': 'OAS2_0001_MR1',
                    'Subject_ID': 'OAS2_0001',
                    'viable': True,
                    'ventricle_area': 100.0,
                    'ventricle_perimeter': 50.0,
                },
                {
                    'MRI_ID': 'OAS2_0001_MR3',
                    'Subject_ID': 'OAS2_0001',
                    'viable': False,
                    'ventricle_area': 130.0,
                    'ventricle_perimeter': 60.0,
                },
                {
                    'MRI_ID': 'OAS2_0001_MR4',
                    'Subject_ID': 'OAS2_0001',
                    'viable': True,
                    'ventricle_area': 170.0,
                    'ventricle_perimeter': 80.0,
                },
                {
                    'MRI_ID': 'OAS2_0002_MR1',
                    'Subject_ID': 'OAS2_0002',
                    'viable': True,
                    'ventricle_area': 200.0,
                    'ventricle_perimeter': 100.0,
                },
                {
                    'MRI_ID': 'OAS2_0002_MR2',
                    'Subject_ID': 'OAS2_0002',
                    'viable': True,
                    'ventricle_area': 150.0,
                    'ventricle_perimeter': 90.0,
                },
                {
                    'MRI_ID': 'X_MR1',
                    'Subject_ID': 'X',
                    'viable': True,
                    'ventricle_area': 0.0,
                    'ventricle_perimeter': 10.0,
                },
                {
                    'MRI_ID': 'X_MR2',
                    'Subject_ID': 'X',
                    'viable': True,
                    'ventricle_area': 25.0,
                    'ventricle_perimeter': 15.0,
                },
            ]
        )

        out = app._calc_longitudinal(df.copy())
        indexed = out.set_index('MRI_ID')

        self.assertEqual(indexed.at['OAS2_0001_MR1', 'visit_number'], 1)
        self.assertEqual(indexed.at['OAS2_0001_MR2', 'visit_number'], 2)
        self.assertEqual(indexed.at['OAS2_0001_MR3', 'visit_number'], 3)
        self.assertEqual(indexed.at['OAS2_0001_MR4', 'visit_number'], 4)

        self.assertTrue(math.isnan(indexed.at['OAS2_0001_MR1', 'area_change']))
        self.assertEqual(indexed.at['OAS2_0001_MR2', 'area_change'], 10.0)
        self.assertEqual(indexed.at['OAS2_0001_MR2', 'perimeter_change'], 5.0)
        self.assertEqual(indexed.at['OAS2_0001_MR2', 'area_change_percent'], 10.0)

        self.assertTrue(math.isnan(indexed.at['OAS2_0001_MR3', 'area_change']))
        self.assertTrue(math.isnan(indexed.at['OAS2_0001_MR4', 'area_change']))
        self.assertTrue(math.isnan(indexed.at['OAS2_0001_MR4', 'area_change_percent']))

        self.assertEqual(indexed.at['OAS2_0002_MR2', 'area_change'], -50.0)
        self.assertEqual(indexed.at['OAS2_0002_MR2', 'perimeter_change'], -10.0)
        self.assertEqual(indexed.at['OAS2_0002_MR2', 'area_change_percent'], -25.0)

        self.assertEqual(indexed.at['X_MR2', 'area_change'], 25.0)
        self.assertTrue(math.isnan(indexed.at['X_MR2', 'area_change_percent']))


if __name__ == '__main__':
    unittest.main()
