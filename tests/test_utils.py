import unittest
from utils import pad_1d

class TestUtils(unittest.TestCase):
    def test_1d_padding(self):
        input_ = [1,2,3,4,5,6]
        padding = 2
        out_ = pad_1d(input_, padding)
        self.assertEqual(out_, [0,0,1,2,3,4,5,6,0,0])