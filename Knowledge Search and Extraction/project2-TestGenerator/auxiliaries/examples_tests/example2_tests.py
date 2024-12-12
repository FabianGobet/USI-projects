from unittest import TestCase
from examples.example2 import f

class Test_example2(TestCase):
	def test_f_1(self):
		y = f(2,7)
		self.assertEqual(y,5)

