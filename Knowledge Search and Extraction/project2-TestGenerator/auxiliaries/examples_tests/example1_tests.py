from unittest import TestCase
from examples.example1 import f

class Test_example1(TestCase):
	def test_f_1(self):
		y = f(2,3)
		self.assertEqual(y,False)

	def test_f_2(self):
		y = f(4,1)
		self.assertEqual(y,True)

