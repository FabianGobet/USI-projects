from unittest import TestCase
from benchmark.check_armstrong import check_armstrong

class Test_check_armstrong(TestCase):
	def test_check_armstrong_1(self):
		y = check_armstrong(0)
		self.assertEqual(y,True)

