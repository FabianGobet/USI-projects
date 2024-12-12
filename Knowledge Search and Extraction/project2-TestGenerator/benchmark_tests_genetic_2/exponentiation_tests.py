from unittest import TestCase
from benchmark.exponentiation import exponentiation

class Test_exponentiation(TestCase):
	def test_exponentiation_1(self):
		y = exponentiation(-75,19)
		self.assertEqual(y,-422828258524532429873943328857421875)

