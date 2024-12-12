from unittest import TestCase
from benchmark.exponentiation import exponentiation

class Test_exponentiation(TestCase):
	def test_exponentiation_1(self):
		y = exponentiation(99,30)
		self.assertEqual(y,739700373388280422730015092316714942252676262352676444347001)

	def test_exponentiation_2(self):
		y = exponentiation(-46,15)
		self.assertEqual(y,-8737103395697172336050176)

