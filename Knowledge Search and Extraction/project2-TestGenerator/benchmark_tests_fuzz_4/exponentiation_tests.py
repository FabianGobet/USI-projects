from unittest import TestCase
from benchmark.exponentiation import exponentiation

class Test_exponentiation(TestCase):
	def test_exponentiation_1(self):
		y = exponentiation(-76,18)
		self.assertEqual(y,7155577026378634231908944079486976)

