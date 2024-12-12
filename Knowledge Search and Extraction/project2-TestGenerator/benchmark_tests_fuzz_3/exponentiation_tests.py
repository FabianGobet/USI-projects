from unittest import TestCase
from benchmark.exponentiation import exponentiation

class Test_exponentiation(TestCase):
	def test_exponentiation_1(self):
		y = exponentiation(45,30)
		self.assertEqual(y,39479842665806602234295041487552225589752197265625)

