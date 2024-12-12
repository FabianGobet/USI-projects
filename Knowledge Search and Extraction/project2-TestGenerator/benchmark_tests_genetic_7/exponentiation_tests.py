from unittest import TestCase
from benchmark.exponentiation import exponentiation

class Test_exponentiation(TestCase):
	def test_exponentiation_1(self):
		y = exponentiation(80,83)
		self.assertEqual(y,90462569716653277674664832038037428010367175520031690655826237506182132531200000000000000000000000000000000000000000000000000000000000000000000000000000000000)

