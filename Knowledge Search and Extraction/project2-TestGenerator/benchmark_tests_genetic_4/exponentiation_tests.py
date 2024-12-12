from unittest import TestCase
from benchmark.exponentiation import exponentiation

class Test_exponentiation(TestCase):
	def test_exponentiation_1(self):
		y = exponentiation(30,3)
		self.assertEqual(y,27000)

	def test_exponentiation_2(self):
		y = exponentiation(95,40)
		self.assertEqual(y,12851215656510336328601947580441059706644450107625752934836782515048980712890625)

