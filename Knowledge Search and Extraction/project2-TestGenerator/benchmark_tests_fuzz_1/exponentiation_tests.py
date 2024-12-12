from unittest import TestCase
from benchmark.exponentiation import exponentiation

class Test_exponentiation(TestCase):
	def test_exponentiation_1(self):
		y = exponentiation(-88,72)
		self.assertEqual(y,100635774849169749924459019322538408697419071408328464152683736409842224350178003597255841038358490590658455045520081479470213392068620845056)

