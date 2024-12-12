from unittest import TestCase
from benchmark.exponentiation import exponentiation

class Test_exponentiation(TestCase):
	def test_exponentiation_1(self):
		y = exponentiation(-4,98)
		self.assertEqual(y,100433627766186892221372630771322662657637687111424552206336)

