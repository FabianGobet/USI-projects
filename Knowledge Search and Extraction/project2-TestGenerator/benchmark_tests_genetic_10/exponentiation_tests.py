from unittest import TestCase
from benchmark.exponentiation import exponentiation

class Test_exponentiation(TestCase):
	def test_exponentiation_1(self):
		y = exponentiation(-18,69)
		self.assertEqual(y,-410963122147519885718066723155689209764147116520545174410510008329199685768993065402368)

