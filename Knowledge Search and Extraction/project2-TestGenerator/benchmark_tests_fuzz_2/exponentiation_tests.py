from unittest import TestCase
from benchmark.exponentiation import exponentiation

class Test_exponentiation(TestCase):
	def test_exponentiation_1(self):
		y = exponentiation(24,57)
		self.assertEqual(y,4699382308225474252372938283914234062843318963858303351272304591070435081191424)

