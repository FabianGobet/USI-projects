from unittest import TestCase
from benchmark.exponentiation import exponentiation

class Test_exponentiation(TestCase):
	def test_exponentiation_1(self):
		y = exponentiation(46,82)
		self.assertEqual(y,22189228266636164145324099826330617661844210679250525121193524388574930843895636191247291030009938355947907398726583353958933509194121216)

