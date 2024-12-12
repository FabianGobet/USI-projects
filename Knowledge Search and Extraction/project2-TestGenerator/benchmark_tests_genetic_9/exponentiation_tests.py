from unittest import TestCase
from benchmark.exponentiation import exponentiation

class Test_exponentiation(TestCase):
	def test_exponentiation_1(self):
		y = exponentiation(-64,40)
		self.assertEqual(y,1766847064778384329583297500742918515827483896875618958121606201292619776)

