from unittest import TestCase
from benchmark.exponentiation import exponentiation

class Test_exponentiation(TestCase):
	def test_exponentiation_1(self):
		y = exponentiation(23,70)
		self.assertEqual(y,209386424652304064049051461204995116953510089281143424135257228513176887924733660903369498848049)

