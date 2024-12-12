from unittest import TestCase
from benchmark.exponentiation import exponentiation

class Test_exponentiation(TestCase):
	def test_exponentiation_1(self):
		y = exponentiation(43,36)
		self.assertEqual(y,63806423321775344604618774305037646254726223988233042609201)

