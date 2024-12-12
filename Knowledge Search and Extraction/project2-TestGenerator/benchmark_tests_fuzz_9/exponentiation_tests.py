from unittest import TestCase
from benchmark.exponentiation import exponentiation

class Test_exponentiation(TestCase):
	def test_exponentiation_1(self):
		y = exponentiation(-42,47)
		self.assertEqual(y,-19620797189109864846683672585216121453209956180728211483132480450702079950848)

