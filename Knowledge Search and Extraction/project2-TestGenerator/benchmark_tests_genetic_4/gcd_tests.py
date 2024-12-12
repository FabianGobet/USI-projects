from unittest import TestCase
from benchmark.gcd import gcd

class Test_gcd(TestCase):
	def test_gcd_1(self):
		y = gcd(11,1)
		self.assertEqual(y,1)

	def test_gcd_2(self):
		y = gcd(48,27)
		self.assertEqual(y,3)

	def test_gcd_3(self):
		y = gcd(35,7)
		self.assertEqual(y,7)

	def test_gcd_4(self):
		y = gcd(52,84)
		self.assertEqual(y,4)

	def test_gcd_5(self):
		y = gcd(9,96)
		self.assertEqual(y,3)

	def test_gcd_6(self):
		y = gcd(2,90)
		self.assertEqual(y,2)

	def test_gcd_7(self):
		y = gcd(1,48)
		self.assertEqual(y,1)

	def test_gcd_8(self):
		y = gcd(24,24)
		self.assertEqual(y,24)

