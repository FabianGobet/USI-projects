from unittest import TestCase
from benchmark.gcd import gcd

class Test_gcd(TestCase):
	def test_gcd_1(self):
		y = gcd(98,95)
		self.assertEqual(y,1)

	def test_gcd_2(self):
		y = gcd(22,1)
		self.assertEqual(y,1)

	def test_gcd_3(self):
		y = gcd(25,96)
		self.assertEqual(y,1)

	def test_gcd_4(self):
		y = gcd(58,7)
		self.assertEqual(y,1)

	def test_gcd_5(self):
		y = gcd(3,39)
		self.assertEqual(y,3)

	def test_gcd_6(self):
		y = gcd(1,62)
		self.assertEqual(y,1)

	def test_gcd_7(self):
		y = gcd(93,93)
		self.assertEqual(y,93)

	def test_gcd_8(self):
		y = gcd(24,12)
		self.assertEqual(y,12)

	def test_gcd_9(self):
		y = gcd(82,4)
		self.assertEqual(y,2)

