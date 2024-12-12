from unittest import TestCase
from benchmark.gcd import gcd

class Test_gcd(TestCase):
	def test_gcd_1(self):
		y = gcd(2,42)
		self.assertEqual(y,2)

	def test_gcd_2(self):
		y = gcd(33,13)
		self.assertEqual(y,1)

	def test_gcd_3(self):
		y = gcd(87,3)
		self.assertEqual(y,3)

	def test_gcd_4(self):
		y = gcd(68,61)
		self.assertEqual(y,1)

	def test_gcd_5(self):
		y = gcd(1,97)
		self.assertEqual(y,1)

	def test_gcd_6(self):
		y = gcd(100,15)
		self.assertEqual(y,5)

	def test_gcd_7(self):
		y = gcd(21,21)
		self.assertEqual(y,21)

	def test_gcd_8(self):
		y = gcd(3,3)
		self.assertEqual(y,3)

	def test_gcd_9(self):
		y = gcd(63,62)
		self.assertEqual(y,1)

	def test_gcd_10(self):
		y = gcd(58,79)
		self.assertEqual(y,1)

	def test_gcd_11(self):
		y = gcd(8,50)
		self.assertEqual(y,2)

	def test_gcd_12(self):
		y = gcd(21,1)
		self.assertEqual(y,1)

