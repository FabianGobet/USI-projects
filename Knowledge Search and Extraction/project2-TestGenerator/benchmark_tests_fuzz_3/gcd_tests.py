from unittest import TestCase
from benchmark.gcd import gcd

class Test_gcd(TestCase):
	def test_gcd_1(self):
		y = gcd(25,32)
		self.assertEqual(y,1)

	def test_gcd_2(self):
		y = gcd(76,65)
		self.assertEqual(y,1)

	def test_gcd_3(self):
		y = gcd(1,8)
		self.assertEqual(y,1)

	def test_gcd_4(self):
		y = gcd(25,25)
		self.assertEqual(y,25)

	def test_gcd_5(self):
		y = gcd(25,2)
		self.assertEqual(y,1)

	def test_gcd_6(self):
		y = gcd(2,98)
		self.assertEqual(y,2)

	def test_gcd_7(self):
		y = gcd(5,1)
		self.assertEqual(y,1)

	def test_gcd_8(self):
		y = gcd(3,54)
		self.assertEqual(y,3)

	def test_gcd_9(self):
		y = gcd(49,21)
		self.assertEqual(y,7)

	def test_gcd_10(self):
		y = gcd(31,3)
		self.assertEqual(y,1)

	def test_gcd_11(self):
		y = gcd(7,85)
		self.assertEqual(y,1)

	def test_gcd_12(self):
		y = gcd(54,68)
		self.assertEqual(y,2)

