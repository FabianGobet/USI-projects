from unittest import TestCase
from benchmark.gcd import gcd

class Test_gcd(TestCase):
	def test_gcd_1(self):
		y = gcd(50,4)
		self.assertEqual(y,2)

	def test_gcd_2(self):
		y = gcd(6,37)
		self.assertEqual(y,1)

	def test_gcd_3(self):
		y = gcd(41,31)
		self.assertEqual(y,1)

	def test_gcd_4(self):
		y = gcd(76,1)
		self.assertEqual(y,1)

	def test_gcd_5(self):
		y = gcd(79,26)
		self.assertEqual(y,1)

	def test_gcd_6(self):
		y = gcd(48,48)
		self.assertEqual(y,48)

	def test_gcd_7(self):
		y = gcd(16,44)
		self.assertEqual(y,4)

	def test_gcd_8(self):
		y = gcd(4,59)
		self.assertEqual(y,1)

	def test_gcd_9(self):
		y = gcd(91,89)
		self.assertEqual(y,1)

	def test_gcd_10(self):
		y = gcd(89,27)
		self.assertEqual(y,1)

	def test_gcd_11(self):
		y = gcd(1,70)
		self.assertEqual(y,1)

	def test_gcd_12(self):
		y = gcd(19,5)
		self.assertEqual(y,1)

