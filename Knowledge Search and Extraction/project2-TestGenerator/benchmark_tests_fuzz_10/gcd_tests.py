from unittest import TestCase
from benchmark.gcd import gcd

class Test_gcd(TestCase):
	def test_gcd_1(self):
		y = gcd(1,67)
		self.assertEqual(y,1)

	def test_gcd_2(self):
		y = gcd(27,2)
		self.assertEqual(y,1)

	def test_gcd_3(self):
		y = gcd(19,59)
		self.assertEqual(y,1)

	def test_gcd_4(self):
		y = gcd(2,27)
		self.assertEqual(y,1)

	def test_gcd_5(self):
		y = gcd(12,1)
		self.assertEqual(y,1)

	def test_gcd_6(self):
		y = gcd(31,4)
		self.assertEqual(y,1)

	def test_gcd_7(self):
		y = gcd(29,26)
		self.assertEqual(y,1)

	def test_gcd_8(self):
		y = gcd(43,43)
		self.assertEqual(y,43)

	def test_gcd_9(self):
		y = gcd(13,74)
		self.assertEqual(y,1)

	def test_gcd_10(self):
		y = gcd(29,30)
		self.assertEqual(y,1)

