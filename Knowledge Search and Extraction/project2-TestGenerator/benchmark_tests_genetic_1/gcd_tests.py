from unittest import TestCase
from benchmark.gcd import gcd

class Test_gcd(TestCase):
	def test_gcd_1(self):
		y = gcd(6,5)
		self.assertEqual(y,1)

	def test_gcd_2(self):
		y = gcd(19,52)
		self.assertEqual(y,1)

	def test_gcd_3(self):
		y = gcd(36,1)
		self.assertEqual(y,1)

	def test_gcd_4(self):
		y = gcd(93,10)
		self.assertEqual(y,1)

	def test_gcd_5(self):
		y = gcd(1,78)
		self.assertEqual(y,1)

	def test_gcd_6(self):
		y = gcd(59,47)
		self.assertEqual(y,1)

	def test_gcd_7(self):
		y = gcd(51,4)
		self.assertEqual(y,1)

	def test_gcd_8(self):
		y = gcd(91,2)
		self.assertEqual(y,1)

	def test_gcd_9(self):
		y = gcd(2,60)
		self.assertEqual(y,2)

	def test_gcd_10(self):
		y = gcd(59,62)
		self.assertEqual(y,1)

