from unittest import TestCase
from benchmark.gcd import gcd

class Test_gcd(TestCase):
	def test_gcd_1(self):
		y = gcd(24,59)
		self.assertEqual(y,1)

	def test_gcd_2(self):
		y = gcd(87,87)
		self.assertEqual(y,87)

	def test_gcd_3(self):
		y = gcd(43,69)
		self.assertEqual(y,1)

	def test_gcd_4(self):
		y = gcd(1,95)
		self.assertEqual(y,1)

	def test_gcd_5(self):
		y = gcd(6,8)
		self.assertEqual(y,2)

	def test_gcd_6(self):
		y = gcd(49,3)
		self.assertEqual(y,1)

	def test_gcd_7(self):
		y = gcd(10,84)
		self.assertEqual(y,2)

	def test_gcd_8(self):
		y = gcd(73,77)
		self.assertEqual(y,1)

	def test_gcd_9(self):
		y = gcd(18,19)
		self.assertEqual(y,1)

	def test_gcd_10(self):
		y = gcd(19,96)
		self.assertEqual(y,1)

	def test_gcd_11(self):
		y = gcd(93,87)
		self.assertEqual(y,3)

	def test_gcd_12(self):
		y = gcd(33,21)
		self.assertEqual(y,3)

	def test_gcd_13(self):
		y = gcd(42,1)
		self.assertEqual(y,1)

