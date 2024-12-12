from unittest import TestCase
from benchmark.gcd import gcd

class Test_gcd(TestCase):
	def test_gcd_1(self):
		y = gcd(86,78)
		self.assertEqual(y,2)

	def test_gcd_2(self):
		y = gcd(10,3)
		self.assertEqual(y,1)

	def test_gcd_3(self):
		y = gcd(13,2)
		self.assertEqual(y,1)

	def test_gcd_4(self):
		y = gcd(1,67)
		self.assertEqual(y,1)

	def test_gcd_5(self):
		y = gcd(48,78)
		self.assertEqual(y,6)

	def test_gcd_6(self):
		y = gcd(77,71)
		self.assertEqual(y,1)

	def test_gcd_7(self):
		y = gcd(7,89)
		self.assertEqual(y,1)

	def test_gcd_8(self):
		y = gcd(67,15)
		self.assertEqual(y,1)

	def test_gcd_9(self):
		y = gcd(33,53)
		self.assertEqual(y,1)

	def test_gcd_10(self):
		y = gcd(42,1)
		self.assertEqual(y,1)

	def test_gcd_11(self):
		y = gcd(22,22)
		self.assertEqual(y,22)

	def test_gcd_12(self):
		y = gcd(74,5)
		self.assertEqual(y,1)

	def test_gcd_13(self):
		y = gcd(65,63)
		self.assertEqual(y,1)

	def test_gcd_14(self):
		y = gcd(5,85)
		self.assertEqual(y,5)

	def test_gcd_15(self):
		y = gcd(15,59)
		self.assertEqual(y,1)

	def test_gcd_16(self):
		y = gcd(76,87)
		self.assertEqual(y,1)

	def test_gcd_17(self):
		y = gcd(9,37)
		self.assertEqual(y,1)

	def test_gcd_18(self):
		y = gcd(34,14)
		self.assertEqual(y,2)

	def test_gcd_19(self):
		y = gcd(93,11)
		self.assertEqual(y,1)

	def test_gcd_20(self):
		y = gcd(14,56)
		self.assertEqual(y,14)

