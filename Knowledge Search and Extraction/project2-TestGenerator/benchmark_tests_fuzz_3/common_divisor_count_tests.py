from unittest import TestCase
from benchmark.common_divisor_count import cd_count

class Test_common_divisor_count(TestCase):
	def test_cd_count_1(self):
		y = cd_count(-91,-63)
		self.assertEqual(y,2)

	def test_cd_count_2(self):
		y = cd_count(-99,5)
		self.assertEqual(y,1)

	def test_cd_count_3(self):
		y = cd_count(37,9)
		self.assertEqual(y,1)

	def test_cd_count_4(self):
		y = cd_count(-2,97)
		self.assertEqual(y,1)

	def test_cd_count_5(self):
		y = cd_count(24,-3)
		self.assertEqual(y,2)

	def test_cd_count_6(self):
		y = cd_count(-70,-2)
		self.assertEqual(y,2)

	def test_cd_count_7(self):
		y = cd_count(3,0)
		self.assertEqual(y,2)

	def test_cd_count_8(self):
		y = cd_count(-92,-84)
		self.assertEqual(y,3)

	def test_cd_count_9(self):
		y = cd_count(-1,-88)
		self.assertEqual(y,1)

	def test_cd_count_10(self):
		y = cd_count(-77,-96)
		self.assertEqual(y,1)

	def test_cd_count_11(self):
		y = cd_count(-2,-4)
		self.assertEqual(y,2)

	def test_cd_count_12(self):
		y = cd_count(-35,31)
		self.assertEqual(y,1)

	def test_cd_count_13(self):
		y = cd_count(0,-18)
		self.assertEqual(y,2)

