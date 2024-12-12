from unittest import TestCase
from benchmark.common_divisor_count import cd_count

class Test_common_divisor_count(TestCase):
	def test_cd_count_1(self):
		y = cd_count(51,54)
		self.assertEqual(y,2)

	def test_cd_count_2(self):
		y = cd_count(-10,1)
		self.assertEqual(y,1)

	def test_cd_count_3(self):
		y = cd_count(-1,-12)
		self.assertEqual(y,1)

	def test_cd_count_4(self):
		y = cd_count(-18,-63)
		self.assertEqual(y,3)

	def test_cd_count_5(self):
		y = cd_count(-21,64)
		self.assertEqual(y,1)

	def test_cd_count_6(self):
		y = cd_count(24,0)
		self.assertEqual(y,2)

	def test_cd_count_7(self):
		y = cd_count(-31,-49)
		self.assertEqual(y,1)

	def test_cd_count_8(self):
		y = cd_count(0,-73)
		self.assertEqual(y,2)

