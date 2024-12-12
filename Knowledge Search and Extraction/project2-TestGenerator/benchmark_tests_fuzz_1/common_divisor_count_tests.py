from unittest import TestCase
from benchmark.common_divisor_count import cd_count

class Test_common_divisor_count(TestCase):
	def test_cd_count_1(self):
		y = cd_count(0,1)
		self.assertEqual(y,2)

	def test_cd_count_2(self):
		y = cd_count(58,56)
		self.assertEqual(y,2)

	def test_cd_count_3(self):
		y = cd_count(-1,-60)
		self.assertEqual(y,1)

	def test_cd_count_4(self):
		y = cd_count(4,48)
		self.assertEqual(y,3)

	def test_cd_count_5(self):
		y = cd_count(12,0)
		self.assertEqual(y,2)

	def test_cd_count_6(self):
		y = cd_count(34,2)
		self.assertEqual(y,2)

	def test_cd_count_7(self):
		y = cd_count(-45,47)
		self.assertEqual(y,1)

	def test_cd_count_8(self):
		y = cd_count(-25,-11)
		self.assertEqual(y,1)

	def test_cd_count_9(self):
		y = cd_count(-21,-63)
		self.assertEqual(y,4)

