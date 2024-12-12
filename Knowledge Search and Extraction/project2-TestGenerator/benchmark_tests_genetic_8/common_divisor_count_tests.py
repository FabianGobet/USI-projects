from unittest import TestCase
from benchmark.common_divisor_count import cd_count

class Test_common_divisor_count(TestCase):
	def test_cd_count_1(self):
		y = cd_count(-29,0)
		self.assertEqual(y,2)

	def test_cd_count_2(self):
		y = cd_count(0,-3)
		self.assertEqual(y,2)

	def test_cd_count_3(self):
		y = cd_count(1,-37)
		self.assertEqual(y,1)

	def test_cd_count_4(self):
		y = cd_count(-6,7)
		self.assertEqual(y,1)

	def test_cd_count_5(self):
		y = cd_count(-15,-30)
		self.assertEqual(y,4)

	def test_cd_count_6(self):
		y = cd_count(-99,6)
		self.assertEqual(y,2)

	def test_cd_count_7(self):
		y = cd_count(29,-85)
		self.assertEqual(y,1)

