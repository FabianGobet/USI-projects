from unittest import TestCase
from benchmark.common_divisor_count import cd_count

class Test_common_divisor_count(TestCase):
	def test_cd_count_1(self):
		y = cd_count(-36,-1)
		self.assertEqual(y,1)

	def test_cd_count_2(self):
		y = cd_count(-8,-3)
		self.assertEqual(y,1)

	def test_cd_count_3(self):
		y = cd_count(0,26)
		self.assertEqual(y,2)

	def test_cd_count_4(self):
		y = cd_count(7,-90)
		self.assertEqual(y,1)

	def test_cd_count_5(self):
		y = cd_count(68,0)
		self.assertEqual(y,2)

	def test_cd_count_6(self):
		y = cd_count(44,18)
		self.assertEqual(y,2)

	def test_cd_count_7(self):
		y = cd_count(-4,-17)
		self.assertEqual(y,1)

	def test_cd_count_8(self):
		y = cd_count(10,-100)
		self.assertEqual(y,4)

	def test_cd_count_9(self):
		y = cd_count(-2,99)
		self.assertEqual(y,1)

