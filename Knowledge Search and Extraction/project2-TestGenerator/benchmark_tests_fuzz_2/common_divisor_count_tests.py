from unittest import TestCase
from benchmark.common_divisor_count import cd_count

class Test_common_divisor_count(TestCase):
	def test_cd_count_1(self):
		y = cd_count(20,5)
		self.assertEqual(y,2)

	def test_cd_count_2(self):
		y = cd_count(23,4)
		self.assertEqual(y,1)

	def test_cd_count_3(self):
		y = cd_count(0,-35)
		self.assertEqual(y,2)

	def test_cd_count_4(self):
		y = cd_count(-76,0)
		self.assertEqual(y,2)

	def test_cd_count_5(self):
		y = cd_count(-73,1)
		self.assertEqual(y,1)

	def test_cd_count_6(self):
		y = cd_count(1,94)
		self.assertEqual(y,1)

	def test_cd_count_7(self):
		y = cd_count(10,26)
		self.assertEqual(y,2)

	def test_cd_count_8(self):
		y = cd_count(-58,35)
		self.assertEqual(y,1)

	def test_cd_count_9(self):
		y = cd_count(67,-21)
		self.assertEqual(y,1)

