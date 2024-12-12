from unittest import TestCase
from benchmark.common_divisor_count import cd_count

class Test_common_divisor_count(TestCase):
	def test_cd_count_1(self):
		y = cd_count(0,71)
		self.assertEqual(y,2)

	def test_cd_count_2(self):
		y = cd_count(-45,11)
		self.assertEqual(y,1)

	def test_cd_count_3(self):
		y = cd_count(7,74)
		self.assertEqual(y,1)

	def test_cd_count_4(self):
		y = cd_count(74,0)
		self.assertEqual(y,2)

	def test_cd_count_5(self):
		y = cd_count(84,-75)
		self.assertEqual(y,2)

	def test_cd_count_6(self):
		y = cd_count(26,3)
		self.assertEqual(y,1)

	def test_cd_count_7(self):
		y = cd_count(-1,89)
		self.assertEqual(y,1)

	def test_cd_count_8(self):
		y = cd_count(76,-7)
		self.assertEqual(y,1)

	def test_cd_count_9(self):
		y = cd_count(-40,-30)
		self.assertEqual(y,4)

