from unittest import TestCase
from benchmark.common_divisor_count import cd_count

class Test_common_divisor_count(TestCase):
	def test_cd_count_1(self):
		y = cd_count(63,-6)
		self.assertEqual(y,2)

	def test_cd_count_2(self):
		y = cd_count(26,21)
		self.assertEqual(y,1)

	def test_cd_count_3(self):
		y = cd_count(0,-22)
		self.assertEqual(y,2)

	def test_cd_count_4(self):
		y = cd_count(-1,-37)
		self.assertEqual(y,1)

	def test_cd_count_5(self):
		y = cd_count(-17,58)
		self.assertEqual(y,1)

	def test_cd_count_6(self):
		y = cd_count(81,-2)
		self.assertEqual(y,1)

	def test_cd_count_7(self):
		y = cd_count(-71,-42)
		self.assertEqual(y,1)

	def test_cd_count_8(self):
		y = cd_count(15,-73)
		self.assertEqual(y,1)

	def test_cd_count_9(self):
		y = cd_count(34,0)
		self.assertEqual(y,2)

	def test_cd_count_10(self):
		y = cd_count(-9,28)
		self.assertEqual(y,1)

	def test_cd_count_11(self):
		y = cd_count(-16,-36)
		self.assertEqual(y,3)

	def test_cd_count_12(self):
		y = cd_count(70,-28)
		self.assertEqual(y,4)

