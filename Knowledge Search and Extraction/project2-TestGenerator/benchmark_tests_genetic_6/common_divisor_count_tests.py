from unittest import TestCase
from benchmark.common_divisor_count import cd_count

class Test_common_divisor_count(TestCase):
	def test_cd_count_1(self):
		y = cd_count(-8,-33)
		self.assertEqual(y,1)

	def test_cd_count_2(self):
		y = cd_count(-11,-1)
		self.assertEqual(y,1)

	def test_cd_count_3(self):
		y = cd_count(-67,-78)
		self.assertEqual(y,1)

	def test_cd_count_4(self):
		y = cd_count(60,-25)
		self.assertEqual(y,2)

	def test_cd_count_5(self):
		y = cd_count(48,42)
		self.assertEqual(y,4)

	def test_cd_count_6(self):
		y = cd_count(26,-19)
		self.assertEqual(y,1)

	def test_cd_count_7(self):
		y = cd_count(22,13)
		self.assertEqual(y,1)

	def test_cd_count_8(self):
		y = cd_count(32,0)
		self.assertEqual(y,2)

	def test_cd_count_9(self):
		y = cd_count(1,26)
		self.assertEqual(y,1)

	def test_cd_count_10(self):
		y = cd_count(-46,-2)
		self.assertEqual(y,2)

	def test_cd_count_11(self):
		y = cd_count(0,69)
		self.assertEqual(y,2)

	def test_cd_count_12(self):
		y = cd_count(-95,5)
		self.assertEqual(y,2)

