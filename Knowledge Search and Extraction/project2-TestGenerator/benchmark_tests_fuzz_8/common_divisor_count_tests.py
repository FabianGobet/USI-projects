from unittest import TestCase
from benchmark.common_divisor_count import cd_count

class Test_common_divisor_count(TestCase):
	def test_cd_count_1(self):
		y = cd_count(0,23)
		self.assertEqual(y,2)

	def test_cd_count_2(self):
		y = cd_count(41,27)
		self.assertEqual(y,1)

	def test_cd_count_3(self):
		y = cd_count(-94,-2)
		self.assertEqual(y,2)

	def test_cd_count_4(self):
		y = cd_count(-31,0)
		self.assertEqual(y,2)

	def test_cd_count_5(self):
		y = cd_count(77,-2)
		self.assertEqual(y,1)

	def test_cd_count_6(self):
		y = cd_count(19,64)
		self.assertEqual(y,1)

	def test_cd_count_7(self):
		y = cd_count(70,14)
		self.assertEqual(y,4)

