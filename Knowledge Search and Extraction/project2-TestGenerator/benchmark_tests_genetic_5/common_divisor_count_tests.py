from unittest import TestCase
from benchmark.common_divisor_count import cd_count

class Test_common_divisor_count(TestCase):
	def test_cd_count_1(self):
		y = cd_count(0,23)
		self.assertEqual(y,2)

	def test_cd_count_2(self):
		y = cd_count(25,0)
		self.assertEqual(y,2)

	def test_cd_count_3(self):
		y = cd_count(85,-44)
		self.assertEqual(y,1)

	def test_cd_count_4(self):
		y = cd_count(-19,4)
		self.assertEqual(y,1)

	def test_cd_count_5(self):
		y = cd_count(74,68)
		self.assertEqual(y,2)

	def test_cd_count_6(self):
		y = cd_count(2,-21)
		self.assertEqual(y,1)

	def test_cd_count_7(self):
		y = cd_count(-1,-55)
		self.assertEqual(y,1)

	def test_cd_count_8(self):
		y = cd_count(-7,-49)
		self.assertEqual(y,2)

	def test_cd_count_9(self):
		y = cd_count(-67,36)
		self.assertEqual(y,1)

	def test_cd_count_10(self):
		y = cd_count(6,-91)
		self.assertEqual(y,1)

