from unittest import TestCase
from benchmark.common_divisor_count import cd_count

class Test_common_divisor_count(TestCase):
	def test_cd_count_1(self):
		y = cd_count(45,-2)
		self.assertEqual(y,1)

	def test_cd_count_2(self):
		y = cd_count(13,-47)
		self.assertEqual(y,1)

	def test_cd_count_3(self):
		y = cd_count(4,-71)
		self.assertEqual(y,1)

	def test_cd_count_4(self):
		y = cd_count(-31,42)
		self.assertEqual(y,1)

	def test_cd_count_5(self):
		y = cd_count(37,3)
		self.assertEqual(y,1)

	def test_cd_count_6(self):
		y = cd_count(-23,1)
		self.assertEqual(y,1)

	def test_cd_count_7(self):
		y = cd_count(-85,87)
		self.assertEqual(y,1)

	def test_cd_count_8(self):
		y = cd_count(-52,13)
		self.assertEqual(y,2)

	def test_cd_count_9(self):
		y = cd_count(0,57)
		self.assertEqual(y,2)

	def test_cd_count_10(self):
		y = cd_count(64,0)
		self.assertEqual(y,2)

	def test_cd_count_11(self):
		y = cd_count(15,-84)
		self.assertEqual(y,2)

	def test_cd_count_12(self):
		y = cd_count(-1,-26)
		self.assertEqual(y,1)

