from unittest import TestCase
from benchmark.common_divisor_count import cd_count

class Test_common_divisor_count(TestCase):
	def test_cd_count_1(self):
		y = cd_count(72,2)
		self.assertEqual(y,2)

	def test_cd_count_2(self):
		y = cd_count(-12,19)
		self.assertEqual(y,1)

	def test_cd_count_3(self):
		y = cd_count(-1,26)
		self.assertEqual(y,1)

	def test_cd_count_4(self):
		y = cd_count(-84,0)
		self.assertEqual(y,2)

	def test_cd_count_5(self):
		y = cd_count(-38,-20)
		self.assertEqual(y,2)

	def test_cd_count_6(self):
		y = cd_count(-19,54)
		self.assertEqual(y,1)

	def test_cd_count_7(self):
		y = cd_count(75,85)
		self.assertEqual(y,2)

	def test_cd_count_8(self):
		y = cd_count(-33,-5)
		self.assertEqual(y,1)

	def test_cd_count_9(self):
		y = cd_count(-77,-97)
		self.assertEqual(y,1)

	def test_cd_count_10(self):
		y = cd_count(49,-1)
		self.assertEqual(y,1)

	def test_cd_count_11(self):
		y = cd_count(-8,4)
		self.assertEqual(y,3)

	def test_cd_count_12(self):
		y = cd_count(0,56)
		self.assertEqual(y,2)

