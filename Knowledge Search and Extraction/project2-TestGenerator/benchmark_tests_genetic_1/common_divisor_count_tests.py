from unittest import TestCase
from benchmark.common_divisor_count import cd_count

class Test_common_divisor_count(TestCase):
	def test_cd_count_1(self):
		y = cd_count(-53,1)
		self.assertEqual(y,1)

	def test_cd_count_2(self):
		y = cd_count(-87,79)
		self.assertEqual(y,1)

	def test_cd_count_3(self):
		y = cd_count(34,20)
		self.assertEqual(y,2)

	def test_cd_count_4(self):
		y = cd_count(-95,-38)
		self.assertEqual(y,2)

	def test_cd_count_5(self):
		y = cd_count(29,0)
		self.assertEqual(y,2)

	def test_cd_count_6(self):
		y = cd_count(-50,2)
		self.assertEqual(y,2)

	def test_cd_count_7(self):
		y = cd_count(-54,-43)
		self.assertEqual(y,1)

	def test_cd_count_8(self):
		y = cd_count(0,14)
		self.assertEqual(y,2)

	def test_cd_count_9(self):
		y = cd_count(48,34)
		self.assertEqual(y,2)

	def test_cd_count_10(self):
		y = cd_count(87,4)
		self.assertEqual(y,1)

	def test_cd_count_11(self):
		y = cd_count(35,-5)
		self.assertEqual(y,2)

