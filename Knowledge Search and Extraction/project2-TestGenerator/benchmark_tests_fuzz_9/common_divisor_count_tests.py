from unittest import TestCase
from benchmark.common_divisor_count import cd_count

class Test_common_divisor_count(TestCase):
	def test_cd_count_1(self):
		y = cd_count(-46,29)
		self.assertEqual(y,1)

	def test_cd_count_2(self):
		y = cd_count(14,-15)
		self.assertEqual(y,1)

	def test_cd_count_3(self):
		y = cd_count(18,-54)
		self.assertEqual(y,6)

	def test_cd_count_4(self):
		y = cd_count(-21,4)
		self.assertEqual(y,1)

	def test_cd_count_5(self):
		y = cd_count(-6,43)
		self.assertEqual(y,1)

	def test_cd_count_6(self):
		y = cd_count(89,-13)
		self.assertEqual(y,1)

	def test_cd_count_7(self):
		y = cd_count(-18,-9)
		self.assertEqual(y,3)

	def test_cd_count_8(self):
		y = cd_count(-98,0)
		self.assertEqual(y,2)

	def test_cd_count_9(self):
		y = cd_count(-39,-6)
		self.assertEqual(y,2)

	def test_cd_count_10(self):
		y = cd_count(-12,5)
		self.assertEqual(y,1)

	def test_cd_count_11(self):
		y = cd_count(31,-42)
		self.assertEqual(y,1)

	def test_cd_count_12(self):
		y = cd_count(0,-82)
		self.assertEqual(y,2)

	def test_cd_count_13(self):
		y = cd_count(99,-1)
		self.assertEqual(y,1)

