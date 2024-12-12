from unittest import TestCase
from benchmark.common_divisor_count import cd_count

class Test_common_divisor_count(TestCase):
	def test_cd_count_1(self):
		y = cd_count(-15,-34)
		self.assertEqual(y,1)

	def test_cd_count_2(self):
		y = cd_count(80,-86)
		self.assertEqual(y,2)

	def test_cd_count_3(self):
		y = cd_count(97,42)
		self.assertEqual(y,1)

	def test_cd_count_4(self):
		y = cd_count(-28,-3)
		self.assertEqual(y,1)

	def test_cd_count_5(self):
		y = cd_count(0,82)
		self.assertEqual(y,2)

	def test_cd_count_6(self):
		y = cd_count(-84,0)
		self.assertEqual(y,2)

	def test_cd_count_7(self):
		y = cd_count(77,93)
		self.assertEqual(y,1)

	def test_cd_count_8(self):
		y = cd_count(-85,-100)
		self.assertEqual(y,2)

	def test_cd_count_9(self):
		y = cd_count(-6,-26)
		self.assertEqual(y,2)

	def test_cd_count_10(self):
		y = cd_count(85,-17)
		self.assertEqual(y,2)

	def test_cd_count_11(self):
		y = cd_count(-55,-1)
		self.assertEqual(y,1)

	def test_cd_count_12(self):
		y = cd_count(1,-71)
		self.assertEqual(y,1)

