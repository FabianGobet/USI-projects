from unittest import TestCase
from benchmark.common_divisor_count import cd_count

class Test_common_divisor_count(TestCase):
	def test_cd_count_1(self):
		y = cd_count(32,-79)
		self.assertEqual(y,1)

	def test_cd_count_2(self):
		y = cd_count(-9,1)
		self.assertEqual(y,1)

	def test_cd_count_3(self):
		y = cd_count(-29,-29)
		self.assertEqual(y,2)

	def test_cd_count_4(self):
		y = cd_count(-63,48)
		self.assertEqual(y,2)

	def test_cd_count_5(self):
		y = cd_count(-61,45)
		self.assertEqual(y,1)

	def test_cd_count_6(self):
		y = cd_count(-8,-82)
		self.assertEqual(y,2)

	def test_cd_count_7(self):
		y = cd_count(4,-44)
		self.assertEqual(y,3)

	def test_cd_count_8(self):
		y = cd_count(0,41)
		self.assertEqual(y,2)

	def test_cd_count_9(self):
		y = cd_count(-11,0)
		self.assertEqual(y,2)

	def test_cd_count_10(self):
		y = cd_count(-16,-84)
		self.assertEqual(y,3)

	def test_cd_count_11(self):
		y = cd_count(-98,2)
		self.assertEqual(y,2)

	def test_cd_count_12(self):
		y = cd_count(-75,24)
		self.assertEqual(y,2)

