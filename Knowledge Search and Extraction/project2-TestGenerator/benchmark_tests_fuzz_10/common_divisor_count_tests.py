from unittest import TestCase
from benchmark.common_divisor_count import cd_count

class Test_common_divisor_count(TestCase):
	def test_cd_count_1(self):
		y = cd_count(26,-34)
		self.assertEqual(y,2)

	def test_cd_count_2(self):
		y = cd_count(-41,1)
		self.assertEqual(y,1)

	def test_cd_count_3(self):
		y = cd_count(-63,36)
		self.assertEqual(y,3)

	def test_cd_count_4(self):
		y = cd_count(-88,0)
		self.assertEqual(y,2)

	def test_cd_count_5(self):
		y = cd_count(0,-97)
		self.assertEqual(y,2)

	def test_cd_count_6(self):
		y = cd_count(-49,-61)
		self.assertEqual(y,1)

	def test_cd_count_7(self):
		y = cd_count(70,69)
		self.assertEqual(y,1)

	def test_cd_count_8(self):
		y = cd_count(-25,2)
		self.assertEqual(y,1)

	def test_cd_count_9(self):
		y = cd_count(-4,-91)
		self.assertEqual(y,1)

	def test_cd_count_10(self):
		y = cd_count(1,77)
		self.assertEqual(y,1)

	def test_cd_count_11(self):
		y = cd_count(-94,-4)
		self.assertEqual(y,2)

