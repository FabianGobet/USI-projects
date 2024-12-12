from unittest import TestCase
from benchmark.common_divisor_count import cd_count

class Test_common_divisor_count(TestCase):
	def test_cd_count_1(self):
		y = cd_count(-46,0)
		self.assertEqual(y,2)

	def test_cd_count_2(self):
		y = cd_count(77,-98)
		self.assertEqual(y,2)

	def test_cd_count_3(self):
		y = cd_count(1,-28)
		self.assertEqual(y,1)

	def test_cd_count_4(self):
		y = cd_count(0,57)
		self.assertEqual(y,2)

	def test_cd_count_5(self):
		y = cd_count(4,-40)
		self.assertEqual(y,3)

	def test_cd_count_6(self):
		y = cd_count(91,80)
		self.assertEqual(y,1)

	def test_cd_count_7(self):
		y = cd_count(12,-65)
		self.assertEqual(y,1)

	def test_cd_count_8(self):
		y = cd_count(-8,1)
		self.assertEqual(y,1)

	def test_cd_count_9(self):
		y = cd_count(-75,-8)
		self.assertEqual(y,1)

