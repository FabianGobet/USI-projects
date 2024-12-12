from unittest import TestCase
from benchmark.anagram_check import anagram_check

class Test_anagram_check(TestCase):
	def test_anagram_check_1(self):
		y = anagram_check("","")
		self.assertEqual(y,True)

	def test_anagram_check_2(self):
		y = anagram_check("avvuj","kfljj")
		self.assertEqual(y,False)

	def test_anagram_check_3(self):
		y = anagram_check("y","fsrdqgzudakcg")
		self.assertEqual(y,False)

	def test_anagram_check_4(self):
		y = anagram_check("","swyzcsouhfxeonlmux")
		self.assertEqual(y,False)

	def test_anagram_check_5(self):
		y = anagram_check("kmesykeu","geideitnqumf")
		self.assertEqual(y,False)

	def test_anagram_check_6(self):
		y = anagram_check("vhybuzzuf","btxbmzifjgtf")
		self.assertEqual(y,False)

	def test_anagram_check_7(self):
		y = anagram_check("ttlgzl","fgjkzh")
		self.assertEqual(y,False)

	def test_anagram_check_8(self):
		y = anagram_check("qo","u")
		self.assertEqual(y,False)

	def test_anagram_check_9(self):
		y = anagram_check("ltrknvvcjyzagq","bdeyvhpjahijfm")
		self.assertEqual(y,False)

	def test_anagram_check_10(self):
		y = anagram_check("cynvnawsaj","gqzxgxkwjf")
		self.assertEqual(y,False)

	def test_anagram_check_11(self):
		y = anagram_check("o","c")
		self.assertEqual(y,False)

