from unittest import TestCase
from benchmark.anagram_check import anagram_check

class Test_anagram_check(TestCase):
	def test_anagram_check_1(self):
		y = anagram_check("frgjvikokpmqwjrhnh","qqililtgbxrzwyw")
		self.assertEqual(y,False)

	def test_anagram_check_2(self):
		y = anagram_check("bhbtygpqym","lndnlkxvnf")
		self.assertEqual(y,False)

	def test_anagram_check_3(self):
		y = anagram_check("","")
		self.assertEqual(y,True)

	def test_anagram_check_4(self):
		y = anagram_check("g","leivhwod")
		self.assertEqual(y,False)

	def test_anagram_check_5(self):
		y = anagram_check("l","lgsmuq")
		self.assertEqual(y,False)

	def test_anagram_check_6(self):
		y = anagram_check("ulqq","qxsp")
		self.assertEqual(y,False)

	def test_anagram_check_7(self):
		y = anagram_check("yzur","oahc")
		self.assertEqual(y,False)

	def test_anagram_check_8(self):
		y = anagram_check("mhysvxnucsqnhzrkdrwi","mrcesiawubakfrwefy")
		self.assertEqual(y,False)

	def test_anagram_check_9(self):
		y = anagram_check("lkjiknfiak","mflshi")
		self.assertEqual(y,False)

	def test_anagram_check_10(self):
		y = anagram_check("q","")
		self.assertEqual(y,False)

	def test_anagram_check_11(self):
		y = anagram_check("j","y")
		self.assertEqual(y,False)

	def test_anagram_check_12(self):
		y = anagram_check("zj","axr")
		self.assertEqual(y,False)

	def test_anagram_check_13(self):
		y = anagram_check("ctmpyeh","mfqkved")
		self.assertEqual(y,False)

