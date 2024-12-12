from unittest import TestCase
from benchmark.anagram_check import anagram_check

class Test_anagram_check(TestCase):
	def test_anagram_check_1(self):
		y = anagram_check("","ruaqlkxmtvcmrcpxt")
		self.assertEqual(y,False)

	def test_anagram_check_2(self):
		y = anagram_check("qwpsqfrckbz","krqtvxypsdqh")
		self.assertEqual(y,False)

	def test_anagram_check_3(self):
		y = anagram_check("","")
		self.assertEqual(y,True)

	def test_anagram_check_4(self):
		y = anagram_check("qlhtjnytjdk","dywhsysaibm")
		self.assertEqual(y,False)

	def test_anagram_check_5(self):
		y = anagram_check("j","k")
		self.assertEqual(y,False)

	def test_anagram_check_6(self):
		y = anagram_check("o","omichfbhyuqmbymeydod")
		self.assertEqual(y,False)

	def test_anagram_check_7(self):
		y = anagram_check("cdjlkqq","biglkvr")
		self.assertEqual(y,False)

	def test_anagram_check_8(self):
		y = anagram_check("wmodait","hdmfyhkcz")
		self.assertEqual(y,False)

	def test_anagram_check_9(self):
		y = anagram_check("k","xtmj")
		self.assertEqual(y,False)

	def test_anagram_check_10(self):
		y = anagram_check("yyorngyksrsifwhr","fqgbuhziuhn")
		self.assertEqual(y,False)

	def test_anagram_check_11(self):
		y = anagram_check("i","")
		self.assertEqual(y,False)

