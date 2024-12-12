from unittest import TestCase
from benchmark.anagram_check import anagram_check

class Test_anagram_check(TestCase):
	def test_anagram_check_1(self):
		y = anagram_check("kqecympjptr","njcgajnnqyg")
		self.assertEqual(y,False)

	def test_anagram_check_2(self):
		y = anagram_check("ejghmgjdtgvysgxxs","hz")
		self.assertEqual(y,False)

	def test_anagram_check_3(self):
		y = anagram_check("n","x")
		self.assertEqual(y,False)

	def test_anagram_check_4(self):
		y = anagram_check("","")
		self.assertEqual(y,True)

	def test_anagram_check_5(self):
		y = anagram_check("fabpyxvy","nptqrvnc")
		self.assertEqual(y,False)

	def test_anagram_check_6(self):
		y = anagram_check("i","pqdsvfqfoqrtoiht")
		self.assertEqual(y,False)

	def test_anagram_check_7(self):
		y = anagram_check("gblzf","wvgfpuufnja")
		self.assertEqual(y,False)

	def test_anagram_check_8(self):
		y = anagram_check("iiygbdecxczkgkfsipzp","pmdhzhlehyhisrucgc")
		self.assertEqual(y,False)

	def test_anagram_check_9(self):
		y = anagram_check("k","tp")
		self.assertEqual(y,False)

	def test_anagram_check_10(self):
		y = anagram_check("mrwr","zwtuxsdjp")
		self.assertEqual(y,False)

	def test_anagram_check_11(self):
		y = anagram_check("d","ymlme")
		self.assertEqual(y,False)

	def test_anagram_check_12(self):
		y = anagram_check("ir","fjsaknqnkgxsiln")
		self.assertEqual(y,False)

	def test_anagram_check_13(self):
		y = anagram_check("hrgabanvo","bckj")
		self.assertEqual(y,False)

