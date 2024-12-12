from unittest import TestCase
from benchmark.anagram_check import anagram_check

class Test_anagram_check(TestCase):
	def test_anagram_check_1(self):
		y = anagram_check("sk","is")
		self.assertEqual(y,False)

	def test_anagram_check_2(self):
		y = anagram_check("ufdfezp","djzwifbgjjtoiziokqg")
		self.assertEqual(y,False)

	def test_anagram_check_3(self):
		y = anagram_check("llhipddso","pihabphpb")
		self.assertEqual(y,False)

	def test_anagram_check_4(self):
		y = anagram_check("b","ltiyrhlmdbtv")
		self.assertEqual(y,False)

	def test_anagram_check_5(self):
		y = anagram_check("o","i")
		self.assertEqual(y,False)

	def test_anagram_check_6(self):
		y = anagram_check("","")
		self.assertEqual(y,True)

	def test_anagram_check_7(self):
		y = anagram_check("","mtc")
		self.assertEqual(y,False)

	def test_anagram_check_8(self):
		y = anagram_check("fzd","mzc")
		self.assertEqual(y,False)

	def test_anagram_check_9(self):
		y = anagram_check("spggpfftfo","o")
		self.assertEqual(y,False)

	def test_anagram_check_10(self):
		y = anagram_check("v","juj")
		self.assertEqual(y,False)

	def test_anagram_check_11(self):
		y = anagram_check("barcytnggwcrhiispy","bfqorxfrbuwyqvzbdf")
		self.assertEqual(y,False)

	def test_anagram_check_12(self):
		y = anagram_check("uvv","mawgeqwvwhyfaiflrc")
		self.assertEqual(y,False)

