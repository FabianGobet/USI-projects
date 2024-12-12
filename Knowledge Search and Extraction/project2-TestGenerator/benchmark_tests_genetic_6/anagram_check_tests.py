from unittest import TestCase
from benchmark.anagram_check import anagram_check

class Test_anagram_check(TestCase):
	def test_anagram_check_1(self):
		y = anagram_check("g","yinufyfhuehfugvfpr")
		self.assertEqual(y,False)

	def test_anagram_check_2(self):
		y = anagram_check("nqzm","mras")
		self.assertEqual(y,False)

	def test_anagram_check_3(self):
		y = anagram_check("","")
		self.assertEqual(y,True)

	def test_anagram_check_4(self):
		y = anagram_check("y","h")
		self.assertEqual(y,False)

	def test_anagram_check_5(self):
		y = anagram_check("bp","tc")
		self.assertEqual(y,False)

	def test_anagram_check_6(self):
		y = anagram_check("vhwmaktxisjjxif","pjsbuxmanwwvono")
		self.assertEqual(y,False)

	def test_anagram_check_7(self):
		y = anagram_check("v","ffw")
		self.assertEqual(y,False)

	def test_anagram_check_8(self):
		y = anagram_check("qtaddzwaiedjazugiqm","gjqxszhrt")
		self.assertEqual(y,False)

	def test_anagram_check_9(self):
		y = anagram_check("z","")
		self.assertEqual(y,False)

	def test_anagram_check_10(self):
		y = anagram_check("t","frfmfxiw")
		self.assertEqual(y,False)

