from unittest import TestCase
from benchmark.anagram_check import anagram_check

class Test_anagram_check(TestCase):
	def test_anagram_check_1(self):
		y = anagram_check("gqwe","ynisgfnxom")
		self.assertEqual(y,False)

	def test_anagram_check_2(self):
		y = anagram_check("","")
		self.assertEqual(y,True)

	def test_anagram_check_3(self):
		y = anagram_check("dpurnif","mcd")
		self.assertEqual(y,False)

	def test_anagram_check_4(self):
		y = anagram_check("ezte","pqsq")
		self.assertEqual(y,False)

	def test_anagram_check_5(self):
		y = anagram_check("","h")
		self.assertEqual(y,False)

	def test_anagram_check_6(self):
		y = anagram_check("vpphh","nnrxi")
		self.assertEqual(y,False)

	def test_anagram_check_7(self):
		y = anagram_check("qmlbnf","smhkp")
		self.assertEqual(y,False)

	def test_anagram_check_8(self):
		y = anagram_check("upw","ycdclanzkgbdkgtprhi")
		self.assertEqual(y,False)

	def test_anagram_check_9(self):
		y = anagram_check("e","tkxkmnk")
		self.assertEqual(y,False)

	def test_anagram_check_10(self):
		y = anagram_check("yenqtxjvellful","vc")
		self.assertEqual(y,False)

	def test_anagram_check_11(self):
		y = anagram_check("x","f")
		self.assertEqual(y,False)

