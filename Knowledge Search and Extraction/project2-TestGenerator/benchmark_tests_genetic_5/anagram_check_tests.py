from unittest import TestCase
from benchmark.anagram_check import anagram_check

class Test_anagram_check(TestCase):
	def test_anagram_check_1(self):
		y = anagram_check("","")
		self.assertEqual(y,True)

	def test_anagram_check_2(self):
		y = anagram_check("m","rk")
		self.assertEqual(y,False)

	def test_anagram_check_3(self):
		y = anagram_check("z","i")
		self.assertEqual(y,False)

	def test_anagram_check_4(self):
		y = anagram_check("i","eiafgm")
		self.assertEqual(y,False)

	def test_anagram_check_5(self):
		y = anagram_check("fk","kv")
		self.assertEqual(y,False)

	def test_anagram_check_6(self):
		y = anagram_check("tixyenasxyevvu","envuvuhvfilsznxfpgi")
		self.assertEqual(y,False)

	def test_anagram_check_7(self):
		y = anagram_check("zuabb","ykufwdsta")
		self.assertEqual(y,False)

	def test_anagram_check_8(self):
		y = anagram_check("wm","miynbxzq")
		self.assertEqual(y,False)

	def test_anagram_check_9(self):
		y = anagram_check("n","inathxwxcozkdcvt")
		self.assertEqual(y,False)

	def test_anagram_check_10(self):
		y = anagram_check("b","wymgmgrakz")
		self.assertEqual(y,False)

