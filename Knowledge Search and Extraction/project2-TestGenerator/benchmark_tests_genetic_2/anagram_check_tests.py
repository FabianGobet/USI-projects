from unittest import TestCase
from benchmark.anagram_check import anagram_check

class Test_anagram_check(TestCase):
	def test_anagram_check_1(self):
		y = anagram_check("vmgxto","lifujq")
		self.assertEqual(y,False)

	def test_anagram_check_2(self):
		y = anagram_check("y","kgrl")
		self.assertEqual(y,False)

	def test_anagram_check_3(self):
		y = anagram_check("","")
		self.assertEqual(y,True)

	def test_anagram_check_4(self):
		y = anagram_check("k","c")
		self.assertEqual(y,False)

	def test_anagram_check_5(self):
		y = anagram_check("rxpvfcwa","phbshzwdbjcunhyzjq")
		self.assertEqual(y,False)

	def test_anagram_check_6(self):
		y = anagram_check("dngpjhduiace","uannocgamg")
		self.assertEqual(y,False)

	def test_anagram_check_7(self):
		y = anagram_check("oxkhotrvtonsxghlonlp","zlqcgxavzxawnlakycs")
		self.assertEqual(y,False)

	def test_anagram_check_8(self):
		y = anagram_check("u","")
		self.assertEqual(y,False)

	def test_anagram_check_9(self):
		y = anagram_check("pmcfrrsudpj","tzqaexawkka")
		self.assertEqual(y,False)

