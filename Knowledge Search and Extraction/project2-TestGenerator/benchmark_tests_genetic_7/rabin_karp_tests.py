from unittest import TestCase
from benchmark.rabin_karp import rabin_karp_search

class Test_rabin_karp(TestCase):
	def test_rabin_karp_search_1(self):
		y = rabin_karp_search("a","dbh")
		self.assertEqual(y,[])

	def test_rabin_karp_search_2(self):
		y = rabin_karp_search("deokrs","ppqehakvqgasxvnyaath")
		self.assertEqual(y,[])

	def test_rabin_karp_search_3(self):
		y = rabin_karp_search("h","szqipgtsmwehpdj")
		self.assertEqual(y,[11])

	def test_rabin_karp_search_4(self):
		y = rabin_karp_search("d","toei")
		self.assertEqual(y,[])

