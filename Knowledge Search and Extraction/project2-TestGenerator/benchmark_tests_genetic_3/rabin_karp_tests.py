from unittest import TestCase
from benchmark.rabin_karp import rabin_karp_search

class Test_rabin_karp(TestCase):
	def test_rabin_karp_search_1(self):
		y = rabin_karp_search("","edzkoajrsnqdnumsoei")
		self.assertEqual(y,[])

	def test_rabin_karp_search_2(self):
		y = rabin_karp_search("kdkmzcmzp","vqtxhmvjtjxqjwsw")
		self.assertEqual(y,[])

	def test_rabin_karp_search_3(self):
		y = rabin_karp_search("a","izllkvdqlpqiaz")
		self.assertEqual(y,[12])

