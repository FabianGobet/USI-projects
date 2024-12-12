from unittest import TestCase
from benchmark.rabin_karp import rabin_karp_search

class Test_rabin_karp(TestCase):
	def test_rabin_karp_search_1(self):
		y = rabin_karp_search("vczyaspv","avzbrpofvltostmomra")
		self.assertEqual(y,[])

	def test_rabin_karp_search_2(self):
		y = rabin_karp_search("k","sgkwancaakezzrzeeoa")
		self.assertEqual(y,[2,9])

	def test_rabin_karp_search_3(self):
		y = rabin_karp_search("","pfmczhoobiffbhm")
		self.assertEqual(y,[])

