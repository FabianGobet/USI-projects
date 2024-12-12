from unittest import TestCase
from benchmark.rabin_karp import rabin_karp_search

class Test_rabin_karp(TestCase):
	def test_rabin_karp_search_1(self):
		y = rabin_karp_search("","xnnfdrzqhllnch")
		self.assertEqual(y,[])

	def test_rabin_karp_search_2(self):
		y = rabin_karp_search("w","kdlw")
		self.assertEqual(y,[3])

	def test_rabin_karp_search_3(self):
		y = rabin_karp_search("luygdbmusyee","kqqhkcjyismckuqvgddz")
		self.assertEqual(y,[])

	def test_rabin_karp_search_4(self):
		y = rabin_karp_search("nrja","tfgcqfaerqumgrcxx")
		self.assertEqual(y,[])

	def test_rabin_karp_search_5(self):
		y = rabin_karp_search("kzsusbh","tfzfsjzsmrkrno")
		self.assertEqual(y,[])

