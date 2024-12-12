from unittest import TestCase
from benchmark.rabin_karp import rabin_karp_search

class Test_rabin_karp(TestCase):
	def test_rabin_karp_search_1(self):
		y = rabin_karp_search("vhm","lypslczdhrzpbubcroyr")
		self.assertEqual(y,[])

	def test_rabin_karp_search_2(self):
		y = rabin_karp_search("f","nsyeufdhbctmocvtdmb")
		self.assertEqual(y,[5])

	def test_rabin_karp_search_3(self):
		y = rabin_karp_search("ezzo","xmffysuquxtxtogfcqlh")
		self.assertEqual(y,[])

	def test_rabin_karp_search_4(self):
		y = rabin_karp_search("swtwhmacmgejrvfqiz","oxqmtywgnjitzxkuonaj")
		self.assertEqual(y,[])

	def test_rabin_karp_search_5(self):
		y = rabin_karp_search("lq","pojjoonrduxgqdviuopp")
		self.assertEqual(y,[])

	def test_rabin_karp_search_6(self):
		y = rabin_karp_search("","aumcelkia")
		self.assertEqual(y,[])

	def test_rabin_karp_search_7(self):
		y = rabin_karp_search("pxtrtsu","bnagqqmculfwfhp")
		self.assertEqual(y,[])

