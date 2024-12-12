from unittest import TestCase
from benchmark.rabin_karp import rabin_karp_search

class Test_rabin_karp(TestCase):
	def test_rabin_karp_search_1(self):
		y = rabin_karp_search("hgqowk","fxmlrpw")
		self.assertEqual(y,[])

	def test_rabin_karp_search_2(self):
		y = rabin_karp_search("fisydh","jfgmhxbtjxeullf")
		self.assertEqual(y,[])

	def test_rabin_karp_search_3(self):
		y = rabin_karp_search("g","mxvuprsxsf")
		self.assertEqual(y,[])

	def test_rabin_karp_search_4(self):
		y = rabin_karp_search("c","rtnrpndysfsdicko")
		self.assertEqual(y,[13])

	def test_rabin_karp_search_5(self):
		y = rabin_karp_search("","kvk")
		self.assertEqual(y,[])

	def test_rabin_karp_search_6(self):
		y = rabin_karp_search("neokht","rnehusncz")
		self.assertEqual(y,[])

