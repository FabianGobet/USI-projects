from unittest import TestCase
from benchmark.rabin_karp import rabin_karp_search

class Test_rabin_karp(TestCase):
	def test_rabin_karp_search_1(self):
		y = rabin_karp_search("loxqmdp","xohktisbhdmwmozocyuj")
		self.assertEqual(y,[])

	def test_rabin_karp_search_2(self):
		y = rabin_karp_search("","ewergokohiomyobasel")
		self.assertEqual(y,[])

	def test_rabin_karp_search_3(self):
		y = rabin_karp_search("hnutfbljr","gvtoztpyqctxabbnv")
		self.assertEqual(y,[])

	def test_rabin_karp_search_4(self):
		y = rabin_karp_search("t","oxdhtnxiwclwrbppkgv")
		self.assertEqual(y,[4])

	def test_rabin_karp_search_5(self):
		y = rabin_karp_search("tx","langqdq")
		self.assertEqual(y,[])

	def test_rabin_karp_search_6(self):
		y = rabin_karp_search("sur","vpsps")
		self.assertEqual(y,[])

