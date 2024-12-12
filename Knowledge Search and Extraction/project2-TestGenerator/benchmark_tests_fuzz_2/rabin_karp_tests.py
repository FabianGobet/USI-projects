from unittest import TestCase
from benchmark.rabin_karp import rabin_karp_search

class Test_rabin_karp(TestCase):
	def test_rabin_karp_search_1(self):
		y = rabin_karp_search("joyvmduhbfqsikxqzyd","wgdwrmfkgfpakqxdjkyn")
		self.assertEqual(y,[])

	def test_rabin_karp_search_2(self):
		y = rabin_karp_search("zfnnjt","kzfmgrrtgxcflgkxktfr")
		self.assertEqual(y,[])

	def test_rabin_karp_search_3(self):
		y = rabin_karp_search("e","hiubovm")
		self.assertEqual(y,[])

	def test_rabin_karp_search_4(self):
		y = rabin_karp_search("s","t")
		self.assertEqual(y,[])

	def test_rabin_karp_search_5(self):
		y = rabin_karp_search("f","jgdfzcavxnt")
		self.assertEqual(y,[3])

	def test_rabin_karp_search_6(self):
		y = rabin_karp_search("","w")
		self.assertEqual(y,[])

