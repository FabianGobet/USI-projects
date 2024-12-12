from unittest import TestCase
from benchmark.rabin_karp import rabin_karp_search

class Test_rabin_karp(TestCase):
	def test_rabin_karp_search_1(self):
		y = rabin_karp_search("zznifao","qoacqvkifw")
		self.assertEqual(y,[])

	def test_rabin_karp_search_2(self):
		y = rabin_karp_search("gb","eoeiwsxzs")
		self.assertEqual(y,[])

	def test_rabin_karp_search_3(self):
		y = rabin_karp_search("eqynrumwhmxpu","vtqqiifwdjujhikgzvlk")
		self.assertEqual(y,[])

	def test_rabin_karp_search_4(self):
		y = rabin_karp_search("","exn")
		self.assertEqual(y,[])

	def test_rabin_karp_search_5(self):
		y = rabin_karp_search("s","hwiomjqqcmswlfteiozt")
		self.assertEqual(y,[10])

	def test_rabin_karp_search_6(self):
		y = rabin_karp_search("tmrsfnnhbjmtv","hbsxxibkdnwxlonaruhb")
		self.assertEqual(y,[])

