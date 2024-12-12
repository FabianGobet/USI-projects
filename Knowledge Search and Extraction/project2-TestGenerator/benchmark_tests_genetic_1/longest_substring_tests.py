from unittest import TestCase
from benchmark.longest_substring import longest_sorted_substr

class Test_longest_substring(TestCase):
	def test_longest_sorted_substr_1(self):
		y = longest_sorted_substr("qedf")
		self.assertEqual(y,"df")

	def test_longest_sorted_substr_2(self):
		y = longest_sorted_substr("donts")
		self.assertEqual(y,"do")

	def test_longest_sorted_substr_3(self):
		y = longest_sorted_substr("zfa")
		self.assertEqual(y,"z")

