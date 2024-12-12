from unittest import TestCase
from benchmark.caesar_cipher import encrypt, decrypt

class Test_caesar_cipher(TestCase):
	def test_encrypt_1(self):
		y = encrypt("rjifvclcjvtvo",10)
		self.assertEqual(y,"|tsp!mvmt!~!y")

	def test_encrypt_2(self):
		y = encrypt("dvajuqm",69)
		self.assertEqual(y,"J\\GP[WS")

	def test_decrypt_1(self):
		y = decrypt("pydoowkqrlyrnzcvflqe",73)
		self.assertEqual(y,"'0z&&.\"()#0)%1y-|#({")

