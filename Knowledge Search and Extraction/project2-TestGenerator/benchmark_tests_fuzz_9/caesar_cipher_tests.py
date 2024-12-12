from unittest import TestCase
from benchmark.caesar_cipher import encrypt, decrypt

class Test_caesar_cipher(TestCase):
	def test_encrypt_1(self):
		y = encrypt("rcfepqcpohhadmlqhnm",10)
		self.assertEqual(y,"|mpoz{mzyrrknwv{rxw")

	def test_encrypt_2(self):
		y = encrypt("hzkttoaqllzjvv",20)
		self.assertEqual(y,"|/ ))$u&!!/~++")

	def test_decrypt_1(self):
		y = decrypt("ayxiqvxxaxo",70)
		self.assertEqual(y,"z32#+022z2)")

