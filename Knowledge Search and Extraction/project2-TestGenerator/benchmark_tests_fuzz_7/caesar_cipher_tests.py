from unittest import TestCase
from benchmark.caesar_cipher import encrypt, decrypt

class Test_caesar_cipher(TestCase):
	def test_encrypt_1(self):
		y = encrypt("xxnqheaim",25)
		self.assertEqual(y,"22(+\"~z#'")

	def test_encrypt_2(self):
		y = encrypt("zzukotdy",55)
		self.assertEqual(y,"RRMCGL<Q")

	def test_decrypt_1(self):
		y = decrypt("ldxwnjuawnsl",78)
		self.assertEqual(y,"}u*) {'r) %}")

	def test_decrypt_2(self):
		y = decrypt("bp",1)
		self.assertEqual(y,"ao")

