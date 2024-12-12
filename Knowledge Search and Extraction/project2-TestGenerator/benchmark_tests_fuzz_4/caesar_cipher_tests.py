from unittest import TestCase
from benchmark.caesar_cipher import encrypt, decrypt

class Test_caesar_cipher(TestCase):
	def test_encrypt_1(self):
		y = encrypt("jacj",8)
		self.assertEqual(y,"rikr")

	def test_encrypt_2(self):
		y = encrypt("wifigvkkrphcmt",75)
		self.assertEqual(y,"cURUSbWW^\\TOY`")

	def test_decrypt_1(self):
		y = decrypt("obqpxdfrlzitygqptwro",61)
		self.assertEqual(y,"2%43;')5/=,7<*437:52")

	def test_decrypt_2(self):
		y = decrypt("vxnrv",65)
		self.assertEqual(y,"57-15")

	def test_decrypt_3(self):
		y = decrypt("cblpa",89)
		self.assertEqual(y,"ihrvg")

