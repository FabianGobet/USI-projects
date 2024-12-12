from unittest import TestCase
from benchmark.caesar_cipher import encrypt, decrypt

class Test_caesar_cipher(TestCase):
	def test_encrypt_1(self):
		y = encrypt("pciiswz",1)
		self.assertEqual(y,"qdjjtx{")

	def test_encrypt_2(self):
		y = encrypt("fvgbscjdvdzdacejpvqn",26)
		self.assertEqual(y,"!1\"|.}%~1~5~{} %+1,)")

	def test_decrypt_1(self):
		y = decrypt("psoyihclnbvbvslnr",84)
		self.assertEqual(y,"{~z%tsnwym\"m\"~wy}")

	def test_decrypt_2(self):
		y = decrypt("lvvfudixoiwpiwv",62)
		self.assertEqual(y,".88(7&+:1+92+98")

