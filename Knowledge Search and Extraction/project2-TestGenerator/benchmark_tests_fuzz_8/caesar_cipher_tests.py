from unittest import TestCase
from benchmark.caesar_cipher import encrypt, decrypt

class Test_caesar_cipher(TestCase):
	def test_encrypt_1(self):
		y = encrypt("uons",1)
		self.assertEqual(y,"vpot")

	def test_encrypt_2(self):
		y = encrypt("fhdbawbvtwsaayqkt",87)
		self.assertEqual(y,"^`\\ZYoZnlokYYqicl")

	def test_decrypt_1(self):
		y = decrypt("wdeiuedwchtlldooiusq",23)
		self.assertEqual(y,"`MNR^NM`LQ]UUMXXR^\\Z")

	def test_decrypt_2(self):
		y = decrypt("vxptxsy",91)
		self.assertEqual(y,"z|tx|w}")

	def test_decrypt_3(self):
		y = decrypt("suvciowifcwcn",6)
		self.assertEqual(y,"mop]ciqc`]q]h")

