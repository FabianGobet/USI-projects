from unittest import TestCase
from benchmark.caesar_cipher import encrypt, decrypt

class Test_caesar_cipher(TestCase):
	def test_encrypt_1(self):
		y = encrypt("xsj",94)
		self.assertEqual(y,"wri")

	def test_encrypt_2(self):
		y = encrypt("ksva",29)
		self.assertEqual(y,")14~")

	def test_decrypt_1(self):
		y = decrypt("ixlwioczhprwliedlr",18)
		self.assertEqual(y,"WfZeW]QhV^`eZWSRZ`")

	def test_decrypt_2(self):
		y = decrypt("amb",89)
		self.assertEqual(y,"gsh")

	def test_decrypt_3(self):
		y = decrypt("eseadbrpczkka",13)
		self.assertEqual(y,"XfXTWUecVm^^T")

