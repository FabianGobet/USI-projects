from unittest import TestCase
from benchmark.caesar_cipher import encrypt, decrypt

class Test_caesar_cipher(TestCase):
	def test_encrypt_1(self):
		y = encrypt("tapzbjfvqqhnxgcxv",92)
		self.assertEqual(y,"q^mw_gcsnnekud`us")

	def test_encrypt_2(self):
		y = encrypt("iyamag",17)
		self.assertEqual(y,"z+r~rx")

	def test_decrypt_1(self):
		y = decrypt("frmujtkfz",88)
		self.assertEqual(y,"myt|q{rm\"")

