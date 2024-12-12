from unittest import TestCase
from benchmark.caesar_cipher import encrypt, decrypt

class Test_caesar_cipher(TestCase):
	def test_encrypt_1(self):
		y = encrypt("wjhqidhh",11)
		self.assertEqual(y,"#us|toss")

	def test_encrypt_2(self):
		y = encrypt("bpwhsynwgmhr",52)
		self.assertEqual(y,"7EL=HNCL<B=G")

	def test_encrypt_3(self):
		y = encrypt("ifuonemnfkjfluilj",26)
		self.assertEqual(y,"$!0*) ()!&%!'0$'%")

	def test_decrypt_1(self):
		y = decrypt("rbtwohabmg",85)
		self.assertEqual(y,"|l~\"yrklwq")

	def test_decrypt_2(self):
		y = decrypt("qqyerfzwgjvd",44)
		self.assertEqual(y,"EEM9F:NK;>J8")

	def test_decrypt_3(self):
		y = decrypt("wkcqak",52)
		self.assertEqual(y,"C7/=-7")

