from unittest import TestCase
from benchmark.caesar_cipher import encrypt, decrypt

class Test_caesar_cipher(TestCase):
	def test_encrypt_1(self):
		y = encrypt("prpiqdbfuhxrx",50)
		self.assertEqual(y,"CEC<D759H;KEK")

	def test_encrypt_2(self):
		y = encrypt("aoxcl",28)
		self.assertEqual(y,"},5 )")

	def test_encrypt_3(self):
		y = encrypt("gicpltav",55)
		self.assertEqual(y,"?A;HDL9N")

	def test_decrypt_1(self):
		y = decrypt("phawatfugtvftkpotahm",92)
		self.assertEqual(y,"skdzdwixjwyiwnsrwdkp")

	def test_decrypt_2(self):
		y = decrypt("rrlxpwuuqgvzq",40)
		self.assertEqual(y,"JJDPHOMMI?NRI")

