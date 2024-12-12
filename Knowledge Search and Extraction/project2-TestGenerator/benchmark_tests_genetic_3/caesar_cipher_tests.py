from unittest import TestCase
from benchmark.caesar_cipher import encrypt, decrypt

class Test_caesar_cipher(TestCase):
	def test_encrypt_1(self):
		y = encrypt("yizabiefcakqjut",2)
		self.assertEqual(y,"{k|cdkghecmslwv")

	def test_encrypt_2(self):
		y = encrypt("whvnju",85)
		self.assertEqual(y,"m^ld`k")

	def test_decrypt_1(self):
		y = decrypt("ousfllogqslwxdd",69)
		self.assertEqual(y,"*0.!''*\",.'23~~")

	def test_decrypt_2(self):
		y = decrypt("edmoghokdjgwmhugthc",12)
		self.assertEqual(y,"YXac[\\c_X^[ka\\i[h\\W")

	def test_decrypt_3(self):
		y = decrypt("r",65)
		self.assertEqual(y,"1")

