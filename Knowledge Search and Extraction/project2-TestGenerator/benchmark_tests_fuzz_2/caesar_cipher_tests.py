from unittest import TestCase
from benchmark.caesar_cipher import encrypt, decrypt

class Test_caesar_cipher(TestCase):
	def test_encrypt_1(self):
		y = encrypt("csqhkqkaxbye",70)
		self.assertEqual(y,"JZXORXRH_I`L")

	def test_encrypt_2(self):
		y = encrypt("dxjhsxuaypzcvhw",25)
		self.assertEqual(y,"}2$\"-2/z3*4|0\"1")

	def test_encrypt_3(self):
		y = encrypt("mkygrcudjiziep",66)
		self.assertEqual(y,"PN\\JUFXGML]LHS")

	def test_decrypt_1(self):
		y = decrypt("hrecsfeeklkrvuzmitdd",81)
		self.assertEqual(y,"v!sq\"tssyzy!%$){w#rr")

	def test_decrypt_2(self):
		y = decrypt("gcmta",52)
		self.assertEqual(y,"3/9@-")

	def test_decrypt_3(self):
		y = decrypt("ijupnneish",47)
		self.assertEqual(y,":;FA??6:D9")

