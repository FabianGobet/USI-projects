from unittest import TestCase
from benchmark.caesar_cipher import encrypt, decrypt

class Test_caesar_cipher(TestCase):
	def test_encrypt_1(self):
		y = encrypt("awlopvtyscnx",42)
		self.assertEqual(y,",B7:;A?D>.9C")

	def test_encrypt_2(self):
		y = encrypt("ospugupukighongko",13)
		self.assertEqual(y,"|!}#t#}#xvtu|{tx|")

	def test_decrypt_1(self):
		y = decrypt("vnqluiifhasitarnkix",68)
		self.assertEqual(y,"2*-(1%%\"$|/%0|.*'%4")

	def test_decrypt_2(self):
		y = decrypt("vvhpvvxnonakrurxuh",16)
		self.assertEqual(y,"ffX`ffh^_^Q[bebheX")

	def test_decrypt_3(self):
		y = decrypt("iiysujafygsy",49)
		self.assertEqual(y,"88HBD905H6BH")

