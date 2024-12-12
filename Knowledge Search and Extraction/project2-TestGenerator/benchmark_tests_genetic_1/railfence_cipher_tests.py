from unittest import TestCase
from benchmark.railfence_cipher import railencrypt, raildecrypt

class Test_railfence_cipher(TestCase):
	def test_railencrypt_1(self):
		y = railencrypt("qhwspeyxyjptlqupy",13)
		self.assertEqual(y,"qhwspeyxyyjpputql")

	def test_railencrypt_2(self):
		y = railencrypt("kkkrmfhdfrzmirsn",8)
		self.assertEqual(y,"kskrnkirmmzfrhfd")

	def test_railencrypt_3(self):
		y = railencrypt("utvsxyeoix",35)
		self.assertEqual(y,"utvsxyeoix")

	def test_railencrypt_4(self):
		y = railencrypt("uk",20)
		self.assertEqual(y,"uk")

	def test_raildecrypt_1(self):
		y = raildecrypt("mtoeqijtxtrbffjy",84)
		self.assertEqual(y,"mtoeqijtxtrbffjy")

	def test_raildecrypt_2(self):
		y = raildecrypt("srkjdqyg",4)
		self.assertEqual(y,"skqgyjrd")

	def test_raildecrypt_3(self):
		y = raildecrypt("nvkmoisdeogijp",65)
		self.assertEqual(y,"nvkmoisdeogijp")

	def test_raildecrypt_4(self):
		y = raildecrypt("nsvqvsaiaamgcat",10)
		self.assertEqual(y,"nsvqvaamctagais")

