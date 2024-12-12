from unittest import TestCase
from benchmark.railfence_cipher import railencrypt, raildecrypt

class Test_railfence_cipher(TestCase):
	def test_railencrypt_1(self):
		y = railencrypt("bteawcqgwluoy",8)
		self.assertEqual(y,"bteyaowuclqwg")

	def test_railencrypt_2(self):
		y = railencrypt("kkcxwuun",99)
		self.assertEqual(y,"kkcxwuun")

	def test_railencrypt_3(self):
		y = railencrypt("qltnroccocqiafmyp",7)
		self.assertEqual(y,"qaliftqmncyropocc")

	def test_raildecrypt_1(self):
		y = raildecrypt("pcqkmfagletqg",18)
		self.assertEqual(y,"pcqkmfagletqg")

	def test_raildecrypt_2(self):
		y = raildecrypt("yxrmbzmtyumwjq",14)
		self.assertEqual(y,"yxrmbzmtyumwjq")

	def test_raildecrypt_3(self):
		y = raildecrypt("dkculns",10)
		self.assertEqual(y,"dkculns")

	def test_raildecrypt_4(self):
		y = raildecrypt("qrhgdhocewubgjchx",3)
		self.assertEqual(y,"qhjorccehwhugbxgd")

	def test_raildecrypt_5(self):
		y = raildecrypt("osucjgbrhemzxz",50)
		self.assertEqual(y,"osucjgbrhemzxz")

