from unittest import TestCase
from benchmark.railfence_cipher import railencrypt, raildecrypt

class Test_railfence_cipher(TestCase):
	def test_railencrypt_1(self):
		y = railencrypt("dfxojcawbfraxifvhjv",7)
		self.assertEqual(y,"dxfaixrfofvjbhcwjav")

	def test_railencrypt_2(self):
		y = railencrypt("wdjxmnf",90)
		self.assertEqual(y,"wdjxmnf")

	def test_railencrypt_3(self):
		y = railencrypt("vzpzylllel",31)
		self.assertEqual(y,"vzpzylllel")

	def test_railencrypt_4(self):
		y = railencrypt("xiiidlgfbvgkw",68)
		self.assertEqual(y,"xiiidlgfbvgkw")

	def test_railencrypt_5(self):
		y = railencrypt("itidelkhcgntdjlabrv",51)
		self.assertEqual(y,"itidelkhcgntdjlabrv")

	def test_raildecrypt_1(self):
		y = raildecrypt("mzpnoeh",76)
		self.assertEqual(y,"mzpnoeh")

	def test_raildecrypt_2(self):
		y = raildecrypt("iklgwxdmaproo",18)
		self.assertEqual(y,"iklgwxdmaproo")

	def test_raildecrypt_3(self):
		y = raildecrypt("busqxjbwn",39)
		self.assertEqual(y,"busqxjbwn")

	def test_raildecrypt_4(self):
		y = raildecrypt("fyun",16)
		self.assertEqual(y,"fyun")

	def test_raildecrypt_5(self):
		y = raildecrypt("znaehcgev",5)
		self.assertEqual(y,"zahgvecen")

