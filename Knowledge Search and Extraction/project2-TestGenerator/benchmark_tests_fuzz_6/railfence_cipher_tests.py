from unittest import TestCase
from benchmark.railfence_cipher import railencrypt, raildecrypt

class Test_railfence_cipher(TestCase):
	def test_railencrypt_1(self):
		y = railencrypt("hwrdxhhzaqw",16)
		self.assertEqual(y,"hwrdxhhzaqw")

	def test_railencrypt_2(self):
		y = railencrypt("hj",57)
		self.assertEqual(y,"hj")

	def test_railencrypt_3(self):
		y = railencrypt("oxkems",53)
		self.assertEqual(y,"oxkems")

	def test_railencrypt_4(self):
		y = railencrypt("fru",2)
		self.assertEqual(y,"fur")

	def test_railencrypt_5(self):
		y = railencrypt("h",5)
		self.assertEqual(y,"h")

	def test_railencrypt_6(self):
		y = railencrypt("ilpcar",51)
		self.assertEqual(y,"ilpcar")

	def test_railencrypt_7(self):
		y = railencrypt("jtruevggqxpl",4)
		self.assertEqual(y,"jgtvglreqpux")

	def test_raildecrypt_1(self):
		y = raildecrypt("wwqnjyaxhcms",41)
		self.assertEqual(y,"wwqnjyaxhcms")

	def test_raildecrypt_2(self):
		y = raildecrypt("atcbqnkfidpmokzl",8)
		self.assertEqual(y,"acnfdmklzopikbtq")

	def test_raildecrypt_3(self):
		y = raildecrypt("ftll",24)
		self.assertEqual(y,"ftll")

	def test_raildecrypt_4(self):
		y = raildecrypt("nimvjzazjikvfmw",55)
		self.assertEqual(y,"nimvjzazjikvfmw")

