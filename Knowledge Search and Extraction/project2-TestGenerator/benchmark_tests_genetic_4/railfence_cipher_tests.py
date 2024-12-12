from unittest import TestCase
from benchmark.railfence_cipher import railencrypt, raildecrypt

class Test_railfence_cipher(TestCase):
	def test_railencrypt_1(self):
		y = railencrypt("gwsihpemieekmpokckgi",16)
		self.assertEqual(y,"gwsihpemieekimgpkock")

	def test_railencrypt_2(self):
		y = railencrypt("jzkiml",23)
		self.assertEqual(y,"jzkiml")

	def test_railencrypt_3(self):
		y = railencrypt("gd",54)
		self.assertEqual(y,"gd")

	def test_railencrypt_4(self):
		y = railencrypt("pbecsnkftunvfvltelno",11)
		self.assertEqual(y,"pboenclsentklfvtfuvn")

	def test_railencrypt_5(self):
		y = railencrypt("qxojfygmnzctihxbovy",8)
		self.assertEqual(y,"qxxhboiojtvfcyyzgnm")

	def test_raildecrypt_1(self):
		y = raildecrypt("cibnewhymjwdvsollxp",7)
		self.assertEqual(y,"cbwmdoxlvjhnieywslp")

	def test_raildecrypt_2(self):
		y = raildecrypt("axk",32)
		self.assertEqual(y,"axk")

	def test_raildecrypt_3(self):
		y = raildecrypt("ehznoijdcum",63)
		self.assertEqual(y,"ehznoijdcum")

	def test_raildecrypt_4(self):
		y = raildecrypt("qkexacdxezct",66)
		self.assertEqual(y,"qkexacdxezct")

	def test_raildecrypt_5(self):
		y = raildecrypt("dtsyggkfhecbikimcdxy",27)
		self.assertEqual(y,"dtsyggkfhecbikimcdxy")

	def test_raildecrypt_6(self):
		y = raildecrypt("hbthoddzopayva",13)
		self.assertEqual(y,"hbthoddzopayav")

	def test_raildecrypt_7(self):
		y = raildecrypt("uugagct",39)
		self.assertEqual(y,"uugagct")

	def test_raildecrypt_8(self):
		y = raildecrypt("oymnztdzyjfmpgcsh",27)
		self.assertEqual(y,"oymnztdzyjfmpgcsh")

