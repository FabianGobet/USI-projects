from unittest import TestCase
from benchmark.railfence_cipher import railencrypt, raildecrypt

class Test_railfence_cipher(TestCase):
	def test_railencrypt_1(self):
		y = railencrypt("fwbscnirpf",14)
		self.assertEqual(y,"fwbscnirpf")

	def test_railencrypt_2(self):
		y = railencrypt("bnlvllalhdhgcpbccfvm",9)
		self.assertEqual(y,"bcncflbvvpmlclgahldh")

	def test_railencrypt_3(self):
		y = railencrypt("aydmxrxqx",7)
		self.assertEqual(y,"aydmxxrqx")

	def test_railencrypt_4(self):
		y = railencrypt("dxdyzfhoq",64)
		self.assertEqual(y,"dxdyzfhoq")

	def test_railencrypt_5(self):
		y = railencrypt("yjew",32)
		self.assertEqual(y,"yjew")

	def test_raildecrypt_1(self):
		y = raildecrypt("sozwdhsfjoihtajlyfe",11)
		self.assertEqual(y,"sozdsjitjyeflahofhw")

	def test_raildecrypt_2(self):
		y = raildecrypt("xjcpoyyupxctvh",36)
		self.assertEqual(y,"xjcpoyyupxctvh")

	def test_raildecrypt_3(self):
		y = raildecrypt("cefarp",9)
		self.assertEqual(y,"cefarp")

	def test_raildecrypt_4(self):
		y = raildecrypt("gzfnhqwinzlcolydec",8)
		self.assertEqual(y,"gfqncldceyozwnzhil")

	def test_raildecrypt_5(self):
		y = raildecrypt("xguhxqfjawiualzt",15)
		self.assertEqual(y,"xguhxqfjawiualtz")

