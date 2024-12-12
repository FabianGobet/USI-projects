from unittest import TestCase
from benchmark.railfence_cipher import railencrypt, raildecrypt

class Test_railfence_cipher(TestCase):
	def test_railencrypt_1(self):
		y = railencrypt("vukjo",14)
		self.assertEqual(y,"vukjo")

	def test_railencrypt_2(self):
		y = railencrypt("ngxmqndbfhudydwxp",5)
		self.assertEqual(y,"nfpgbhxxduwmnddqy")

	def test_railencrypt_3(self):
		y = railencrypt("il",7)
		self.assertEqual(y,"il")

	def test_railencrypt_4(self):
		y = railencrypt("hgrregh",78)
		self.assertEqual(y,"hgrregh")

	def test_railencrypt_5(self):
		y = railencrypt("vzchmajzgl",12)
		self.assertEqual(y,"vzchmajzgl")

	def test_railencrypt_6(self):
		y = railencrypt("iroljkjuxm",24)
		self.assertEqual(y,"iroljkjuxm")

	def test_raildecrypt_1(self):
		y = raildecrypt("ypombgorvurvdot",5)
		self.assertEqual(y,"yoguorompbrvtdv")

	def test_raildecrypt_2(self):
		y = raildecrypt("dvfkqyaohtgcpguzwwr",67)
		self.assertEqual(y,"dvfkqyaohtgcpguzwwr")

