import numpy as np


class HomoFloat(object):
    def __init__(self, n=20):
        self.n = n
        self.k = None
        self.s = None
        self.key_gen()
        self.dj = np.random.permutation(self.n)

    def key_gen(self):
        self.k = np.random.random([self.n, 1])
        self.s = np.random.random([self.n, 1])

    @staticmethod
    def enc_dj(dj):
        return 2 * dj

    @staticmethod
    def dec_dj(dj):
        return dj / 2

    def enc(self, v: float):
        r = np.random.random([self.n, 1])
        p = np.random.random([self.n, 1])
        c1 = self.k * (self.s * v + p) + r
        c1[-1] = self.k[-1] * self.s[-1] * (p + r / self.k)[:-1].sum()
        c2 = c1[self.dj]
        return c2, self.enc_dj(self.dj)

    def dec(self, c, e_dj):
        dj = self.dec_dj(e_dj)
        di = np.argsort(dj)
        c = c[di]
        s = self.s[:-1].sum()
        v = (c / (self.k * s))[:-1].sum() - c[-1] / (self.k[-1] * self.s[-1] * s)
        return v


if __name__ == "__main__":
    hf = HomoFloat()
    print("Public/Private key generated.")
    plaintext = np.math.e
    print("Original text:", plaintext)
    ciphertext, edj = hf.enc(plaintext)
    point_one_ct = 0.1 * ciphertext
    print("Ciphertext:", ciphertext)
    deciphertext = hf.dec(ciphertext, edj)
    point_one_dt = hf.dec(point_one_ct, edj)
    print("Deciphertext: ", deciphertext)
    print("0.1 times Deciphertext: ", point_one_dt)

    plaintext2 = np.math.e
    print("Original text:", plaintext2)
    ciphertext2, edj2 = hf.enc(plaintext2)
    print("Ciphertext2:", ciphertext2)
    ciphertext3 = ciphertext2 + ciphertext
    deciphertext3 = hf.dec(ciphertext3, edj2)
    print("Deciphertext: ", deciphertext3)

    print("Difference: ", (deciphertext3 - plaintext - plaintext2))
