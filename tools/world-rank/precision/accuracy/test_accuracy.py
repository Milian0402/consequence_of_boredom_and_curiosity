"""Independent scalar, register, coefficient and decision regression checks."""
import ctypes as C
from fractions import Fraction
import json
import math
import random
import struct
import unittest

from fixtures import build
from oracle import MAG, SIGN, INF, NAN, round_fp32, units, exact, bf, fma, ulps
from quality import compare, decision
from registration import REGISTER, verify


def value(bits):
    return struct.unpack('<f',struct.pack('<I',bits))[0]


def word(x):
    return struct.unpack('<I',struct.pack('<f',x))[0]


class Checks(unittest.TestCase):
    def test_ties_underflow_overflow(self):
        self.assertEqual(round_fp32(1,-150),0)
        self.assertEqual(round_fp32(3,-150),2)
        self.assertEqual(round_fp32(-1,-150),SIGN)
        self.assertEqual(round_fp32((1<<24)-1,-150),0x00800000)
        self.assertEqual(round_fp32((1<<25)-1,103),INF)
        self.assertEqual(round_fp32((1<<26)-3,102),0x7f7fffff)
        self.assertEqual(bf(0x3f808000),0x3f800000)
        self.assertEqual(bf(0x3f818000),0x3f820000)
        self.assertEqual(bf(0xff7fffff),SIGN|INF)
        self.assertEqual(ulps(1, SIGN|1),2)

    def test_fused_against_system_fmaf(self):
        lib=C.CDLL(None)
        fn=lib.fmaf
        fn.argtypes=(C.c_float,)*3
        fn.restype=C.c_float
        rng=random.Random(319517)
        examples=(0,SIGN,1,SIGN|1,0x007fffff,0x00800000,0x7f7fffff,0x3f800000,0xbf800000,INF,SIGN|INF,NAN)
        terms=((x,y,z) for x in examples for y in examples for z in examples)
        samples=([rng.getrandbits(32) for _ in range(3)] for _ in range(15000))
        for stream in (terms,samples):
            for x,y,z in stream:
                want=word(fn(value(x),value(y),value(z)))
                got=fma(x,y,z)
                if want&MAG>INF:
                    self.assertGreater(got&MAG,INF)
                else:
                    self.assertEqual(want,got,(hex(x),hex(y),hex(z)))

    def test_output_integer_fraction(self):
        rng=random.Random(87721)
        for _ in range(50):
            a=[rng.randrange(0x3b800000,0x3f800000) ^ (rng.randrange(2)<<31) for i in range(6)]
            b=[rng.randrange(0x3b800000,0x3f800000) ^ (rng.randrange(2)<<31) for i in range(6)]
            actual=exact(a,b,2,2,3)
            for idx,(_,p,s) in enumerate(actual):
                i,j=divmod(idx,2)
                scaled=[Fraction(value(a[i*3+l]))*Fraction(value(b[l*2+j]))*(1<<298) for l in range(3)]
                self.assertEqual(sum(scaled),p)
                self.assertEqual(sum(map(abs,scaled)),s)

    def test_class_contamination(self):
        a=[0,0x3f800000, 0x3f800000,0xbf800000]
        b=[INF,0x3f800000, INF,0x3f800000]
        out=exact(a,b,2,2,2)
        self.assertEqual([r[0] for r in out],[NAN,0x3f800000,NAN,0])

    def test_direct_budget(self):
        one=0x3f800000
        p=units(one)<<149
        ref=[(one,p,p)]
        for got, expected in [(one,True),(one+1,True),(one+2,False)]:
            r=compare(ref,[got],[one])
            self.assertIs(r['quality_max_pass'],expected)
            self.assertIs(r['quality_rms_pass'],expected)
            self.assertEqual(r['half_ulp_max'],2**-24)
        r=compare([(0,0,0)],[1],[0])
        self.assertTrue(r['quality_rms_pass'])
        self.assertEqual(r['control_zero_changed'],1)
        self.assertIsNone(r['control_delta_relrms'])
        self.assertFalse(decision([])['accuracy_gate'])

    def test_registration(self):
        self.assertEqual(verify(),'ec4bea33f96d3d7d6c13ff87a7f8cfa8e95a221662b1c708263867b70a4cbfc5')
        self.assertEqual(len(json.loads(REGISTER.read_text())['inputs']),38)


if __name__=='__main__':
    unittest.main()
