"""Scalar independent Philox4x32-10 oracle for acceptance thresholds.

Known-answer vectors: DEShawResearch/random123, tests/kat_vectors.
https://github.com/DEShawResearch/random123/blob/main/tests/kat_vectors
No Triton, vLLM or GPU import is used here.
"""
import struct
import math

MASK=(1<<32)-1


def philox(counter,key):
    c0,c1,c2,c3=counter;k0,k1=key
    for _ in range(10):
        a=0xD2511F53*c0;b=0xCD9E8D57*c2
        c0,c1,c2,c3=((b>>32)^c1^k0)&MASK,b&MASK,((a>>32)^c3^k1)&MASK,a&MASK
        k0=(k0+0x9E3779B9)&MASK;k1=(k1+0xBB67AE85)&MASK
    return c0,c1,c2,c3


def f32(x):return struct.unpack('<f',struct.pack('<f',x))[0]


def acceptance_uniform(seed,position):
    value=philox((position&MASK,(position>>32)&MASK,0,0),(seed&MASK,(seed>>32)&MASK))[0]
    # Triton folds the signed bit pattern to a nonnegative integer before
    # correctly rounded FP32 conversion and multiplication by its open scale.
    folded=(~value)&MASK if value&(1<<31) else value
    scale=f32(4.6566127342e-10)
    return max(f32(f32(folded)*scale),scale)


def rotl(value,bits):return ((value<<bits)|(value>>(32-bits)))&MASK


def murmur(seed,position,token,domain=0):
    h=domain
    for word in [seed&MASK,(seed>>32)&MASK,position&MASK,token&MASK]:
        k=rotl((word*0xCC9E2D51)&MASK,15)
        k=(k*0x1B873593)&MASK
        h=(rotl(h^k,13)*5+0xE6546B64)&MASK
    h^=16;h^=h>>16;h=(h*0x85EBCA6B)&MASK
    h^=h>>13;h=(h*0xC2B2AE35)&MASK
    return (h^(h>>16))&MASK


def gumbel(seed,position,token,fp64):
    low=murmur(seed,position,token)
    if fp64:
        high=murmur(seed,position,token,0x9E3779B9)
        bits53=((high<<32)|low)>>11
        u=(bits53+.5)*2**-53
        if u==1:u-=2**-53
        return -math.log(-math.log(u))
    u=f32((low+.5)*2**-32)
    # Independent log1p instead of the production degree-eight polynomial.
    return -math.log(-math.log1p(-u)) if u<.25 else -math.log(-math.log(max(1-u,2**-24)))


def sampled_winner(logits,seed,position,fp64):
    values=[(float(value)+gumbel(seed,position,int(token),fp64),int(token))
        for token,value in logits if math.isfinite(float(value))]
    assert values
    values.sort(reverse=True)
    margin=values[0][0]-values[1][0] if len(values)>1 else math.inf
    return values[0][1],margin


def known_answers():
    cases=[((0,0,0,0),(0,0),(0x6627E8D5,0xE169C58D,0xBC57AC4C,0x9B00DBD8)),
        ((MASK,)*4,(MASK,)*2,(0x408F276D,0x41C83B0E,0xA20BC7C6,0x6D5451FD)),
        ((0x243F6A88,0x85A308D3,0x13198A2E,0x03707344),(0xA4093822,0x299F31D0),
            (0xD16CFE09,0x94FDCCEB,0x5001E420,0x24126EA1))]
    for counter,key,expected in cases:assert philox(counter,key)==expected
    return len(cases)


if __name__=='__main__':print('PASS_RANDOM123_KNOWN_ANSWERS',known_answers())
