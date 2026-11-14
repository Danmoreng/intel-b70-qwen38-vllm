class RangeSet:
    def __init__(self,ranges=()):self._ranges=list(ranges)
    def ranges(self):return self._ranges
    def add(self,start,end):self._ranges.append((start,end))
    def remove(self,start,end):self._ranges=[(a,b) for a,b in self._ranges if b<=start or a>=end]
    def contains(self,point):return any(a<=point<=b for a,b in self._ranges)
    def clip(self,start,end):return [(max(a,start),min(b,end)) for a,b in self._ranges if b>start and a<end]
