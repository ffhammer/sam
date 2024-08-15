from py_lmcurve_ll5 import lmcurve_ll5, ll5Params
a = lmcurve_ll5([1, 2, 3], [1, 2, 3], b=1, c=None, d=3, e=4, f=5)
print(a)


b = vars(a)
print(b)
print(*a)