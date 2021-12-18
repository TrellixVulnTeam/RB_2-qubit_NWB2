import qecc as q

Cliff_1 = q.clifford_group(1, consider_phases=True)

for i in range(24):
    a = next(Cliff_1)
    b = a.as_bsm()
    c = a.as_unitary()
    print(a)
    print(b)
    print(c, "\n")
