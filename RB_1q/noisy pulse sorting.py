import copy as cp

class Pulse:
    def __init__(self, pulse_type, axis, sign, angle):
        if not (pulse_type == 'noise' or pulse_type == 'pulse'):
            raise Exception('"pulse_type" could only has value "noise" or "pulse".')
        self.pulse_type = pulse_type

        if not (axis == 'X' or axis == 'Y' or axis == 'Z'):
            raise Exception('"axis" could only has value \'X\', \'Y\' or \'Z\'.')
        self.axis = axis

        if not (sign == 1 or sign == -1):
            raise Exception('"sign" could only has value 1 or -1.')
        self.sign = sign

        if not isinstance(angle, str):
            raise TypeError('"angle" could only has type "char".')
        self.angle = angle


def eijk(p1, p2, p3):   # Levi-Civita symbol
    a = (p1, p2, p3)
    if a == ('X', 'Y', 'Z') or a == ('Y', 'Z', 'X') or a == ('Z', 'X', 'Y'):
        return 1
    elif a == ('Z', 'Y', 'X') or a == ('Y', 'X', 'Z') or a == ('X', 'Z', 'Y'):
        return -1
    else:
        return None

def third_pauli(a1, a2):
    return chr(267 - ord(a1) - ord(a2))

def commute_transform(p1, p2):
    if not (isinstance(p1, Pulse) and isinstance(p2, Pulse)):
        raise TypeError('The two inputs should be of type "Pulse" objects.')
    if p1.pulse_type == 'pulse' and p2.pulse_type == 'noise':
        if p1.axis == p2.axis:
            temp = cp.deepcopy(p2)

            p2.pulse_type = p1.pulse_type
            p2.axis = p1.axis
            p2.sign = p1.sign
            p2.angle = p1.angle

            p1.pulse_type = temp.pulse_type
            p1.axis = temp.axis
            p1.sign = temp.sign
            p1.angle = temp.angle
        else:
            ax = third_pauli(p1.axis, p2.axis)
            sgn = p1.sign * p2.sign * eijk(p1.axis, p2.axis, ax)
            temp = cp.deepcopy(p2)

            p2.pulse_type = p1.pulse_type       # 'pulse'
            p2.axis = p1.axis
            p2.sign = p1.sign
            p2.angle = p1.angle

            p1.pulse_type = temp.pulse_type     # 'noise'
            p1.axis = ax
            p1.sign = sgn
            p1.angle = temp.angle
        return True
    else:
        return False

def print_pulse(p):
    if not (isinstance(p, Pulse)):
        raise TypeError('The input should be of type "Pulse" object.')
    print("{" + p.pulse_type + " ; " + p.axis + " ; " + str(p.sign) + " * " + p.angle + "}")

def noisy_X(sign):
    if not (sign == 1 or sign == -1):
        raise Exception('"sign" of noisy_X pulse could only has value 1 or -1.')
    else:
        a1 = Pulse('noise', 'Y', -1, 'gamma')
        a2 = Pulse('pulse', 'X', sign, 'pi/2')
        a3 = Pulse('noise', 'X', sign, 'delta')
        a4 = Pulse('noise', 'Y', 1, 'gamma')
        return [a1, a2, a3, a4]

def Z(sign):
    if not (sign == 1 or sign == -1):
        raise Exception('"sign" of noisy_X pulse could only has value 1 or -1.')
    a = Pulse('pulse', 'Z', sign, 'pi/2')
    return [a]

def X(sign):
    if not (sign == 1 or sign == -1):
        raise Exception('"sign" of noisy_X pulse could only has value 1 or -1.')
    a = Pulse('pulse', 'X', sign, 'pi/2')
    return [a]

def list_flatten(t):
    return [item for sublist in t for item in sublist]


L = [Z(-1), noisy_X(1), Z(-1), noisy_X(1)]

L = list_flatten(L)
print("\nSimplify...\n")
for i in range(len(L)-1):
    if i >= len(L)-1:
        break
    elif L[i].pulse_type == L[i+1].pulse_type and L[i].axis == L[i+1].axis and L[i].angle == L[i+1].angle:
        if L[i].sign != L[i+1].sign:
            del L[i:i+2]

for i in range(len(L)):
    print_pulse(L[i])

print("\nTransform noises into LHS...\n")
UNSORTED = True
boo = True
while UNSORTED:
    boo = False
    for i in range(len(L)-1):
        boo = boo or commute_transform(L[i], L[i+1])
    UNSORTED = boo

for i in range(len(L)):
    print_pulse(L[i])






