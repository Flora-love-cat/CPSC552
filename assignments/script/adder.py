"""four Bit Half Adder"""

import numpy as np
# solution 1

def AND(x, y):
    return 1*x+1*y-1 >0

def OR(x, y):
    return 2*x+2*y-1 >0
    
def NAND(x, y):
    return (-2)*x+(-2)*y+3 > 0 
    
def XOR(x, y):
    return AND(OR(x, y), NAND(x, y)) > 0 


def fourBitHalfAdder(a1: int, a0: int, b1: int, b0: int) -> tuple[int, int, int]:
    s0 = XOR(a0, b0)
    s1 = XOR(AND(a0, b0), XOR(a1, b1)) 
    c = OR(AND(AND(a0, b0), XOR(a1, b1)), AND(a1, b1))
    
    return c, s1, s0 


# Solution 2
def step(x): 
    return x>0


def twoBitHalfAdder(x1: int, x2: int) -> tuple[int, int]:
    """
    x1: 1-bit input
    x2: 1-bit input
    
    return 
    a3: sum of two 1-bit inputs x1 and x2 
    c: carry bit  
    """
    b = 2
    w = np.array([-1, -1]).T 
    x = np.array([x1, x2])
    a1 = step(w @ x + b) 
    a2_1 = step(w @ np.array([x1, a1]) + b)
    a2_2 = step(w @ np.array([x2, a1]) + b)
    c = step(w @ np.array([a1, a1]) + b) 
    s = step(w @ np.array([a2_1, a2_2]) + b) 
 
    return c, s


def fourBitHalfAdder2(a1: int, a0: int, b1: int, b0: int) -> tuple[int, int, int]:
    """
    a1: second bit of 2-bit input a
    a0: first bit of 2-bit input a
    b1: second bit of 2-bit input b
    b0: first bit of 2-bit input b 
    
    return 
    c: carry bit
    s1: second bit of sum of a and b 
    s0: first bit of sum of a and b 
    """
    c0, s0 = twoBitHalfAdder(a0, b0)
    c1, s0_2 = twoBitHalfAdder(a1, b1) 
    c2, s1 = twoBitHalfAdder(s0_2, c0) 
    _, c = twoBitHalfAdder(c1, c2) 
      
    return c, s1, s0