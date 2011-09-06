import numpy as np
from rpc import interfaces, kernels

program = kernels.loadProgram(interfaces.operators,operators=[("add","+"), ("sub","-"),("mul","*"),("div","/")])

add = program.add
add_res = program.add_res
sub = program.sub
sub_res = program.sub_res
mul = program.mul
mul_res = program.mul_res
div = program.div
div_res = program.div_res

def test_operators():
    # 1+0 -> 1
    a = np.ones((100,100), dtype=np.float32)
    b = np.zeros_like(a)
    c = add(a,b)
    d = np.empty_like(a)
    add_res(a,b,d)
    assert c.sum() == c.size*1.0
    program.read(d)
    assert a.sum() == c.sum()
    assert a.sum() == d.sum()
    # 1-1 == 0
    b[:,:] = 1.0
    c = sub(a,b)
    d[:,:]=10
    sub_res(a,b,d)
    program.read(d)
    assert c.sum() == 0.0
    assert c.sum() == d.sum()
    # 1*1 = 1
    c = mul(a,b)
    d[:,:]=10
    mul_res(a,b,d)
    program.read(d)
    assert a.sum() == c.sum()
    assert a.sum() == d.sum()
    # 1/1 = 1
    #c = div(a,b)
    #d[:,:]=10
    #div_res(a,b,d)
    #program.read(d)
    #assert a.sum() == c.sum()
    #assert a.sum() == d.sum()    

if __name__ == "__main__":
    test_operators()
    print "All is well"
