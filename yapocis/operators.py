from yapocis.rpc import interfaces, kernels

program = kernels.load_program(interfaces.operators, operators=[("add", "+"), ("sub", "-"), ("mul", "*"), ("div", "/")])

add = program.add
add_res = program.add_res
sub = program.sub
sub_res = program.sub_res
mul = program.mul
mul_res = program.mul_res
div = program.div
div_res = program.div_res

