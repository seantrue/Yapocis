from yapocis.rpc import interfaces, kernels

program = kernels.load_program(interfaces.zcs)
zcs = program.zcs
zcs_res = program.zcs_res
