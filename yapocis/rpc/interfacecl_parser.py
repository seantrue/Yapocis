#
# interfacecl.py
#
# Process an improper subset of the CORBA IDL grammar to facilitate calling opencl from a python Numpy framework
# Inspired by an IDL parser by Paul McGuire, shipped as a demo with pyparser
#

from pyparsing import Literal, CaselessLiteral, Word, Upcase, OneOrMore, ZeroOrMore, \
        Forward, NotAny, delimitedList, oneOf, Group, Optional, Combine, alphas, nums, restOfLine, cStyleComment, \
        alphanums, printables, empty, quotedString, ParseException, ParseResults, Keyword
import pprint
#~ import tree2image

bnf = None
def INTERFACECL_BNF():
    """\
    pyparser grammar for the yapocis interface specification. Inspired by an IDL parser by Paul McGuire, shipped as a demo with pyparser.
    """
    global bnf
    
    if not bnf:

        # punctuation
        lbrace = Literal("{")
        rbrace = Literal("}")
        lparen = Literal("(")
        rparen = Literal(")")
        dot    = Literal(".")
        star   = Literal("*")
        semi   = Literal(";")
        
        # keywords
        boolean_   = Keyword("boolean")
        char_      = Keyword("char")
        complex64_ = Keyword("complex64")
        float_   = Keyword("float")
        float32_   = Keyword("float32")
        inout_     = Keyword("inout")
        interface_ = Keyword("interface")
        in_        = Keyword("in")
        int_     = Keyword("int")
        int16_     = Keyword("int16")
        int32_     = Keyword("int32")
        kernel_    = Keyword("kernel")
        out_       = Keyword("out")
        short_     = Keyword("short")
        uint16_     = Keyword("uint16")
        uint32_     = Keyword("uint32")
        void_      = Keyword("void")
        # Special keywords 
        alias_     = Keyword("alias")
        as_        = Keyword("as")
        outlike_   = Keyword("outlike")
        resident_  = Keyword("resident")
        widthof_   = Keyword("widthof")
        heightof_  = Keyword("heightof")
        sizeof_    = Keyword("sizeof") 
        
        identifier = Word( alphas, alphanums + "_" )
        typeName = (boolean_ ^ char_  ^ int16_ ^ int32_ ^ float32_ ^ complex64_ ^ uint16_ ^ uint32_ ^ int_ ^ float_ ^ short_)
        bufferHints = (inout_ | in_ | out_ | outlike_ | resident_ | widthof_ | heightof_ | sizeof_)
        paramlist = delimitedList( Group(bufferHints + Optional(typeName) + Optional(star) + identifier))
        interfaceItem = ((kernel_^void_^alias_^typeName) + identifier + Optional(Group(as_+identifier)) + lparen + Optional(paramlist) + rparen + semi)
        interfaceDef = Group(interface_ + identifier  + lbrace + ZeroOrMore(interfaceItem) + rbrace + semi)
        moduleItem = interfaceDef

        bnf = OneOrMore( moduleItem )
        
        singleLineComment = "//" + restOfLine
        bnf.ignore( singleLineComment )
        bnf.ignore( cStyleComment )
        
    return bnf

class InterfaceCL:
    """\
    Manages the kernel definitions, particularly the parameter specifications.
    """
    def __init__(self, interfacename, kerneldefs, kernelaliases):
        self.interfacename = interfacename
        self.kerneldefs_ = kerneldefs
        self.kernelaliases = kernelaliases
        self.program = None
    def kernels(self):
        "Returns a list of kernel names"
        return self.kerneldefs_.keys()
    def kernelparams(self, kernel):
        "Returns the parameter specifications for a kernel"
        return self.kerneldefs_.get(kernel, None)
    def kernelalias(self, kernelname):
        return self.kernelaliases.get(kernelname,kernelname)

dtypemap={"int":"int32","float":"float32","short":"int16"}
def fixParam(param):
    """\
    Deals with variants of parameter specifications, returns a uniform array of information.
    """
    # Length 2: outlike, identifier (exists)
    # Length 3: outlike, dtype, identifier (exists)
    # Length 3: in, dtype, identifier 
    # Length 4 direction, dtype, *, identifier
    op = ["","","",""]
    if len(param) == 2:
        assert param[0] in ("outlike","resident","sizeof","widthof","heightof")
        op[0] = param[0]
        op[2] = '*'
        op[3] = param[1]
    elif len(param) == 3:
        assert param[0] in ("in","outlike","sizeof","widthof","heightof")
        if param[0] in ("outlike","sizeof","widthof","heightof"):
            op[0] = param[0]
            op[1] = param[1]
            op[2] = '*'
            op[3] = param[2]
        else:
            op[0] = param[0]
            op[1] = param[1]
            op[3] = param[2]
    else:
        op = param
    # Map from OpenCL datatype to a numpy data type
    op[1] = dtypemap.get(op[1],op[1])
    return op
        
        
def getInterfaceCL(s):
    """\
    Builds an InterfaceCL for the source interface definition.
    """
    try:
        bnf = INTERFACECL_BNF()
        tokens = bnf.parseString(s)
        tokens = tokens.asList()
        tokens = tokens.pop(0)
        assert tokens.pop(0) == "interface"
        interfacename = tokens.pop(0)
        kerneldefs = {}
        kernelaliases = {}
        assert tokens.pop(0) == "{"
        while 1:
            token = tokens.pop(0)
            if token == "}":
                break
            assert token in ("kernel","alias")
            if token == "alias":
                alias = tokens.pop(0)
                assert tokens[0][0] == "as"
                kernelname = tokens.pop(0)[1]
                kernelaliases[alias] = kernelname
                kernelname = alias
            else:
                alias = None
                kernelname = tokens.pop(0)
            params = []
            assert tokens.pop(0) == '('
            while 1:
                param = tokens.pop(0)
                if param == ")":
                    break
                params.append(fixParam(param))
            assert tokens.pop(0) == ";"
            kerneldefs[kernelname] =  params
        return InterfaceCL(interfacename, kerneldefs, kernelaliases)
    except ParseException, err:
        print err.line
        print " "*(err.column-1) + "^"
        print err
        raise 

def test_getinterfacecl():
    interface = getInterfaceCL(
        """
        interface boundedmedian {
             kernel boundedmedian(sizeof int input, in float32 *input, in int32 *zcs, outlike int16 input, out short *trace);
             alias bm as boundedmedian(in int32 offset, resident float32 *input, in int32 *zcs, resident float *input, out int16 *trace);
          };
        """
        )
    print "Interface:", interface.interfacename
    for kernel in interface.kernels():
        print "Kernel: %s alias for %s" % (kernel, interface.kernelalias(kernel))
        symbols = {}
        iparam = 0
        for param in interface.kernelparams(kernel):
            assert len(param) == 4
            print "Param:", param,
            direction, dtype, isbuffer, name = param
            assert direction in ("in","out","inout","outlike","resident","sizeof","widthof","heightof")
            if direction == "outlike":
                assert name in symbols
                iparam, olparam = symbols[name]
                dtype,isbuffer = olparam[1],olparam[2]
                print "->", olparam,
                assert isbuffer
            else:
                symbols[name] = iparam,param
                iparam += 1
            assert isbuffer in ("*","")
            assert dtype in ("int16", "int32", "float32","uint16","uint32","complex64","int","float","short")
            togpu = direction in ("in","inout")
            fromgpu = direction in ("out","inout","outlike")
            if isbuffer:
                if direction == "resident":
                    print "Buffer is resident on GPU.",
                elif direction == "outlike":
                    print "Allocate a buffer like %(name)s (position=%(iparam)s). (%(olparam)s)" % locals(),
                elif direction in  ("sizeof","heightof","widthof"):
                    print "Pass in %s of %s" % (direction,name),
                else:
                    print "Coerce %(name)s to numpy.%(dtype)s. " % locals(), 
                    print "Allocate a buffer for %(name)s. " % locals(),
                if togpu:
                    print "Copy to GPU. ",
                if fromgpu:
                    print "Copy back from GPU.",
            else:
                assert not fromgpu
                print "Use as parameter.",
            print
        print
    
if __name__ == "__main__":
    test_getinterfacecl()
