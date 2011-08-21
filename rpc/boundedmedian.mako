#define order(A,B) {l=A<B?A:B;h=A<B?B:A;A=l;B=h;}
__kernel
void boundedmedian(int offset, __global float* input, __global int *zcs, __global float* output, __global int* trace)
{
    size_t i;
    float val;
    i = get_global_id(0);
    int size = get_global_size(0);
    val = input[i];
    if ((i >= offset) &&  (i <= size-offset)) {
        float al = input[i-offset];
        float am = input[i];
        float ar = input[i+offset];
        float unchanged = am;
        float l,h;
        int zcl = zcs[i-offset];
        int zcm = zcs[i];
        int zcr = zcs[i+offset];
        int status = zcl ? 1:0;
        status |= zcm ? 2:0;
        status |= zcr ? 4:0;
        float avel=(al+am)/2.0;
        float aver= (am+ar)/2.0;
        order(al,am);
        order(am,ar);
        order(al,am);
        float median = am;
        trace[i] = status;
        status = status/2;
        val = median;
        val = status==1 ? aver:val;
        val = status==2 ? avel:val;
        val = status==3 ? unchanged:val;
    }
    output[i] = val;
    return;
}