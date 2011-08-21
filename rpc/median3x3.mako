<%include file="median.mako" />

__kernel
void median3x3(int awidth, int height, __global float* a, __global float* ret )
{
    float aa[9];
    size_t i;
    float m;
    size_t left = height + 1;
    size_t right = awidth - left;
    int h = height;
    i = get_global_id(0);
    m = 0.0;
    if ((i >= left) &&  (i < right)) {
		aa[0] =  a[(i-h)-1];
		aa[1] =  a[i-h];
		aa[2] =  a[(i-h)+1];
		aa[3] =  a[i-1];
		aa[4] =  a[i];
		aa[5] =  a[i+1];
		aa[6] =  a[(i+h)-1];
		aa[7] =  a[i+h];
		aa[8] =  a[(i+h)+1];
	    m = median9(&aa[0], 9);
    }
    ret[i] = m;
    return;
}