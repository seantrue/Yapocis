#define eps FLT_EPSILON
#define pi 3.1415926535
__kernel
void gradient(__global int awidth, __global int height, __global float* a, __global int reach, __global float* grad, global float* angle)
{
    size_t i;
    float gr,ang;
	float dx, dy;
    int left = (reach*height) + 1;
    int right = awidth - left;
    int h = height;
    i = get_global_id(0);
    gr = 0.0;
    ang = 0.0;
    if ((i >= left) &&  (i < right)) {
		dx = dy = 0.0;
		for (int j = 1; j <= reach; j++) {
		    dx = a[i+j]-a[i-j];
	    	    dy = a[i+j*h]-a[i-j*h];
		}
		gr = sqrt(dx*dx + dy*dy);
		ang = atan(dy/(dx+eps));
		ang /= 2*pi;
    }
    grad[i] = gr;
    angle[i] = ang;
    return;
}