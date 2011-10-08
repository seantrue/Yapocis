#define eps FLT_EPSILON
#define pi 3.141592653589793
__kernel
void gradient(int awidth, int height, __global float* a, int reach, __global float* grad, __global float* angle)
{
    size_t i;
    float gr,ang;
	float dx, dy;
    size_t left = (reach*height) + 1;
    size_t right = awidth - left;
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
		ang = atan2pi(dy,dx);
    }
    grad[i] = gr;
    // TODO: Don't return nans. Seems to do it anyway. 
    angle[i] = isnan(ang) ? 0.0: ang;
    return;
}