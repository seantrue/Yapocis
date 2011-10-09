#define ADDRESS(xx,yy) (((yy)*height)+(xx))
#define Y(i) ((i)/height)
#define X(i) ((i)-(Y(i)*height))

__kernel
void gradient(int width, int height, __global float* input,  __global float* grad, __global float *theta )
{
    size_t i = get_global_id(0);
    int x = X(i);
    int y = Y(i);
    float gr = 0.0;
    float th = 0.0;
    float v, below, left;
    if (0 <= x-1  && x+1 < (width-1) && 0 <= y-1 && y < (height-1)) {
       float dx = input[ADDRESS(x+1,y)] - input[ADDRESS(x-1,y)];
       float dy = input[ADDRESS(x,y+1)] - input[ADDRESS(x,y-1)];
       gr = sqrt(dx*dx+dy*dy);
       th = atan2pi(dy, dx);
    }
    grad[i] = gr;
    theta[i] = th;
}
