#define ADDRESS(xx,yy) (((yy)*height)+(xx))
#define Y(i) ((i)/height)
#define X(i) ((i)-(Y(i)*height))

__kernel
void zcs(int width, int height, __global float* input, __global float* zcs )
{
    size_t i = get_global_id(0);
    int x = X(i);
    int y = Y(i);
    float zc = 0.0;
    float v, below, left;
    if (x > 0 && y > 0) {
       v = input[i];
       below = input[ADDRESS(x,y-1)];
       left = input[ADDRESS(x-1,y)];
       // TODO: Change this to conditional expressions
       if ((below < 0.0 || left < 0.0) && v >= 0.0) zc=1.0;
       if ((below >= 0.0 || left >= 0.0) && v < 0.0) zc=1.0;
    }
    zcs[i] = zc;
}
