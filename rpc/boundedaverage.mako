#define ADDRESS(xx,yy) (((yy)*height)+(xx))
#define eps FLT_EPSILON
#define pi 3.14159265359
#define TWOPI (2*pi)

__kernel 
void boundedaverage(int width,  int height,  __global float *img, __global int *zcs, __global float *output) {
	// width is width of image
	// height is height of image
	// img is a 2d numpy array of shape=(width,height) flattened to 1d
	// output conforms to img
	size_t index = get_global_id(0);
	int y = (int) (index/height);
	int x = (int) (index - y*height);
	float sum = 0.0;
	int n = 0;
	float v = img[index];
	// Scan left, stop after zc
	for (int i = x; i >= 0; i--) {
	    int addr = ADDRESS(i,y);
	    sum += img[addr];
	    n += 1;
	    if (zcs[addr]) break;
	    }
	// Scan right, stop before zc
	for (int i = x+1; i < width; i++) {
	    int addr = ADDRESS(i,y);
	    if (zcs[addr]) break;
	    sum += img[addr];
	    n += 1;
	    }
	// Scan down, stop after zc
	for (int i = y-1; y >= 0; i--) {
	    int addr = ADDRESS(x,i);
	    sum += img[addr];
	    n += 1;
	    if (zcs[addr]) break;
	    }
	// Scan up, stop before zc
	for (int i = y+1; i < height; i++) {
	    int addr = ADDRESS(x,i);
	    if (zcs[addr]) break;
	    sum += img[addr];
	    n += 1;
	    }
	float m = sum/(float)n;
	m = m < 1.0 ? m : 1.0-eps;
	m = m >= 0.0 ? m : 0.0;
	output[index] = isnan(m) ? v : m;
}

__kernel 
void boundedaverageangle(int width,  int height,  __global float *img, __global int *zcs, __global float *output) {
	// width is width of image
	// height is height of image
	// img is a 2d numpy array of shape=(width,height) flattened to 1d, with values that are angle/2*pi
	// output conforms to img
	size_t index = get_global_id(0);
	int y = (int) (index/height);
	int x = (int) (index - y*height);
	float angle;
	float dx = 0;
	float dy = 0;
	int n = 0;
	output[index] = img[index];
	// Scan left, stop after zc
	for (int i = x; i >= 0; i--) {
	    int addr = ADDRESS(i,y);
	    angle = img[addr] * TWOPI;
	    dx += cos(angle);
	    dy += sin(angle);
	    n += 1;
	    if (zcs[addr]) break;
	    }
	// Scan right, stop before zc
	for (int i = x+1; i < width; i++) {
	    int addr = ADDRESS(i,y);
	    if (zcs[addr]) break;
	    angle = img[addr] * TWOPI;
	    dx += cos(angle);
	    dy += sin(angle);
	    n += 1;
	    }
	// Scan down, stop after zc
	for (int i = y-1; y >= 0; i--) {
	    int addr = ADDRESS(x,i);
	    angle = img[addr] * TWOPI;
	    dx += cos(angle);
	    dy += sin(angle);
	    n += 1;
	    if (zcs[addr]) break;
	    }
	// Scan up, stop before zc
	for (int i = y+1; i < height; i++) {
	    int addr = ADDRESS(x,i);
	    if (zcs[addr]) break;
	    angle = img[addr] * TWOPI;
	    dx += cos(angle);
	    dy += sin(angle);
	    n += 1;
	    }
	float m =pi+atan2(-dy,-dx);
	m /= TWOPI;
	m = m < 1.0 ? m : 1.0-eps;
	m = m > 0.0 ? m : 0.0; 
	output[index] = m;
}