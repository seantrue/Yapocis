#define eps FLT_EPSILON
#define ADDRESS(xx,yy) (((yy)*height)+(xx))
__kernel 
void averagesegments(__global int *segments,  int width,  int height, float maxval, __global float *img, __global float *output) {
	// nsegments is the number of 5 integer segments
	// segments is a flat array containing chunks in 5 integer increments
	// n, x1,y1, x2,y2
	// width is width of image
	// height is height of image
	// img is a 2d numpy array of shape=(width,height) flattened to 1d
	// output conforms to img
	int isegment;
	int n;
	int x1;
	int y1;
	int x2;
	int y2;
	int addr;
	int addr2;
	float m;
	int index = 0;
	int i;
	int j;
	float sum;
	i = get_global_id(0);
	isegment = i*5;
	n = segments[isegment];
	x1 = segments[isegment+1];
	y1 = segments[isegment+2];
	x2 = segments[isegment+3];
	y2 = segments[isegment+4];
	if (n == 1) {
		addr = ADDRESS(x1,y1);
		output[addr] = img[addr];
	}
	else if(n == 2) {
		addr = ADDRESS(x1,y1);
		addr2 = ADDRESS(x2,y2);
		sum = img[addr]+img[addr2];
		sum = sum < maxval ? sum : sum - maxval;
		output[addr] = output[addr2] = sum/2;
		}
	else {
		sum = 0.0;
		for (i = x1; i <= x2; i += 1) {
			for (j = y1; j <= y2; j+= 1) {
				addr = ADDRESS(i,j);
				sum += img[addr];
				index += 1;
			}
		}
		m = sum/n;
		m = m < eps ? 0: m;
		m = m > 1.0-eps ? 1.0-eps : m;
		for (i = x1; i <= x2; i += 1) {
			for (j = y1; j <= y2; j+= 1) {
				addr = ADDRESS(i,j);
				output[addr] = m;
			}
		}
	}
}
