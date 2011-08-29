#define eps FLT_EPSILON
#define pi 1.0

__constant float pi1o3 = (pi/3.0);
__constant float pi2o3 = (2.0*pi/3.0);
__constant float pi4o3 = (4.0*pi/3.0);
__constant float pi5o3 = (5.0*pi/3.0);

__kernel
void rgb2hsi(__global float *ra, __global float *ga, __global float *ba, __global float *ha, __global float *sa, __global float *ia, __global float *trace) {
     size_t i = get_global_id(0);
     float r = ra[i];
     float b = ba[i];
     float g = ga[i];
     float H,S,I;
     float rlb = r-b;
     float rlg = r-g;
     float glb = g-b;
     float num = 0.5*(rlg + rlb);
     float den = sqrt((rlg*rlg) + (rlb*glb));
     H = acospi(num/(den + eps));
     H = b<=g ? H : (2*pi)-H;
     H = H/(2*pi);
     trace[i] = H;
     
    num = r < g ? r : g;
    num = r < b ? r : b;
    den = r + g + b;
    den = den==0? eps : den;
    S = 1 - 3.* num/den; 
    H = S==0 ? 0 : H;
    I = (r + g + b)/3; 
    
    H = H >= 0.0 ? H: 0.0;
    S = S >= 0.0 ? S: 0.0;
    I = I >= 0.0 ? I: 0.0;
    
    S = I < eps ?  0.0 : S;
    H = I < eps ?  0.0 : H;

    ha[i] = H < 1.0 ? H : 1.0-eps; 
    sa[i] = S < 1.0 ? S : 1.0-eps;
    ia[i] = I < 1.0 ? I : 1.0-eps;
}


__inline float ratio(float H, float angle1, float angle2) {
	return cospi(H-angle1)/cospi(angle2-H);
}

__kernel	
void hsi2rgb(__global float *ha, __global float *sa, __global float *ia, __global float *ra, __global float *ga, __global float *ba) {
	size_t i = get_global_id(0);
	float H = ha[i]*2.0*pi;
	float S = sa[i];
	float I = ia[i];
	float R,G,B;
	bool sector;
	
    B = I * (1.0-S);
    R = I * (1.0 + (S*cospi(H)/cospi(pi1o3-H))); 
    G = 3.0*I - (R+B);

    sector = ((pi2o3 <= H) && (H < pi4o3)); 
    R = sector ? I * (1.0-S) : R;
    G = sector ? (I * (1.0 + (S*ratio(H,pi2o3,pi)))) : G;
    B = sector ? 3.0*I - (R+G) :B;
     
    sector = ((pi4o3 <= H) && (H <= pi2o3));
    G = sector ? I * (1.0-S) : G;
    B = sector ? I * (1.0 + (S*ratio(H,pi4o3,pi5o3))) : B;
    R = sector ? 3.0*I - (G+B) : R;

    R = R >= 0.0 ? R: 0.0;
    G = G >= 0.0 ? G: 0.0;
    B = B >= 0.0 ? B: 0.0;

    R = S < eps ? I : R;
    G = S < eps ? I : G;
    B = S < eps ? I : B;

    ra[i] = R < 1.0 ? R : 1.0-eps;
    ga[i] = G < 1.0 ? G : 1.0-eps;
    ba[i] = B < 1.0 ? B : 1.0-eps;
}
