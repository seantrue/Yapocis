#define eps FLT_EPSILON
#define pi 1.0f
#define UNSAT .001f

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
     H = H*0.5;
     trace[i] = H;
     
    num = r < g ? r : g;
    num = r < b ? r : b;
    den = r + g + b;
    den = den <= eps ? eps : den;
    S = 1 - 3.* num/den; 
    H = S==0 ? 0 : H;
    I = (r + g + b) * (1.0/3.0); 
    
    //S = I < eps ?  0.0 : S;
    //H = I < eps ?  0.0 : H;

    ha[i] = clamp(H, 0.0f, 1.0f-eps);
    sa[i] = clamp(S, 0.0f, 1.0f-eps);
    ia[i] = clamp(I, 0.0f, 1.0f-eps);
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

    R = S < UNSAT ? I : R;
    G = S < UNSAT ? I : G;
    B = S < UNSAT ? I : B;

    ra[i] = clamp(R, 0.0f, 1.0f-eps);
    ga[i] = clamp(G, 0.0f, 1.0f-eps);
    ba[i] = clamp(B, 0.0f, 1.0f-eps);
}
