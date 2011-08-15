<%include file="./median.mako" />

<% left = width/2 %>
__kernel
void ${name}(__global int awidth, __global float* a, __global float* ret )
{
	float aa[${width}];
    size_t i;
    float m;
    int right;
    right = awidth-${left};
    i = get_global_id(0);
    m = 0.0;
    if ((i > ${left}) &&  (i < right)) {
	%for j in range(width):
		aa[${j}] =  a[i+(${j-left})];
    %endfor
    m = median${width}(&aa[0], ${width});
    }
    ret[i] = m;
    return;
}