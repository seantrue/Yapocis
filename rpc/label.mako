#define ADDRESS(xx,yy) (((yy)*h)+(xx))
__kernel
void label(int awidth, int height, __global int* a, __global int *zcs, __global int* r)
{
    size_t i = get_global_id(0);
    int h = height;
    int y = i/h;
    int x = i - (y*h);
    int l = a[i];
    size_t j = ADDRESS(x,y-1);
    int ll = a[j];
    if (ll > 0 && ll < l && zcs[j]==0) l = ll;
    j = ADDRESS(x-1,y);
    ll = a[j];
    if (ll > 0 && ll < l && zcs[j]==0) l = ll;
    j = ADDRESS(x+1,y);
    ll = a[j];
    if (ll > 0 && ll < l && zcs[j]==0) l = ll;
    j = ADDRESS(x,y+1);
    ll = a[j];
    if (ll > 0 && ll < l && zcs[j]==0) l = ll;
    if (l > 0) r[i] = l;
    return;
}