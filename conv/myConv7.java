import java.lang.Math;
public class myConv7
{
    public static void main(String[] args) {
        int[] i = new int[]{13,13}; int[] k = new int[]{4,4};
        int[] p = new int[]{3,3};   int[] s = new int[]{2,2};
        int[] d = new int[]{3,3};   int[] a = new int[]{1,1};
        
        double[][] input = rand(i);
        double[][] kernel = rand(k);
        
        //output is convolution of input with kernel
        double[][] output = conv(input, kernel, p, s, d, a, false);
        
        //define convolution matrix and transpose-convolution matrix
        double[][][][] mat = convMat(input, kernel, p, s, d, a, false);
        double[][][][] mat_trans = convMat(output, kernel, p, s, d, a, true);
        
        //check if output can be obtained by multiplying input with convolution matrix
        print(output);
        print(mul(mat, input, false));
        
        //check if multiplication by transpose-convolution matrix is same as 
        //transpose multiplication with convoluton matrix is same as
        //transpose convolution of output
        print(mul(mat_trans, output, false));
        print(mul(mat, output, true));
        print(conv(output, kernel, p, s, d, a, true));
        
        //check if we can recover kernel using matConv
        print(kernel);
        print(matConv(mat, input, k, p, s, d, a, false));
        print(matConv(mat_trans, output, k, p, s, d, a, true));
    }
        
    public static double[][] conv(double[][] input, double[][] kernel, int[] p, int[] s, int[] d, int[] a, boolean transpose) {
        int[] i = shape(input);
        int[] k = shape(kernel);
        
        int[] pad = new int[]{k[0]-p[0]-1, k[1]-p[1]-1};
        int[] add = new int[]{(i[0]-k[0]+2*p[0])%s[0], (i[1]-k[1]+2*p[1])%s[1]};
        
        if(transpose) {
            return conv(input, rot(kernel), pad, d, s, add);
        } 
        else {
            return conv(input, kernel, p, s, d, a);
        }
    }
    
    public static double[][] conv(double[][] input, double[][] kernel, int[] p, int[] s, int[] d, int[] a) {
        assert(a[0]<d[0] && a[1]<d[1]) : "bad input";
        
        int[] ishape = shape(input);
        int[] kshape = shape(kernel);
        
        int i0 = (ishape[0]-1)*(d[0]-1)+ishape[0]+a[0];
        int k0 = kshape[0];
        int o0 = (i0+2*p[0]-k0)/s[0]+1;
        
        int i1 = (ishape[1]-1)*(d[1]-1)+ishape[1]+a[1];
        int k1 = kshape[1];
        int o1 = (i1+2*p[1]-k1)/s[1]+1;

        double[][] output = new double[o0][o1];
        for(int x0=0; x0<o0; x0++) {
            int v0 = x0*s[0]-p[0];
            for(int x1=0; x1<o1; x1++) {
                int v1 = x1*s[1]-p[1];
                for(int y0=ceil_div(max(v0,0),d[0]); y0*d[0]<min(k0+v0,i0); y0++) {
                    for(int y1=ceil_div(max(v1,0),d[1]); y1*d[1]<min(k1+v1,i1); y1++) {
                        output[x0][x1] += kernel[y0*d[0]-v0][y1*d[1]-v1]*input[y0][y1];
                    }
                }
            }
        }
        return output;
    }
        
    public static double[][][][] convMat(double[][] input, double[][] kernel, int[] p, int[] s, int[] d, int[] a, boolean transpose) {
        int[] i = shape(input);
        int[] k = shape(kernel);
        
        int[] pad = new int[]{k[0]-p[0]-1, k[1]-p[1]-1};
        int[] add = new int[]{(i[0]-k[0]+2*p[0])%s[0], (i[1]-k[1]+2*p[1])%s[1]};
        if(transpose) {
            return convMat(input, rot(kernel), pad, d, s, add);
        }      
        else {
            return convMat(input, kernel, p, s, d, a);
        }
    }
    
    public static double[][][][] convMat(double[][] input, double[][] kernel, int[] p, int[] s, int[] d, int[] a) {
        assert(a[0]<d[0] && a[1]<d[1]) : "bad input";
        
        int[] ishape = shape(input);
        int[] kshape = shape(kernel);
        
        int i0 = (ishape[0]-1)*(d[0]-1)+ishape[0]+a[0];
        int k0 = kshape[0];
        int o0 = (i0+2*p[0]-k0)/s[0]+1;
        
        int i1 = (ishape[1]-1)*(d[1]-1)+ishape[1]+a[1];
        int k1 = kshape[1];
        int o1 = (i1+2*p[1]-k1)/s[1]+1;

        double[][][][] convMat = new double[o0][o1][ishape[0]][ishape[1]];
        for(int x0=0; x0<o0; x0++) {
            int v0 = x0*s[0]-p[0];
            for(int x1=0; x1<o1; x1++) {
                int v1 = x1*s[1]-p[1];
                for(int y0=ceil_div(max(v0,0),d[0]); y0*d[0]<min(k0+v0,i0); y0++) {
                    for(int y1=ceil_div(max(v1,0),d[1]); y1*d[1]<min(k1+v1,i1); y1++) {
                        convMat[x0][x1][y0][y1] = kernel[y0*d[0]-v0][y1*d[1]-v1];
                    }
                }
            }
        }
        return convMat;
    }
    
    public static double[][] matConv(double[][][][] convMat, double[][] input, int[] k, int[] p, int[] s, int[] d, int[] a, boolean transpose) {
        int[] i = shape(input);
        double[][] kernel = new double[k[0]][k[1]];
        
        int[] pad = new int[]{k[0]-p[0]-1, k[1]-p[1]-1};
        int[] add = new int[]{(i[0]-k[0]+2*p[0])%s[0], (i[1]-k[1]+2*p[1])%s[1]};
        
        if(transpose) {
            return matConv(convMat, input, kernel, pad, d, s, add, transpose);
        }      
        else {
            return matConv(convMat, input, kernel, p, s, d, a, transpose);
        }
    }
    
    ///in reality you would add values corresponding to gradients with += instead of =
    public static double[][] matConv(double[][][][] convMat, double[][] input, double[][] kernel, int[] p, int[] s, int[] d, int[] a, boolean transpose) {
        assert(a[0]<d[0] && a[1]<d[1]) : "bad input";
        
        int[] ishape = shape(input);
        int[] kshape = shape(kernel);
        
        int i0 = (ishape[0]-1)*(d[0]-1)+ishape[0]+a[0];
        int k0 = kshape[0];
        int o0 = (i0+2*p[0]-k0)/s[0]+1;
        
        int i1 = (ishape[1]-1)*(d[1]-1)+ishape[1]+a[1];
        int k1 = kshape[1];
        int o1 = (i1+2*p[1]-k1)/s[1]+1;
        
        for(int x0=0; x0<o0; x0++) {
            int v0 = x0*s[0]-p[0];
            for(int x1=0; x1<o1; x1++) {
                int v1 = x1*s[1]-p[1];
                for(int y0=ceil_div(max(v0,0),d[0]); y0*d[0]<min(k0+v0,i0); y0++) {
                    for(int y1=ceil_div(max(v1,0),d[1]); y1*d[1]<min(k1+v1,i1); y1++) {
                        kernel[transpose ? rot(y0*d[0]-v0,k0) : y0*d[0]-v0][transpose ? rot(y1*d[1]-v1,k1) : y1*d[1]-v1] = convMat[x0][x1][y0][y1];
                    }
                }
            }
        }
        return kernel;
    }
    
    //combine with matConv to calculate indexing using dot product
    public static double[][] mul(double[][][][] a, double[][] b, boolean transpose) {
        int a1 = a.length;
        int a2 = a[0].length;
        int a3 = a[0][0].length;
        int a4 = a[0][0][0].length;
        
        int b1 = b.length;
        int b2 = b[0].length;
        if(transpose) {
            assert((a1==b1)&&(a2==b2)) : "wrong dim";
            double[][] c = new double[a3][a4];
            for(int i=0; i<c.length; i++) {
                for(int j=0; j<c[i].length; j++) {
                    for(int k=0; k<b.length; k++) {
                        for(int l=0; l<b[k].length; l++) {
                            c[i][j] += a[k][l][i][j]*b[k][l];
                        }
                    }
                }
            } 
            return c;
        }
        else {
            assert((a3==b1)&&(a4==b2)) : "wrong dim";
            double[][] c = new double[a1][a2];
            for(int i=0; i<c.length; i++) {
                for(int j=0; j<c[i].length; j++) {
                    for(int k=0; k<b.length; k++) {
                        for(int l=0; l<b[k].length; l++) {
                            c[i][j] += a[i][j][k][l]*b[k][l];
                        }
                    }
                }
            }
            return c;
        }
    }
    
    public static int ceil_div(int x, int d) {
        return (x%d==0) ? x/d : x/d+1;
    }
    
    public static double[][] rot(double[][] input) {
        int[] i = shape(input);
        double[][] rot = new double[i[0]][i[1]];
        for(int x0=0; x0<i[0]; x0++) {
            for(int x1=0; x1<i[1]; x1++) {
                rot[x0][x1] = input[i[0]-x0-1][i[1]-x1-1];
            }
        }
        return rot;
    }
    
    public static int rot(int x, int k) {
        return k-x-1;
    }
    
    public static int max(int x, int y) {
        return (x>y) ? x : y;
    }
    
    public static int min(int x, int y) {
        return (x<y) ? x : y;
    }
    
    public static int[] shape(double[][] input) {
        return new int[]{input.length, input[0].length};
    }
    
    public static double[][] rand(int[] n) {
        double[][] rand = new double[n[0]][n[1]];
        for(int i=0; i<n[0]; i++) {
            for(int j=0; j<n[1]; j++) {
                rand[i][j] = Math.random();
            }
        }
        return rand;
    }
    
    public static void print(double[][] input) {        
        for(int i=0; i<input.length; i++) {
            for(int j=0; j<input[i].length; j++) {
                System.out.printf("%.4f ",input[i][j]);
            }
            System.out.println();
        }
        System.out.println();
    }
}