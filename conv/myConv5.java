import java.lang.Math;
public class myConv5
{
    public static void main(String[] args) {
        int i = 13;
        int k = 4;
        int p = 3;
        int s = 2;        
        int d = 3;
        int a = 1;
        
        double[] input = rand(i);
        double[] kernel = rand(k);
        double[] output = conv(input, kernel, p, s, d, a, false);
        
        double[][] mat = convMat(input, kernel, p, s, d, a, false);
        double[][] mat_trans = convMat(output, kernel, p, s, d, a, true);
        
        print(output);
        print(mul(mat, input, false));
        
        print(mul(mat_trans, output, false));
        print(mul(mat, output, true));
    }
    
    public static double diff(double[] a, double[] b) {
        double diff = 0;
        assert(a.length==b.length) : "unequal lengths";
        for(int i=0; i<a.length; i++) {
            diff+=(a[i]-b[i])*(a[i]-b[i]);
        }
        return diff/a.length;
    }
    
    public static double[] rot(double[] input) {
        int i = input.length;
        double[] rot = new double[i];
        for(int x=0; x<i; x++) {
            rot[i-x-1] = input[x];
        }
        return rot;
    }
    
    public static double[] conv(double[] input, double[] kernel, int p, int s, int d, int a, boolean transpose) {
        int i = input.length;
        int k = kernel.length;
        
        if(transpose) {
            return conv(input, rot(kernel), k-p-1, d, s, (i-k+2*p)%s);
        }
        
        else {
            return conv(input, kernel, p, s, d, a);
        }
    }
    
    public static double[] conv(double[] input, double[] kernel, int p, int s, int d, int a) {
        assert(a<d) : "bad input";
        int i = (input.length-1)*(d-1)+input.length+a;
        int k = kernel.length;
        int o = (i+2*p-k)/s + 1;
        double[] output = new double[o];
        for(int x=0; x<o; x++) {
            int v = x*s-p;
            for(int y=ceil_div(max(v,0),d); y*d<min(k+v,i); y++) {
                output[x] += kernel[y*d-v]*input[y];
            }
        }
        return output;
    }
    
    public static int ceil_div(int x, int d) {
        return (x%d==0) ? x/d : x/d+1;
    }
    
    public static int max(int x, int y) {
        return (x>y) ? x : y;
    }
    
    public static int min(int x, int y) {
        return (x<y) ? x : y;
    }
    
    public static double[][] convMat(double[] input, double[] kernel, int p, int s, int d, int a, boolean transpose) {
        int i = input.length;
        int k = kernel.length;
        
        if(transpose) {
            return convMat(input, rot(kernel), k-p-1, d, s, (i-k+2*p)%s);
        }
        
        else {
            return convMat(input, kernel, p, s, d, a);
        }
    }
    
    public static double[][] convMat(double[] input, double[] kernel, int p, int s, int d, int a) {
        assert(a<d) : "bad input";
        int i = (input.length-1)*(d-1)+input.length+a;
        int k = kernel.length;
        int o = (i+2*p-k)/s + 1;
        double[][] convMat = new double[o][input.length];
        for(int x=0; x<o; x++) {
            int v = p-x*s;
            for(int y=Math.max(0,v); y<Math.min(k,i+v); y++) {
                if((y-v)%d==0) {
                    convMat[x][(y-v)/d] = kernel[y];
                }
            }
        }
        return convMat;
    }

    //combine with matConv to calculate indexing using dot product
    public static double[] mul(double[][] a, double[] b, boolean transpose) {
        int m;
        int n;
        
        if(transpose) {
            assert(a.length == b.length) : "wrong dim";
            m = a[0].length;
            n = b.length;
        }
        
        else {
            assert(a[0].length == b.length) : "wrong dim";
            m = a.length;
            n = b.length;
        }

        double[] c = new double[m];
        for(int i=0; i<m; i++) {
            for(int j=0; j<n; j++) {
                if(transpose) {
                    c[i] += a[j][i]*b[j];
                }
                
                else {
                    c[i] += a[i][j]*b[j];
                }
            }
        }  
        return c;
    }
    
    public static double[] rand(int n) {
        double[] rand = new double[n];
        for(int i=0; i<n; i++) {
            rand[i] = Math.random();
        }
        return rand;
    }
    
    public static void print(double[][] a) {        
        for(int i=0; i<a.length; i++) {
            for(int j=0; j<a[i].length; j++) {
                System.out.printf("%.4f ",a[i][j]);
            }
            System.out.println();
        }
        System.out.println();
    }
    
    public static void print(double[] input) {        
        for(int i=0; i<input.length; i++) {
            System.out.printf("%.4f ",input[i]);
        }
        System.out.println();
    }
}