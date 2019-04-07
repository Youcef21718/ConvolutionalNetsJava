import java.lang.Math;
public class myConv6
{
    public static void main(String[] args) {
        int i=13; int k=4; int p=3; int s=2; int d=3; int a=1;
        
        double[] input = rand(i);
        double[] kernel = rand(k);
        
        //output is convolution of input with kernel
        double[] output = conv(input, kernel, p, s, d, a, false);
        
        //define convolution matrix and transpose-convolution matrix
        double[][] mat = convMat(input, kernel, p, s, d, a, false);
        double[][] mat_trans = convMat(output, kernel, p, s, d, a, true);
        
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
            int v = x*s-p;
            for(int y=ceil_div(max(v,0),d); y*d<min(k+v,i); y++) {
                convMat[x][y] = kernel[y*d-v];
            }
        }
        return convMat;
    }
    
    public static double[] matConv(double[][] convMat, double[] input, int k, int p, int s, int d, int a, boolean transpose) {
        int i = input.length;
        double[] kernel = new double[k];
        if(transpose) {
            return matConv(convMat, input, kernel, k-p-1, d, s, (i-k+2*p)%s, transpose);
        }      
        else {
            return matConv(convMat, input, kernel, p, s, d, a, transpose);
        }
    }
    
    ///in reality you would add values corresponding to gradients with += instead of =
    public static double[] matConv(double[][] convMat, double[] input, double[] kernel, int p, int s, int d, int a, boolean transpose) {
        assert(a<d) : "bad input";
        int i = (input.length-1)*(d-1)+input.length+a;
        int k = kernel.length;
        int o = (i+2*p-k)/s + 1;
        for(int x=0; x<o; x++) {
            int v = x*s-p;
            for(int y=ceil_div(max(v,0),d); y*d<min(k+v,i); y++) {
                kernel[transpose ? rot(y*d-v,k) : y*d-v] = convMat[x][y];
            }
        }
        return kernel;
    }
    
    //combine with matConv to calculate indexing using dot product
    public static double[] mul(double[][] a, double[] b, boolean transpose) {
        if(transpose) {
            assert(a.length == b.length) : "wrong dim";
            double[] c = new double[a[0].length];
            for(int i=0; i<c.length; i++) {
                for(int j=0; j<b.length; j++) {
                    c[i] += a[j][i]*b[j];
                }
            } 
            return c;
        }
        else {
            assert(a[0].length == b.length) : "wrong dim";
            double[] c = new double[a.length];
            for(int i=0; i<c.length; i++) {
                for(int j=0; j<b.length; j++) {
                    c[i] += a[i][j]*b[j];
                }
            } 
            return c;
        }
    }
    
    public static int ceil_div(int x, int d) {
        return (x%d==0) ? x/d : x/d+1;
    }
    
    public static double[] rot(double[] input) {
        int i = input.length;
        double[] rot = new double[i];
        for(int x=0; x<i; x++) {
            rot[i-x-1] = input[x];
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
    
    public static double[] rand(int n) {
        double[] rand = new double[n];
        for(int i=0; i<n; i++) {
            rand[i] = Math.random();
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
    
    public static void print(double[] input) {        
        for(int i=0; i<input.length; i++) {
            System.out.printf("%.4f ",input[i]);
        }
        System.out.println();
    }
}