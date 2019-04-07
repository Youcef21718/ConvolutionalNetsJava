import java.lang.Math;
public class myConv2
{
    public static void main(String[] args) {
        int i = 6;
        int k = 3;
        int p = 1;
        int s = 2;
        
        int d = 2;
        int a = 1;
        
        double[] input = rand(i);
        double[] kernel = rand(k);
        double[] output = conv(dil(input,d,a), kernel, p, s);
        
        double[][] mat = convMat(dil(input,d,a), kernel, p, s);
        double[][] mat_trans = convMat(dil(output,s,(i-k+2*p)%s), rot(kernel), k-p-1, d);
        
        print(output);
        print(mul(mat, dil(input,d,a), false));
        
        print(mul(mat_trans, dil(output,s,(i-k+2*p)%s), false));
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
    
    public static double[] dil(double[] input, int d, int a) {
        int i = input.length;
        int o = (i-1)*(d-1)+i+a;   
        double[] dil = new double[o];
        for(int x=0; x<i; x++) {
            dil[x*d] = input[x];
        }
        return dil;
    }
    
    public static double[] pad(double[] input, int p) {        
        int i = input.length;
        int o = i+2*p;
        double[] pad = new double[o];
        for(int x=0; x<i; x++) {
            pad[i+p] = input[i];
        }
        return pad;
    }
    
    public static double[] rot(double[] input) {
        int i = input.length;
        double[] rot = new double[i];
        for(int x=0; x<i; x++) {
            rot[i-x-1] = input[x];
        }
        return rot;
    }
    
    public static double[] conv(double[] input, double[] kernel, int p, int s) {
        int i = input.length;
        int k = kernel.length;
        int o = (i+2*p-k)/s + 1;
        double[] output = new double[o];
        for(int x=0; x<o; x++) {
            int v = p-x*s;
            for(int y=Math.max(0,v); y<Math.min(k,i+v); y++) {
                output[x] += kernel[y]*input[y-v];
            }
        }
        return output;
    }
    
    public static double[][] convMat(double[] input, double[] kernel, int p, int s) {
        int i = input.length;
        int k = kernel.length;
        int o = (i+2*p-k)/s + 1;
        double[][] convMat = new double[o][i];
        for(int x=0; x<o; x++) {
            int v = p-x*s;
            for(int y=Math.max(0,v); y<Math.min(k,i+v); y++) {                
                convMat[x][y-v] = kernel[y];
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