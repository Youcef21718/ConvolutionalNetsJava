import java.lang.Math;
public class myPool2
{
    public static void main(String[] args) {
        int i=6; int k=2; int p=1; int s=1; int d=3; int a=2;
        
        int[] switches = switches(i, k, p, s, d, a);
        
        double[] input = rand(i);
        double[] output = pool(input, switches, k, p, s, d, a);
                
        double[][] poolMat = poolMat(input, k, p, s, d, a);
        
        print(output);
        print(mul(poolMat, input, false));
        
        //print(input);
        print(mul(poolMat, output, true));
        print(trans_pool(output, switches, i));
    }
    
    //CHANGED = TO +=, NOW WORKS!!!!!!!!
    public static double[] trans_pool(double[] output, int[] switches, int i) {
        int o = output.length;
        double[] input = new double[i];
        for(int x=0; x<o; x++) {
            //if(switches[x] == -1) {
            //}
            
            //else {
                int y = switches[x];
                input[y] += output[x];
            //}
        }
        return input;
    }
    
    public static double[][] poolMat(double[] input, int k, int p, int s, int d, int a) {
        assert(a<d) : "wrong params";
        
        int i = (input.length-1)*(d-1)+input.length+a;
        int o = (i+2*p-k)/s + 1;
        double[][] poolMat = new double[o][input.length];
        for(int x=0; x<o; x++) {
            int v = x*s-p;
            int maxIndex = 0;
            double max = Double.NEGATIVE_INFINITY;
            for(int y=ceil_div(max(v,0),d); y*d<min(k+v,i); y++) {
                if(max<input[y]) {
                    maxIndex = y;
                    max = input[y];
                }
            }
            
            if(max == Double.NEGATIVE_INFINITY) {
                poolMat[x][maxIndex] = 0;
            }
            else {
                poolMat[x][maxIndex] = 1;
            }
        }
        return poolMat;
    }
    
    public static double[] pool(double[] input, int[] switches, int k, int p, int s, int d, int a) {
        assert(a<d) : "wrong params";
        
        int i = (input.length-1)*(d-1)+input.length+a;
        int o = (i+2*p-k)/s + 1;        
        double[] output = new double[o];
        for(int x=0; x<o; x++) {
            int v = x*s-p;
            int maxIndex = 0;
            double max = Double.NEGATIVE_INFINITY;
            for(int y=ceil_div(max(v,0),d); y*d<min(k+v,i); y++) {
                if(max<input[y]) {
                    maxIndex = y;
                    max = input[y];
                }
            }
            
            if(max == Double.NEGATIVE_INFINITY) {
                output[x] = 0;
                //switches[x] = -1;
            }
            else {
                output[x] = max;
                switches[x] = maxIndex;
            }
        }
        return output;
    }
    
    public static int[] switches(int i, int k, int p, int s, int d, int a) {
        assert(a<d) : "wrong params";
        
        i = (i-1)*(d-1)+i+a;
        int o = (i+2*p-k)/s + 1;
        int[] switches = new int[o];
        return switches;
    }
    
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