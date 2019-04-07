import java.lang.Math;
public class myPool
{
    public static void main(String[] args) {
        int i=6; int k=2; int p=1; int s=1;
        
        int[] switches = switches(i, k, p, s);
        
        double[] input = rand(i);
        double[] output = pool(input, switches, k, p, s);
        
        double[][] poolMat = poolMat(input, k, p, s);
        
        print(output);
        print(mul(poolMat, input, false));
        
        print(input);
        print(mul(poolMat, output, true));
        print(trans_pool(output, switches, i));
    }
    
    //CHANGED = TO +=, NOW WORKS!!!!!!!!
    public static double[] trans_pool(double[] output, int[] switches, int i) {
        int o = output.length;
        double[] input = new double[i];
        for(int x=0; x<o; x++) {
            int y = switches[x];
            input[y] += output[x];
        }
        return input;
    }
    
    public static double[][] poolMat(double[] input, int k, int p, int s) {
        int i = input.length;
        int o = (i+2*p-k)/s + 1;
        double[][] poolMat = new double[o][i];
        for(int x=0; x<o; x++) {
            int v = x*s-p;
            int maxIndex = 0;
            double max = Double.NEGATIVE_INFINITY;
            for(int y = max(v,0); y<min(k+v,i); y++) {
                if(max<input[y]) {
                    maxIndex = y;
                    max = input[y];
                }
            }
            poolMat[x][maxIndex] = 1;
        }
        return poolMat;
    }
    
    public static double[] pool(double[] input, int[] switches, int k, int p, int s) {
        int i = input.length;        
        int o = (i+2*p-k)/s + 1;        
        double[] output = new double[o];
        for(int x=0; x<o; x++) {
            int v = x*s-p;
            int maxIndex = 0;
            double max = Double.NEGATIVE_INFINITY;
            for(int y = max(v,0); y<min(k+v,i); y++) {
                if(max<input[y]) {
                    maxIndex = y;
                    max = input[y];
                }
            }
            output[x] = max;
            switches[x] = maxIndex;
        }
        return output;
    }
    
    public static int[] switches(int i, int k, int p, int s) {
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