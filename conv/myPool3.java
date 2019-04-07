import java.lang.Math;
public class myPool3
{
    public static void main(String[] args) {
        int[] i = new int[]{6,6}; int[] k = new int[]{2,2};
        int[] p = new int[]{1,1}; int[] s = new int[]{1,1};
        int[] d = new int[]{3,3}; int[] a = new int[]{2,2};
                
        int[][][] switches = switches(i, k, p, s, d, a);
        
        double[][] input = rand(i);
        double[][] output = pool(input, switches, k, p, s, d, a);
                
        double[][][][] poolMat = poolMat(input, k, p, s, d, a);
        
        print(output);
        print(mul(poolMat, input, false));
        
        //print(input);
        print(mul(poolMat, output, true));
        print(trans_pool(output, switches, i));
    }
    
    //CHANGED = TO +=, NOW WORKS!!!!!!!!
    public static double[][] trans_pool(double[][] output, int[][][] switches, int[] i) {
        int[] o = shape(output);
        double[][] input = new double[i[0]][i[1]];
        for(int x0=0; x0<o[0]; x0++) {
            for(int x1=0; x1<o[1]; x1++) {
                //if(switches[x0][x1][0] == -1 && switches[x0][x1][1] == -1) {
                //}
            
                //else {
                    int y0 = switches[x0][x1][0];
                    int y1 = switches[x0][x1][1];
                    input[y0][y1] += output[x0][x1];
                //}
            }
        }
        return input;
    }
    
    public static double[][][][] poolMat(double[][] input, int[] k, int[] p, int[] s, int[] d, int[] a) {
        assert(a[0]<d[0] && a[1]<d[1]) : "bad input";
        
        int[] ishape = shape(input);
        
        int i0 = (ishape[0]-1)*(d[0]-1)+ishape[0]+a[0];
        int k0 = k[0];
        int o0 = (i0+2*p[0]-k0)/s[0]+1;
        
        int i1 = (ishape[1]-1)*(d[1]-1)+ishape[1]+a[1];
        int k1 = k[1];
        int o1 = (i1+2*p[1]-k1)/s[1]+1;

        double[][][][] poolMat = new double[o0][o1][ishape[0]][ishape[1]];       
        for(int x0=0; x0<o0; x0++) {
            int v0 = x0*s[0]-p[0];
            for(int x1=0; x1<o1; x1++) {
                int v1 = x1*s[1]-p[1];               
                int maxIndex0 = 0;
                int maxIndex1 = 0;
                double max = Double.NEGATIVE_INFINITY;
                for(int y0=ceil_div(max(v0,0),d[0]); y0*d[0]<min(k0+v0,i0); y0++) {
                    for(int y1=ceil_div(max(v1,0),d[1]); y1*d[1]<min(k1+v1,i1); y1++) {
                        if(max<input[y0][y1]) {
                            maxIndex0 = y0;
                            maxIndex1 = y1;
                            max = input[y0][y1];
                        }
                    }
                }                
                if(max == Double.NEGATIVE_INFINITY) {
                    poolMat[x0][x1][maxIndex0][maxIndex1] = 0;
                }
                else {
                    poolMat[x0][x1][maxIndex0][maxIndex1] = 1;
                }
            }
        }
        return poolMat;
    }
    
    public static double[][] pool(double[][] input, int[][][] switches, int[] k, int[] p, int[] s, int[] d, int[] a) {
        assert(a[0]<d[0] && a[1]<d[1]) : "bad input";
        
        int[] ishape = shape(input);
        
        int i0 = (ishape[0]-1)*(d[0]-1)+ishape[0]+a[0];
        int k0 = k[0];
        int o0 = (i0+2*p[0]-k0)/s[0]+1;
        
        int i1 = (ishape[1]-1)*(d[1]-1)+ishape[1]+a[1];
        int k1 = k[1];
        int o1 = (i1+2*p[1]-k1)/s[1]+1;
        
        double[][] output = new double[o0][o1];        
        for(int x0=0; x0<o0; x0++) {
            int v0 = x0*s[0]-p[0];
            for(int x1=0; x1<o1; x1++) {
                int v1 = x1*s[1]-p[1];                
                int maxIndex0 = 0;
                int maxIndex1 = 0;
                double max = Double.NEGATIVE_INFINITY;
                for(int y0=ceil_div(max(v0,0),d[0]); y0*d[0]<min(k0+v0,i0); y0++) {
                    for(int y1=ceil_div(max(v1,0),d[1]); y1*d[1]<min(k1+v1,i1); y1++) {
                        if(max<input[y0][y1]) {
                            maxIndex0 = y0;
                            maxIndex1 = y1;
                            max = input[y0][y1];
                        }
                    }
                }
                if(max == Double.NEGATIVE_INFINITY) {
                    output[x0][x1] = 0;
                    //switches[x0][x1][0] = -1;
                    //switches[x0][x1][1] = -1;
                }
                else {
                    output[x0][x1] = max;
                    switches[x0][x1][0] = maxIndex0;
                    switches[x0][x1][1] = maxIndex1;
                }
            }
        }
        return output;
    }
    
    public static int[][][] switches(int[] i, int[] k, int[] p, int[] s, int[] d, int[] a) {
        int i0 = (i[0]-1)*(d[0]-1)+i[0]+a[0];
        int i1 = (i[1]-1)*(d[1]-1)+i[1]+a[1];
        
        int o0 = (i0+2*p[0]-k[0])/s[0] + 1;
        int o1 = (i1+2*p[1]-k[1])/s[1] + 1;
        
        int[][][] switches = new int[o0][o1][2];
        return switches;
    }
    
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