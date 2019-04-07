import java.lang.Math;
import java.util.Random;
public class conv3
{
    public static void main(String[] args) {
        int digits = 10;
        
        int b = 1;
        int d = 3;
        int k = 2;
        
        int i1=5, i2=5;
        int k1=3, k2=3;
        int p1=1, p2=1;
        int s1=2, s2=2;
        
        int o1 = (i1+2*p1-k1)/s1 + 1;
        int o2 = (i2+2*p2-k2)/s2 + 1;
        
        float[][][][] input = 
        {{       
            {{0,0,2,2,1},{1,1,2,1,0},{0,1,0,2,0},{0,2,0,1,2},{1,1,0,2,0}},
            {{1,2,1,1,0},{0,2,2,0,1},{0,1,2,1,0},{1,2,1,2,0},{1,0,0,0,0}},
            {{1,2,0,2,1},{1,0,1,1,2},{2,1,1,2,0},{2,2,1,0,0},{0,2,1,2,0}},
        }};

        float[][][][] kernel =
        {
            {{{0,-1,-1},{0,-1,0},{1,1,0}},{{1,1,0},{-1,-1,1},{-1,-1,-1}}},  
            {{{0,-1,0},{-1,-1,0},{-1,1,1}},{{0,0,0},{1,1,0},{1,-1,-1}}},
            {{{1,1,1},{1,0,1},{-1,0,-1}},{{1,-1,1},{-1,0,-1},{0,1,1}}},
        };

        float[] biases = 
        {1,0};

        //float[][][][] input = random(b,d,i1,i2);
        print(input);
        
        System.out.println("*********************************");
        
        //float[][][][] kernel = random(d,k,k1,k2);
        print(kernel);
        
        System.out.println("*********************************");
        
        //float[] biases = random(k);
        print(biases);
        
        System.out.println("*********************************");
        
        float[][][][] output = new float[b][k][o1][o2];
        conv(input,kernel,output,p1,p2,s1,s2);
        print(output);

        System.out.println("*********************************");
        
        float[][][][] sigmoidOutput = sigmoid(output,biases);
        print(sigmoidOutput);
        
        System.out.println("*********************************");
        
        i1=o1; i2=o2;
        k1=2; k2=2;
        p1=1; p2=1;
        s1=1; s2=1;
        
        o1 = (i1+2*p1-k1)/s1 + 1;
        o2 = (i2+2*p2-k2)/s2 + 1;
               
        int[][][][][] switches = new int[b][k][o1][o2][2];
        float[][][][] pool_output = new float[b][k][o1][o2];
        pool(sigmoidOutput,pool_output,switches,k1,k2,p1,p2,s1,s2);
        print(pool_output);
        
        System.out.println("*********************************");
        
        float[][] mat = mat(pool_output);
        print(mat);
        
        System.out.println("*********************************");
        
        float[][] weights = random(mat[0].length,digits);
        print(weights);
        
        System.out.println("*********************************");
        
        float[] finalBiases = random(digits);
        print(finalBiases);
        
        System.out.println("*********************************");
        
        float[][] finalOutput = mul(mat, weights, true, true);
        print(finalOutput);
        
        System.out.println("*********************************");
        
        float[][] finalOutputSig = sigmoid(finalOutput, finalBiases);
        print(finalOutputSig);
        
        System.out.println("*********************************");
        
        float[][] label = random(finalOutputSig.length, digits);
        print(label);
        
        System.out.println("*********************************");
        
        float cost = cost(finalOutputSig, label);
        System.out.println(cost);
    }
    
    public static float cost(float a[][], float b[][])
    {
        float x = 0;
        
        for(int i=0; i<a.length; i++){
            for(int j=0; j<a[0].length; j++){
                x+= (a[i][j] - b[i][j])*(a[i][j] - b[i][j]);
            }
        }
        
        x/= (2*a.length);
        return x;
    }
    
    public static float[][][][] sigmoid(float a[][][][], float b[]) {
        float[][][][] x = new float[a.length][a[0].length][a[0][0].length][a[0][0][0].length];
        
        for(int i=0; i<a.length; i++) {
            for(int j=0; j<a[0].length; j++) {
                for(int k=0; k<a[0][0].length; k++) {
                    for(int l=0; l<a[0][0][0].length; l++) {
                        x[i][j][k][l] = (float)(Math.exp(a[i][j][k][l])*Math.exp(b[j])/(Math.exp(a[i][j][k][l])*Math.exp(b[j])+1));
                    }
                }
            }
        }
        
        return x;
    }
    
    public static float[][] sigmoid(float a[][], float b[])
    {
        float[][] x = new float[a.length][a[0].length];
        
        for(int i=0; i<a.length; i++){
            for(int j=0; j<a[0].length; j++){
                x[i][j] = (float)(Math.exp(a[i][j])*Math.exp(b[j])/(Math.exp(a[i][j])*Math.exp(b[j])+1));
            }
        }
        
        return x;
    }
    
    public static float[][] mul(float[][] a, float[][] b, boolean left, boolean right) {
        int m1,n1;
        int m2,n2;
        
        float[][] c;
        
        if(left) {
            if(right) {
                m1 = a.length;
                n1 = a[0].length;
                m2 = b.length;
                n2 = b[0].length;
                
                c = new float[m1][n2];
                assert(n1 == m2): "wrong matrix dimension";
                for(int i=0; i<m1; i++) {
                    for(int j=0; j<n2; j++) {
                        for(int k=0; k<n1; k++) {
                            c[i][j] += a[i][k]*b[k][j];
                        }
                    }
                }
            }
            
            else {
                m1 = a.length;
                n1 = a[0].length;
                m2 = b[0].length;
                n2 = b.length;
                
                c = new float[m1][n2];
                assert(n1 == m2): "wrong matrix dimension";
                for(int i=0; i<m1; i++) {
                    for(int j=0; j<n2; j++) {
                        for(int k=0; k<n1; k++) {
                            c[i][j] += a[i][k]*b[j][k];
                        }
                    }
                }
            }
        }
        
        else {
            if(right) {
                m1 = a[0].length;
                n1 = a.length;
                m2 = b.length;
                n2 = b[0].length;
                
                c = new float[m1][n2];
                assert(n1 == m2): "wrong matrix dimension";
                for(int i=0; i<m1; i++) {
                    for(int j=0; j<n2; j++) {
                        for(int k=0; k<n1; k++) {
                            c[i][j] += a[k][i]*b[k][j];
                        }
                    }
                }
            }
            
            else {
                m1 = a[0].length;
                n1 = a.length;
                m2 = b[0].length;
                n2 = b.length;
                
                c = new float[m1][n2];
                assert(n1 == m2): "wrong matrix dimension";
                for(int i=0; i<m1; i++) {
                    for(int j=0; j<n2; j++) {
                        for(int k=0; k<n1; k++) {
                            c[i][j] += a[k][i]*b[j][k];
                        }
                    }
                }
            }
        }

        return c;
    }
    
    public static float[][] mat(float[][][][] input) {
        int i1 = input.length;
        int i2 = input[0].length*input[0][0].length*input[0][0][0].length;
        float[][] mat = new float[i1][i2];
        for(int i=0; i<input.length; i++) {
            int index = 0;
            for(int j=0; j<input[0].length; j++) {
                for(int k=0; k<input[0][0].length; k++) {
                    for(int l=0; l<input[0][0][0].length; l++) {
                        mat[i][index] = input[i][j][k][l];
                        index++;
                    }
                }
            }
        }
        return mat;
    }
    
    public static void pool(float[][][][] input, float[][][][] output, int[][][][][] switches, int k1, int k2, int p1, int p2, int s1, int s2) {
        assert(switches.length == output.length && switches[0].length == output[0].length) : "switches, output mismatch";        
        for(int i=0; i<input.length; i++) {
            for(int j=0; j<input[0].length; j++) {
                pool(input[i][j],output[i][j],switches[i][j],k1,k2,p1,p2,s1,s2);
            }
        }
    }
    
    public static void pool(float[][] input, float[][] output, int[][][] switches, int k1, int k2, int p1, int p2, int s1, int s2) {
        int i1 = input.length;
        int i2 = input[0].length;
        
        int o1 = (i1+2*p1-k1)/s1 + 1;
        int o2 = (i2+2*p2-k2)/s2 + 1;
        
        assert(k1>p1 && k2>p2): "kernel, padding not compatible";
        for(int i=0; i<o1; i++) {
            for(int j=0; j<o2; j++) {
                //pooling
                int maxInputX = 0;
                int maxInputY = 0;
                float max = Float.NEGATIVE_INFINITY;
                for(int k=Math.max(0,-pad(i*s1,p1)); k<Math.min(k1,i1-pad(i*s1,p1)); k++) {
                    for(int l=Math.max(0,-pad(j*s2,p2)); l<Math.min(k2,i2-pad(j*s2,p2)); l++) {
                        //i*s1 gives distance travelled after i strides
                        //k gives displacement due to kernel
                        //-p1 gives displacement due to padding
                        int x = pad(i*s1+k,p1);
                        int y = pad(j*s2+l,p2);

                        if(max<input[x][y]) {
                            maxInputX = x;
                            maxInputY = y;
                            max = input[x][y];
                        }
                    }
                }                
                //x*i2+y gives the index of the input
                //i*o2+j gives the index of the output
                output[i][j] = max;
                switches[i][j][0] = maxInputX;
                switches[i][j][1] = maxInputY;
            }
        }
        //return output;
    }
    
    public static void conv(float[][][][] input, float[][][][] kernel, float[][][][] output, int p1, int p2, int s1, int s2) {
        assert(input[0].length == kernel.length) : "input, kernel mismatch";
        for(int i=0; i<input.length; i++) {
            for(int j=0; j<kernel[0].length; j++) {
                for(int k=0; k<input[0].length; k++) {
                    conv(input[i][k],kernel[k][j],output[i][j],p1,p2,s1,s2);
                }
            }
        }
    }
    
    public static void conv(float[][] input, float[][] kernel, float[][] output, int p1, int p2, int s1, int s2) {
        int i1 = input.length;
        int i2 = input[0].length;
        
        int k1 = kernel.length;
        int k2 = kernel[0].length;
                
        int o1 = output.length;
        int o2 = output[0].length;
        
        assert(k1>p1 && k2>p2): "kernel, padding are not compatible";    
        for(int i=0; i<o1; i++) {
            for(int j=0; j<o2; j++) {
                //convolution
                for(int k=Math.max(0,-pad(i*s1,p1)); k<Math.min(k1,i1-pad(i*s1,p1)); k++) {
                    for(int l=Math.max(0,-pad(j*s2,p2)); l<Math.min(k2,i2-pad(j*s2,p2)); l++) {
                        //i*s1 gives distance travelled after i strides
                        //k gives displacement due to kernel
                        //-p1 gives displacement due to padding  
                        int x = pad(i*s1+k,p1);
                        int y = pad(j*s2+l,p2);

                        output[i][j] += kernel[k][l]*input[x][y];
                    }
                }
            }
        }
    }
    
    public static int pad(int i, int p) {
        return i-p;
    }
    
    public static float[][][][] random(int b, int d, int i1, int i2) {
        float[][][][] a = new float[b][d][i1][i2];
        for(int i=0; i<b; i++) {
            for(int j=0; j<d; j++) {
                a[i][j] = random(i1, i2);
            }
        }
        return a;
    }
    
    public static float[][] random(int i1, int i2) {
        float[][] a = new float[i1][i2];
        for(int i=0; i<i1; i++){
            for(int j=0; j<i2; j++){
                Random r = new Random();
                a[i][j] = (float)r.nextGaussian();
            }
        }
        return a;
    }
    
    public static float[] random(int k) {
        float[] a = new float[k];
        for(int i=0; i<k; i++){
            Random r = new Random();
            a[i] = (float)r.nextGaussian();
        }
        return a;
    }
    
    public static void print(float[][][][] a) {        
        for(int i=0; i<a.length; i++) {
            for(int j=0; j<a[0].length; j++) {
                print(a[i][j]);
            }
            System.out.println();
        }
        System.out.println();
    }
    
    public static void print(float[][] a) {        
        for(int i=0; i<a.length; i++) {
            for(int j=0; j<a[0].length; j++) {
                System.out.printf("%.4f ",a[i][j]);
            }
            System.out.println();
        }
        System.out.println();
    }
    
    public static void print(float[] a) {        
        for(int i=0; i<a.length; i++) {
            System.out.printf("%.4f ",a[i]);
        }
        System.out.println();
    }
}
