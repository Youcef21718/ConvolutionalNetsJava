import java.lang.Math;
import java.util.Random;
public class conv5
{
    public static void main(String[] args) {       
        double epsilon = (double)0.00000001;

        double[][][][] input = random(2,3,10,10);
        double[][][][] label = random(2,2,6,6);
        
        double[] biases1 = random(2);
        double[][][][] kernel1 = random(3,2,2,2);
        
        double[] biases2 = random(2);
        double[][][][] kernel2 = random(2,2,3,3);

        System.out.println(networkCost(input, label, biases1, kernel1, biases2, kernel2, true));
        System.out.println("*********************************************************************");
        
        double[] gradBiases1 = clone(biases1);
        for(int i=0; i<gradBiases1.length; i++) {
            double[] newBias1_1 = eps(biases1,i,epsilon);
            double newCost1 = networkCost(input, label, newBias1_1, kernel1, biases2, kernel2, false);
            
            double[] newBias1_2 = eps(biases1,i,-epsilon);
            double newCost2 = networkCost(input, label, newBias1_2, kernel1, biases2, kernel2, false);
            
            gradBiases1[i] = (newCost1 - newCost2)/(2*epsilon);
        }
        print(gradBiases1);
        System.out.println("*********************************************************************");

        double[] gradBiases2 = clone(biases2);
        for(int i=0; i<gradBiases2.length; i++) {
            double[] newBias2_1 = eps(biases2,i,epsilon);
            double newCost1 = networkCost(input, label, biases1, kernel1, newBias2_1, kernel2, false);
            
            double[] newBias2_2 = eps(biases2,i,-epsilon);
            double newCost2 = networkCost(input, label, biases1, kernel1, newBias2_2, kernel2, false);
            
            gradBiases2[i] = (newCost1 - newCost2)/(2*epsilon);
        }       
        print(gradBiases2);
        System.out.println("*********************************************************************");
    }
    
    public static double[] col(double a[][][][])
    {
        double[] x = new double[a[0].length];
        
        for(int j=0; j<a[0].length; j++) {
            for(int i=0; i<a.length; i++) {
                for(int k=0; k<a[0][0].length; k++) {
                    for(int l=0; l<a[0][0][0].length; l++) {
                        x[j] += a[i][j][k][l];
                    }
                }
            }
        }
        
        return x;
    }
        
    public static double[][][][] hadMul(double a[][][][], double b[][][][])
    {
        assert(a.length==b.length && a[0].length==b[0].length && a[0][0].length==b[0][0].length && a[0][0][0].length==b[0][0][0].length) : "wrong size";
        double[][][][] x = new double[a.length][a[0].length][a[0][0].length][a[0][0][0].length];
        
        for(int i=0; i<a.length; i++){
            for(int j=0; j<a[0].length; j++){
                for(int k=0; k<a[0][0].length; k++){
                    for(int l=0; l<a[0][0][0].length; l++){
                        x[i][j][k][l] = a[i][j][k][l]*b[i][j][k][l];
                    }
                }   
            }
        }
        
        return x;
    }
    
    public static double[][][][] sub(double a[][][][], double b[][][][])
    {
        assert(a.length==b.length && a[0].length==b[0].length && a[0][0].length==b[0][0].length && a[0][0][0].length==b[0][0][0].length) : 
        a.length+" "+a[0].length+" "+a[0][0].length+" "+a[0][0][0].length;
        double[][][][] x = new double[a.length][a[0].length][a[0][0].length][a[0][0][0].length];
        
        for(int i=0; i<a.length; i++){
            for(int j=0; j<a[0].length; j++){
                for(int k=0; k<a[0][0].length; k++){
                    for(int l=0; l<a[0][0][0].length; l++){
                        x[i][j][k][l] = a[i][j][k][l]-b[i][j][k][l];
                    }
                }   
            }
        }
        
        return x;
    }
    
    public static double[][][][] dsigmoid(double a[][][][], double b[]) {
        assert(a[0].length == b.length) : "wrong size";
        double[][][][] x = new double[a.length][a[0].length][a[0][0].length][a[0][0][0].length];
        
        for(int i=0; i<a.length; i++) {
            for(int j=0; j<a[0].length; j++) {
                for(int k=0; k<a[0][0].length; k++) {
                    for(int l=0; l<a[0][0][0].length; l++) {
                        double sum = a[i][j][k][l] + b[j];
                        double sig = (double)(1/(1+Math.exp(-sum)));
                        x[i][j][k][l] = sig*(1-sig);
                    }
                }
            }
        }
        
        return x;
    }
    
    public static double[][][][] sigmoid(double a[][][][], double b[]) {
        assert(a[0].length == b.length) : "wrong size";
        double[][][][] x = new double[a.length][a[0].length][a[0][0].length][a[0][0][0].length];
        
        for(int i=0; i<a.length; i++) {
            for(int j=0; j<a[0].length; j++) {
                for(int k=0; k<a[0][0].length; k++) {
                    for(int l=0; l<a[0][0][0].length; l++) {
                        double sum = a[i][j][k][l] + b[j];
                        double sig = (double)(1/(1+Math.exp(-sum)));
                        x[i][j][k][l] = sig;
                    }
                }
            }
        }
        
        return x;
    }

    public static double[][][][] eps(double[][][][] a, int a1, int a2, int a3, int a4, double epsilon) {
        double[][][][] x = clone(a);
        for(int i=0; i<a.length; i++) {
            for(int j=0; j<a[0].length; j++) {
                for(int k=0; k<a[0][0].length; k++) {
                    for(int l=0; l<a[0][0][0].length; l++) {
                        x[i][j][k][l] = a[i][j][k][l];
                    }
                }
            }
        }
        x[a1][a2][a3][a4] = a[a1][a2][a3][a4] + epsilon;
        return x;
    }
    
    public static double[] eps(double[] a, int a1, double epsilon) {
        double[] x = clone(a);
        for(int i=0; i<a.length; i++) {
            x[i] = a[i];
        }
        x[a1] = a[a1] + epsilon;
        return x;
    }
    
    public static double[][][][] clone(double[][][][] a) {
        double[][][][] x = new double[a.length][a[0].length][a[0][0].length][a[0][0][0].length];
        return x;
    }
    
    public static double[] clone(double[] a) {
        double[] x = new double[a.length];
        return x;
    }
    
    public static double networkCost(double[][][][] input, double[][][][] label, double[] biases1, double[][][][] kernel1, double[] biases2, double[][][][] kernel2, boolean print) {       
        //*********************************************************       
        int p1_1=1, p2_1=1;
        int s1_1=2, s2_1=2;

        double[][][][] output1 = conv(input,kernel1,p1_1,p2_1,s1_1,s2_1);
        double[][][][] sig_output1 = sigmoid(output1,biases1);
        
        //*********************************************************
        
        int p1_3=1, p2_3=1;
        int s1_3=1, s2_3=1;

        double[][][][] output2 = conv(sig_output1,kernel2,p1_3,p2_3,s1_3,s2_3);
        double[][][][] sig_output2 = sigmoid(output2,biases2);

        //*********************************************************
        
        if(print) {
            double[][][][] gB2 = hadMul(sub(sig_output2, label),dsigmoid(output2,biases2));
            print(col(gB2));

            double[][][][] gB1 = hadMul(trans_conv(gB2,kernel2,p1_3,p2_3,s1_3,s2_3,0,0),dsigmoid(output1,biases1));
            print(col(gB1));
        }
        
        //*********************************************************
        double cost = cost(sig_output2, label);
        return cost;
    }
    
    public static double cost(double a[][][][], double b[][][][])
    {
        assert(a.length==b.length && a[0].length==b[0].length && a[0][0].length==b[0][0].length && a[0][0][0].length==b[0][0][0].length) : "wrong size";
        double x = 0;        
        for(int i=0; i<a.length; i++){
            for(int j=0; j<a[0].length; j++){
                for(int k=0; k<a[0][0].length; k++){
                    for(int l=0; l<a[0][0][0].length; l++){
                        x+= (a[i][j][k][l] - b[i][j][k][l])*(a[i][j][k][l] - b[i][j][k][l]);
                    }
                }
            }
        }
        
        x/= 2;
        return x;
    }

    public static double[][][][] trans_conv(double[][][][] input, double[][][][] kernel, int p1, int p2, int s1, int s2, int a1, int a2) {
        assert(input[0].length == kernel[0].length) : "input, kernel mismatch";
        
        int i1 = (input[0][0].length-1)*(s1-1)+input[0][0].length+a1;
        int i2 = (input[0][0][0].length-1)*(s2-1)+input[0][0][0].length+a2;
        
        int k1 = kernel[0][0].length;
        int k2 = kernel[0][0][0].length;
        
        p1 = k1-p1-1;
        p2 = k2-p2-1;
                
        final int s = 1;
        
        int o1 = (i1+2*p1-k1)/s + 1;
        int o2 = (i2+2*p2-k2)/s + 1;
        
        double[][][][] output = new double[input.length][kernel[0].length][o1][o2];
        for(int i=0; i<input.length; i++) {
            for(int j=0; j<kernel[0].length; j++) {
                for(int k=0; k<kernel.length; k++) {
                    trans_conv(input[i][k],kernel[j][k],output[i][j],p1,p2,s1,s2,a1,a1);
                }
            }
        }
        return output;
    }
    
    public static void trans_conv(double[][] input, double[][] kernel, double[][] output, int p1, int p2, int s1, int s2, int a1, int a2) {       
        int i1 = (input.length-1)*(s1-1)+input.length+a1;
        int i2 = (input[0].length-1)*(s2-1)+input[0].length+a2;
        
        int k1 = kernel.length;
        int k2 = kernel[0].length;
        
        p1 = k1-p1-1;
        p2 = k2-p2-1;
                
        final int s = 1;
        
        int o1 = (i1+2*p1-k1)/s + 1;
        int o2 = (i2+2*p2-k2)/s + 1;
        
        for(int i=0; i<o1; i++) {
            for(int j=0; j<o2; j++) {
                //convolution
                for(int k=Math.max(0,-pad(i*s,p1)); k<Math.min(k1,i1-pad(i*s,p1)); k++) {
                    for(int l=Math.max(0,-pad(j*s,p2)); l<Math.min(k2,i2-pad(j*s,p2)); l++) {
                        //i*s gives distance travelled after i strides
                        //k gives displacement due to kernel
                        //-p1 gives displacement due to padding
                        int x = pad(i*s+k,p1);
                        int y = pad(j*s+l,p2);

                        if(rot(x,i1)%s1==0 && rot(y,i2)%s2==0) {
                            output[rot(i,o1)][rot(j,o2)] += kernel[k][l]*input[dil(rot(x,i1),s1)][dil(rot(y,i2),s2)];
                        }
                    }
                }
            }
        }
    }
    
    public static double[][][][] conv(double[][][][] input, double[][][][] kernel, int p1, int p2, int s1, int s2) {
        assert(input[0].length == kernel.length) : "input, kernel mismatch";
        
        int i1 = input[0][0].length;
        int i2 = input[0][0][0].length;
        
        int k1 = kernel[0][0].length;
        int k2 = kernel[0][0][0].length;
                
        int o1 = (i1+2*p1-k1)/s1 + 1;
        int o2 = (i2+2*p2-k2)/s2 + 1;
        
        double[][][][] output = new double[input.length][kernel[0].length][o1][o2];
        for(int i=0; i<input.length; i++) {
            for(int j=0; j<kernel[0].length; j++) {
                for(int k=0; k<kernel.length; k++) {
                    conv(input[i][k],kernel[k][j],output[i][j],p1,p2,s1,s2);
                }
            }
        }
        return output;
    }
    
    public static void conv(double[][] input, double[][] kernel, double[][] output, int p1, int p2, int s1, int s2) {
        int i1 = input.length;
        int i2 = input[0].length;
        
        int k1 = kernel.length;
        int k2 = kernel[0].length;
                
        int o1 = (i1+2*p1-k1)/s1 + 1;
        int o2 = (i2+2*p2-k2)/s2 + 1;
        
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
    
    public static int dil(int i, int s) {
        return i/s;
    }
    
    public static int rot(int i, int x) {
        return x-i-1;
    }
        
    public static double[][][][] random(int b, int d, int i1, int i2) {
        double[][][][] a = new double[b][d][i1][i2];
        for(int i=0; i<b; i++) {
            for(int j=0; j<d; j++) {
                a[i][j] = random(i1, i2);
            }
        }
        return a;
    }
    
    public static double[][] random(int i1, int i2) {
        double[][] a = new double[i1][i2];
        for(int i=0; i<i1; i++){
            for(int j=0; j<i2; j++){
                Random r = new Random();
                a[i][j] = (double)(0.01*r.nextGaussian());
            }
        }
        return a;
    }
    
    public static double[] random(int k) {
        double[] a = new double[k];
        for(int i=0; i<k; i++){
            Random r = new Random();
            a[i] = (double)(r.nextGaussian());
        }
        return a;
    }
    
    public static void print(double[][][][] a) {        
        for(int i=0; i<a.length; i++) {
            for(int j=0; j<a[0].length; j++) {
                print(a[i][j]);
            }
            System.out.println();
        }
        System.out.println();
    }
    
    public static void print(double[][] a) {        
        for(int i=0; i<a.length; i++) {
            for(int j=0; j<a[0].length; j++) {
                //System.out.printf("%.8f ",a[i][j]);
                System.out.print(a[i][j]+" ");
            }
            System.out.println();
        }
        System.out.println();
    }
    
    public static void print(double[] a) {        
        for(int i=0; i<a.length; i++) {
            //System.out.printf("%.8f ",a[i]);
            System.out.print(a[i]+" ");
        }
        System.out.println();
    }
}
