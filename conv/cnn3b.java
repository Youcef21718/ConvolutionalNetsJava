import java.lang.Math;
import java.util.Random;
public class cnn3b
{
    public static void main(String[] args) {       
        double epsilon = (double)0.0000001;
        
        double[][][][] input = random(2,3,10,10);
        double[][] label = random(2,10);
        
        double[] biases1 = random(2);
        double[][][][] kernel1 = random(3,2,2,2);
        
        double[] biases2 = random(18);
        double[][] weights2 = random(32,18);
        
        double[] biases3 = random(10);
        double[][] weights3 = random(18,10);

        networkCost(input, label, biases1, kernel1, biases2, weights2, biases3, weights3, true);
        
        System.out.println("*********************************************************************");
        
        double[] gradBiases1 = clone(biases1);
        for(int i=0; i<gradBiases1.length; i++) {
            biases1[i] += epsilon;
            double newCost1 = networkCost(input, label, biases1, kernel1, biases2, weights2, biases3, weights3, false);
            biases1[i] -= epsilon;
            
            biases1[i] -= epsilon;
            double newCost2 = networkCost(input, label, biases1, kernel1, biases2, weights2, biases3, weights3, false);
            biases1[i] += epsilon;
            
            gradBiases1[i] = (newCost1 - newCost2)/(2*epsilon);
        }
        print(gradBiases1);
        System.out.println("*********************************************************************");
        
        double[] gradBiases2 = clone(biases2);
        for(int i=0; i<gradBiases2.length; i++) {
            biases2[i] += epsilon;
            double newCost1 = networkCost(input, label, biases1, kernel1, biases2, weights2, biases3, weights3, false);
            biases2[i] -= epsilon;
            
            biases2[i] -= epsilon;
            double newCost2 = networkCost(input, label, biases1, kernel1, biases2, weights2, biases3, weights3, false);
            biases2[i] += epsilon;
            
            gradBiases2[i] = (newCost1 - newCost2)/(2*epsilon);
        }       
        print(gradBiases2);
        System.out.println("*********************************************************************");
        
        double[] gradBiases3 = clone(biases3);
        for(int i=0; i<gradBiases3.length; i++) {
            biases3[i] += epsilon;
            double newCost1 = networkCost(input, label, biases1, kernel1, biases2, weights2, biases3, weights3, false);
            biases3[i] -= epsilon;
            
            biases3[i] -= epsilon;
            double newCost2 = networkCost(input, label, biases1, kernel1, biases2, weights2, biases3, weights3, false);
            biases3[i] += epsilon;
            
            gradBiases3[i] = (newCost1 - newCost2)/(2*epsilon);
        }       
        print(gradBiases3);
        System.out.println("*********************************************************************");

        double[][][][] gradKernel1 = clone(kernel1);
        for(int i=0; i<gradKernel1.length; i++) {
            for(int j=0; j<gradKernel1[0].length; j++) {
                for(int k=0; k<gradKernel1[0][0].length; k++) {
                    for(int l=0; l<gradKernel1[0][0][0].length; l++) {
                        kernel1[i][j][k][l] += epsilon;
                        double newCost1 = networkCost(input, label, biases1, kernel1, biases2, weights2, biases3, weights3, false);
                        kernel1[i][j][k][l] -= epsilon;
                        
                        kernel1[i][j][k][l] -= epsilon;
                        double newCost2 = networkCost(input, label, biases1, kernel1, biases2, weights2, biases3, weights3, false);
                        kernel1[i][j][k][l] += epsilon;
                        
                        gradKernel1[i][j][k][l] = (newCost1 - newCost2)/(2*epsilon);
                    }   
                }   
            }   
        }
        print(gradKernel1);
        System.out.println("*********************************************************************");
        
        double[][] gradWeights2 = clone(weights2);
        for(int i=0; i<gradWeights2.length; i++) {
            for(int j=0; j<gradWeights2[0].length; j++) {
                weights2[i][j] += epsilon;
                double newCost1 = networkCost(input, label, biases1, kernel1, biases2, weights2, biases3, weights3, false);
                weights2[i][j] -= epsilon;
                        
                weights2[i][j] -= epsilon;
                double newCost2 = networkCost(input, label, biases1, kernel1, biases2, weights2, biases3, weights3, false);
                weights2[i][j] += epsilon;
                        
                gradWeights2[i][j] = (newCost1 - newCost2)/(2*epsilon);
            }
        }
        print(gradWeights2);

        System.out.println("*********************************************************************");
        
        double[][] gradWeights3 = clone(weights3);
        for(int i=0; i<gradWeights3.length; i++) {
            for(int j=0; j<gradWeights3[0].length; j++) {
                weights3[i][j] += epsilon;
                double newCost1 = networkCost(input, label, biases1, kernel1, biases2, weights2, biases3, weights3, false);
                weights3[i][j] -= epsilon;
                        
                weights3[i][j] -= epsilon;
                double newCost2 = networkCost(input, label, biases1, kernel1, biases2, weights2, biases3, weights3, false);
                weights3[i][j] += epsilon;
                        
                gradWeights3[i][j] = (newCost1 - newCost2)/(2*epsilon);
            }
        }
        print(gradWeights3);
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
    
    public static double[] col(double a[][])
    {
        double[] x = new double[a[0].length];
        
        for(int j=0; j<a[0].length; j++) {
            for(int i=0; i<a.length; i++) {
                x[j] += a[i][j];
            }
        }
        
        return x;
    }
        
    public static double[][][][] hadMul(double a[][][][], double b[][][][])
    {
        assert(a.length==b.length && a[0].length==b[0].length && a[0][0].length==b[0][0].length && a[0][0][0].length==b[0][0][0].length) : "wrong size";
        double[][][][] x = clone(a);
        
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
    
    public static double[][] hadMul(double a[][], double b[][])
    {
        assert(a.length==b.length && a[0].length==b[0].length) : "wrong size";
        double[][] x = clone(a);
        
        for(int i=0; i<a.length; i++){
            for(int j=0; j<a[0].length; j++){
                x[i][j] = a[i][j]*b[i][j];
            }
        }
        
        return x;
    }
    
    public static double[][][][] sub(double a[][][][], double b[][][][])
    {
        assert(a.length==b.length && a[0].length==b[0].length && a[0][0].length==b[0][0].length && a[0][0][0].length==b[0][0][0].length) : "wrong size";
        double[][][][] x = clone(a);
        
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
    
    public static double[][] sub(double a[][], double b[][])
    {
        assert(a.length==b.length && a[0].length==b[0].length) : "wrong size";
        double[][] x = clone(a);
        
        for(int i=0; i<a.length; i++){
            for(int j=0; j<a[0].length; j++){
                x[i][j] = a[i][j]-b[i][j];   
            }
        }
        
        return x;
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
    
    public static double cost(double a[][], double b[][])
    {
        assert(a.length==b.length && a[0].length==b[0].length) : "wrong size";
        double x = 0;        
        for(int i=0; i<a.length; i++){
            for(int j=0; j<a[0].length; j++){
                x+= (a[i][j] - b[i][j])*(a[i][j] - b[i][j]);
            }
        }
        
        x/= 2;
        return x;
    }
    
    public static double[][][][] sigmoid(double a[][][][], double b[]) {
        assert(a[0].length == b.length) : "wrong size";
        double[][][][] x = clone(a);
        
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
    
    public static double[][] sigmoid(double a[][], double b[]) {
        assert(a[0].length == b.length) : "wrong size";
        double[][] x = clone(a);
        
        for(int i=0; i<a.length; i++) {
            for(int j=0; j<a[0].length; j++) {
                double sum = a[i][j] + b[j];
                double sig = (double)(1/(1+Math.exp(-sum)));
                x[i][j] = sig;
            }
        }
        
        return x;
    }
    
    public static double[][][][] dsigmoid(double a[][][][], double b[]) {
        assert(a[0].length == b.length) : "wrong size";
        double[][][][] x = clone(a);
        
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
    
    public static double[][] dsigmoid(double a[][], double b[]) {
        assert(a[0].length == b.length) : "wrong size";
        double[][] x = clone(a);
        
        for(int i=0; i<a.length; i++) {
            for(int j=0; j<a[0].length; j++) {
                double sum = a[i][j] + b[j];
                double sig = (double)(1/(1+Math.exp(-sum)));
                x[i][j] = sig*(1-sig);
            }
        }
        
        return x;
    }
        
    public static double[][][][] clone(double[][][][] a) {
        double[][][][] x = new double[a.length][a[0].length][a[0][0].length][a[0][0][0].length];
        return x;
    }
    
    public static double[][] clone(double[][] a) {
        double[][] x = new double[a.length][a[0].length];
        return x;
    }
    
    public static double[] clone(double[] a) {
        double[] x = new double[a.length];
        return x;
    }
    
    public static double[][][][] unbatch(double[][] input, double[][][][] shape) {
        double[][][][] output = clone(shape);
        for(int i=0; i<output.length; i++) {
            int cnt = 0;
            for(int j=0; j<output[0].length; j++) {
                for(int k=0; k<output[0][0].length; k++) {
                    for(int l=0; l<output[0][0][0].length; l++) {
                        output[i][j][k][l] = input[i][cnt];
                        cnt++;
                    }
                }
            }
        }
        return output;
    }
    
    public static double[][] batch(double[][][][] input) {
        double[][] output = new double[input.length][input[0].length*input[0][0].length*input[0][0][0].length];
        for(int i=0; i<input.length; i++) {
            int cnt = 0;
            for(int j=0; j<input[0].length; j++) {
                for(int k=0; k<input[0][0].length; k++) {
                    for(int l=0; l<input[0][0][0].length; l++) {
                        output[i][cnt] = input[i][j][k][l];
                        cnt++;
                    }
                }
            }
        }
        return output;
    }
    
    public static void printShape(double a[][][][]) {
        System.out.println(a.length+" "+a[0].length+" "+a[0][0].length+" "+a[0][0][0].length);
    }
    
    public static void printShape(double a[][]) {
        System.out.println(a.length+" "+a[0].length);
    }
    
    public static void printShape(double a[]) {
        System.out.println(a.length);
    }
    
    public static double networkCost(double[][][][] input, double[][] label,
        double[] biases1, double[][][][] kernel1, double[] biases2, double[][] weights2, double[] biases3, double[][] weights3, boolean print) {       
        int p1_1=1, p2_1=1;
        int s1_1=2, s2_1=2;
        
        double[][][][] output1 = conv(input,kernel1,p1_1,p2_1,s1_1,s2_1);
        double[][][][] sig_output1 = sigmoid(output1,biases1);
                
        //*********************************************************
        
        int k1_2=2, k2_2=2;
        int p1_2=1, p2_2=1;
        int s1_2=2, s2_2=2;
        
        int[][][][][] switches1 = switches(sig_output1,k1_2,k2_2,p1_2,p2_2,s1_2,s2_2);
        double[][][][] pool_output1 = pool(sig_output1,switches1,k1_2,k2_2,p1_2,p2_2,s1_2,s2_2);
                
        //*********************************************************
        double[][] output2 = mul(batch(pool_output1),weights2,false,false);
        double[][] sig_output2 = sigmoid(output2,biases2);
        //*********************************************************
        
        double[][] output3 = mul(sig_output2,weights3,false,false);
        double[][] sig_output3 = sigmoid(output3,biases3);
        
        if(print) {
            double[][] gB3 = hadMul(sub(sig_output3, label),dsigmoid(output3,biases3));
            System.out.println("*********************************************************************");
            print(col(gB3));
            
            double[][] gB2 = hadMul(mul(gB3,weights3,false,true),dsigmoid(output2,biases2));
            System.out.println("*********************************************************************");
            print(col(gB2));
            
            double[][][][] gB1 = hadMul(trans_pool(unbatch(mul(gB2,weights2,false,true),pool_output1),switches1,sig_output1),dsigmoid(output1,biases1));
            System.out.println("*********************************************************************");
            print(col(gB1));
            
            double[][][][] grad1 = prod(input,gB1,kernel1,p1_1,p2_1,s1_1,s2_1);
            System.out.println("*********************************************************************");
            print(grad1);
            
            double[][] grad2 = mul(batch(pool_output1),gB2,true,false);
            System.out.println("*********************************************************************");
            print(grad2);
            
            double[][] grad3 = mul(sig_output2,gB3,true,false);
            System.out.println("*********************************************************************");
            print(grad3);
        }
                
        double cost = cost(sig_output3, label);
        return cost;
    }
    
    public static double[][][][] prod(double[][][][] input, double[][][][] output, double[][][][] kernel, int p1, int p2, int s1, int s2) {
        assert(input.length==output.length): "wrong size";
        int B = input.length; 
        int D = input[0].length;
        int K = output[0].length;
        
        double[][][][] kern = clone(kernel);
        for(int i=0; i<D; i++) {
            for(int j=0; j<K; j++) {
                for(int k=0; k<B; k++) {
                    matConv(kern[i][j],input[k][i],output[k][j],p1,p2,s1,s2);
                }
            }
        }        
        return kern;
    }
    
    public static void matConv(double[][] kern, double[][] input, double[][] output, int p1, int p2, int s1, int s2) {
        int i1 = input.length;
        int i2 = input[0].length;
        
        int k1 = kern.length;
        int k2 = kern[0].length;
        
        int o1 = output.length;
        int o2 = output[0].length;
        
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
                        
                        kern[k][l] += input[x][y]*output[i][j];
                    }
                }
            }
        }
    }
    
    public static double[][][][] trans_pool(double[][][][] output, int[][][][][] switches, double[][][][] sig_output) {
        assert(switches.length == output.length && switches[0].length == output[0].length) : "switches, output mismatch";
        
        int i1 = sig_output[0][0].length;
        int i2 = sig_output[0][0][0].length;
        
        int o1 = output.length;
        int o2 = output[0].length;
        
        double[][][][] outputPrime = new double[o1][o2][i1][i2];
        for(int i=0; i<output.length; i++) {
            for(int j=0; j<output[0].length; j++) {
                outputPrime[i][j] = trans_pool(output[i][j],switches[i][j],sig_output);
            }
        }
        
        return outputPrime;
    }
        
    public static double[][] trans_pool(double[][] output, int[][][] switches, double[][][][] sig_output) {
        int i1 = sig_output[0][0].length;
        int i2 = sig_output[0][0][0].length;
        
        int o1 = output.length;
        int o2 = output[0].length;

        double[][] outputPrime = new double[i1][i2];
        for(int i=0; i<o1; i++) {
            for(int j=0; j<o2; j++) {
                int x = switches[i][j][0];
                int y = switches[i][j][1];
                outputPrime[x][y] = output[i][j];
            }
        }
        return outputPrime;
    }

    public static int[][][][][] switches(double[][][][] sig_output, int k1, int k2, int p1, int p2, int s1, int s2) {
        int j1 = sig_output.length;
        int j2 = sig_output[0].length;
        
        int i1 = sig_output[0][0].length;
        int i2 = sig_output[0][0][0].length;
        
        int o1 = (i1+2*p1-k1)/s1 + 1;
        int o2 = (i2+2*p2-k2)/s2 + 1;

        int[][][][][] switches = new int[j1][j2][o1][o2][2];
        return switches;
    }

    public static double[][][][] pool(double[][][][] input, int[][][][][] switches, int k1, int k2, int p1, int p2, int s1, int s2) {
        double[][][][] output = new double[input.length][input[0].length][][];
        for(int i=0; i<input.length; i++) {
            for(int j=0; j<input[0].length; j++) {
                output[i][j] = pool(input[i][j],switches[i][j],k1,k2,p1,p2,s1,s2);
            }
        }
        return output;
    }
    
    public static double[][] pool(double[][] input, int[][][] switches, int k1, int k2, int p1, int p2, int s1, int s2) {
        int i1 = input.length;
        int i2 = input[0].length;
        
        int o1 = (i1+2*p1-k1)/s1 + 1;
        int o2 = (i2+2*p2-k2)/s2 + 1;
        
        double[][] output = new double[o1][o2];
        assert(k1>p1 && k2>p2): "kernel, padding not compatible";
        for(int i=0; i<o1; i++) {
            for(int j=0; j<o2; j++) {
                //pooling
                int maxInputX = 0;
                int maxInputY = 0;
                double max = Double.NEGATIVE_INFINITY;
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
                output[i][j] = max;
                switches[i][j][0] = maxInputX;
                switches[i][j][1] = maxInputY;
            }
        }
        return output;
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
        Random r = new Random();
        double[][][][] a = new double[b][d][i1][i2];
        for(int i=0; i<b; i++) {
            for(int j=0; j<d; j++) {
                for(int k=0; k<i1; k++) {
                    for(int l=0; l<i2; l++) {
                        a[i][j][k][l] = (double)(0.01*r.nextGaussian());
                    }
                }
            }
        }
        return a;
    }
    
    public static double[][] random(int b, int d) {
        Random r = new Random();
        double[][] a = new double[b][d];
        for(int i=0; i<b; i++) {
            for(int j=0; j<d; j++) {
                a[i][j] = (double)(0.01*r.nextGaussian());
            }
        }
        return a;
    }
    
    public static double[] random(int d) {
        Random r = new Random();
        double[] a = new double[d];
        for(int i=0; i<d; i++){
            a[i] = (double)r.nextGaussian();
        }
        return a;
    }
    
    public static void print(double[][][][] a) {
        for(int i=0; i<a.length; i++) {
            for(int j=0; j<a[0].length; j++) {
                for(int k=0; k<a[0][0].length; k++) {
                    for(int l=0; l<a[0][0][0].length; l++) {
                        //System.out.printf("%.8f ",a[i][j][k][l]);
                        System.out.print(a[i][j][k][l]+" ");
                    }
                    System.out.println();
                }
                System.out.println();
            }
            System.out.println();
        }
        System.out.println();
    }
    
    public static void print(double[][] a) {
        for(int i=0; i<a.length; i++) {
            for(int j=0; j<a[0].length; j++) {
                //System.out.printf("%.8f ",a[i][j][k][l]);
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
    
    public static double[][] mul(double[][] a, double[][] b, boolean left, boolean right) {
        int m1,n1;
        int m2,n2;
        
        double[][] c;
        
        if(!left) {
            if(!right) {
                m1 = a.length;
                n1 = a[0].length;
                m2 = b.length;
                n2 = b[0].length;
                
                c = new double[m1][n2];
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
                
                c = new double[m1][n2];
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
            if(!right) {
                m1 = a[0].length;
                n1 = a.length;
                m2 = b.length;
                n2 = b[0].length;
                
                c = new double[m1][n2];
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
                
                c = new double[m1][n2];
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
}
