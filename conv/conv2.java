import java.lang.Math;
public class conv2
{
    public static void main(String[] args) {
        /////////////////////////////////////////////////////////////////////////////////////////////////////////////
        //CONVOLUTION
       
        int i1=4, i2=4;
        int k1=3, k2=3;
        int p1=1, p2=1;
        int s1=1, s2=1;
        
        int a1=(i1-k1+2*p1)%s1, a2=(i2-k2+2*p2)%s2;
        
        System.out.println("a1: "+a1);
        System.out.println("a2: "+a2);
        
        double[][] input = random(i1, i2);
        System.out.println("INPUT");
        print(input);
        
        double[][] kernel = random(k1, k2);
        System.out.println("KERNEL");
        print(kernel);
        
        double[][] output = conv(input, kernel, p1, p2, s1, s2);
        System.out.println("OUTPUT");
        print(output);
                
        double[][] convMat = convMat(input, kernel, p1, p2, s1, s2);
        System.out.println("CONVMAT");
        print(convMat);
        
        double[][] matConv = matConv(convMat, i1, i2, k1, k2, p1, p2, s1, s2);
        System.out.println("MATCONV");
        print(matConv);

        //convolution
        System.out.println("VEC OUTPUT");
        print(vec(output));
        
        //convolution using matrix
        System.out.println("CONVOLUTION");
        print(vec(mul(convMat,vec(input),true,true)));
        
        //demonstrate equivalence
        System.out.println("DIFF");
        System.out.println(diff(vec(output), vec(mul(convMat,vec(input),true,true))));
        
        //transposed convolution
        System.out.println("TRANS CONV");
        double[][] outputPrime = trans_conv(output, kernel, p1, p2, s1, s2, a1, a2);
        print(vec(outputPrime));
        
        //transposed convolution using matrix
        System.out.println("TRANS CONV");
        print(mul(convMat,vec(output),false,true));
        
        //demonstrate equivalence
        System.out.println("DIFF");
        System.out.println(diff(vec(outputPrime) ,mul(convMat,vec(output),false,true)) );
    }

    public static int dil(int i, int s) {
        return i/s;
    }
    
    public static int pad(int i, int p) {
        return i-p;
    }
    
    public static int rot(int i, int x) {
        return x-i-1;
    }

    public static double[][] trans_conv(double[][] input, double[][] kernel, int p1, int p2, int s1, int s2, int a1, int a2) {       
        int i1 = (input.length-1)*(s1-1)+input.length+a1;
        int i2 = (input[0].length-1)*(s2-1)+input[0].length+a2;
        
        int k1 = kernel.length;
        int k2 = kernel[0].length;
        
        p1 = k1-p1-1;
        p2 = k2-p2-1;
                
        final int s = 1;
        
        int o1 = (i1+2*p1-k1)/s + 1;
        int o2 = (i2+2*p2-k2)/s + 1;
        
        double[][] output = new double[o1][o2];
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
        return output;
    }
    
    public static double[][] conv(double[][] input, double[][] kernel, int p1, int p2, int s1, int s2) {
        int i1 = input.length;
        int i2 = input[0].length;
        
        int k1 = kernel.length;
        int k2 = kernel[0].length;
                
        int o1 = (i1+2*p1-k1)/s1 + 1;
        int o2 = (i2+2*p2-k2)/s2 + 1;
        
        double[][] output = new double[o1][o2];
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
        return output;
    }
    
    public static double[][] convMat(double[][] input, double[][] kernel, int p1, int p2, int s1, int s2) {
        int i1 = input.length;
        int i2 = input[0].length;
        
        int k1 = kernel.length;
        int k2 = kernel[0].length;
               
        int o1 = (i1+2*p1-k1)/s1 + 1;
        int o2 = (i2+2*p2-k2)/s2 + 1;
        
        double[][] convMat = new double[o1*o2][i1*i2];
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

						//x*i2+y gives the index of the input
						//i*o2+j gives the index of the output
                        convMat[i*o2+j][x*i2+y] = kernel[k][l];
                    }
                }
            }
        }
        return convMat;
    }
    
    public static double[][] matConv(double[][] convMat, int i1, int i2, int k1, int k2, int p1, int p2, int s1, int s2) {               
        int o1 = (i1+2*p1-k1)/s1 + 1;
        int o2 = (i2+2*p2-k2)/s2 + 1;
        
        double[][] kernel = new double[k1][k2];
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

						//x*i2+y gives the index of the input
						//i*o2+j gives the index of the output
                        kernel[k][l] = convMat[i*o2+j][x*i2+y];

						//in reality you would add values corresponding to gradients with += instead of =
                    }
                }
            }
        }
        return kernel;
    }
    
    //combine with matConv to calculate indexing using dot product
    public static double[][] mul(double[][] a, double[][] b, boolean left, boolean right) {
        int m1,n1;
        int m2,n2;
        
        double[][] c;
        
        if(left) {
            if(right) {
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
            if(right) {
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
    
    public static double[][] vec(double[][] a) {
        double[][] vec = new double[a.length*a[0].length][1];
        int index=0;
        for(int i=0; i<a.length; i++) {
            for(int j=0; j<a[i].length; j++) {
                vec[index][0] = a[i][j];
                index++;
            }
        }
        return vec;
    }
    
    public static double[][] random(int i1, int i2) {
        double[][] a = new double[i1][i2];
        for(int i=0; i<i1; i++) {
            for(int j=0; j<i2; j++) {
                a[i][j] = Math.random();
            }
        }
        return a;
    }
    
    public static double diff(double[][] a, double[][] b) {
        double diff = 0;
        for(int i=0; i<a.length; i++) {
            diff+=(a[i][0]-b[i][0])*(a[i][0]-b[i][0]);
        }
        System.out.println(a.length-b.length);
        return diff/a.length;
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

	public static void print(int[][] a) {        
        for(int i=0; i<a.length; i++) {
            for(int j=0; j<a[i].length; j++) {
                System.out.print(a[i][j]);
            }
            System.out.println();
        }
        System.out.println();
   	}
}