import java.lang.Math;
public class conv
{
    public static void main(String[] args) {
        /////////////////////////////////////////////////////////////////////////////////////////////////////////////
        //CONVOLUTION
        
        int i1=8, i2=7;
        int k1=2, k2=2;
        int p1=1, p2=1;
        int s1=3, s2=3;
        
        int a1=(i1-k1+2*p1)%s1, a2=(i2-k2+2*p2)%s2;
        
        System.out.println(a1);
        System.out.println(a2);
        
        double[][] input = random(i1, i2);
        print(input);
        
        double[][] kernel = random(k1, k2);
        print(kernel);
        
        double[][] output = conv(input, kernel, p1, p2, s1, s2);
        print(output);
                
        double[][] convMat = convMat(input, kernel, p1, p2, s1, s2);
        print(convMat);
        
        double[][] matConv = matConv(convMat, i1, i2, k1, k2, p1, p2, s1, s2);
        print(matConv);

        //standard convolution
       
        print(vec(output));
        
        //standard convolution using matrix
        
        print(vec(mul(convMat,vec(input),true,true)));
        
        //demonstrate equivalence
        
        System.out.println(diff(vec(output),vec(mul(convMat,vec(input),true,true))));
        
        //transposed convolution with zeros
        
        //double[][] outputPrime = conv(tl_pad(rot(dil(output,s1,s2)),a1,a2), kernel, k1-p1-1, k2-p2-1, 1, 1);
        //print(vec(rot(outputPrime)));
        
        //transposed convolution without zeros

        double[][] outputPrime = trans_conv(output, kernel, p1, p2, s1, s2, a1, a2);
        print(vec(outputPrime));
        
        //transposed convolution using matrix
        
        print(mul(convMat,vec(output),false,true));
        
        //demonstrate equivalence
        
        //System.out.println( diff(vec(rot(outputPrime)) ,mul(convMat,vec(output),false,true)) );
        System.out.println( diff(vec(outputPrime)      ,mul(convMat,vec(output),false,true)) );
        
        //sample methods
        
        //double[][] mat1 = random(3,3);
        //print(mat1);
        //print(pad(mat1,1,2));
        //print(rot(mat1));
        //print(dil(mat1,2,2));
        //print(tl_pad(mat1, 2,3));
   
        /////////////////////////////////////////////////////////////////////////////////////////////////////////////
        //POOLING
        
        //int i1=3, i2=6;
        //int k1=3, k2=3;
        //int p1=0, p2=0;
        //int s1=3, s2=3;
        
        //int a1=(i1-k1+2*p1)%s1, a2=(i2-k2+2*p2)%s2;
        
        //double[][] input = random(i1,i2);
        //print(pad(input,p1,p2));
 
        //double[][] output = pool(input, k1, k2, p1, p2, s1, s2);
        //print(vec(output));
        //double[][] poolMat = poolMat(input, k1, k2, p1, p2, s1, s2);
        //print(poolMat);
               
        //int o1 = (i1+2*p1-k1)/s1 + 1;
        //int o2 = (i2+2*p2-k2)/s2 + 1;
        //int[][] switches = poolSwitch(input, k1, k2, p1, p2, s1, s2);
        //for(int i=0; i<o1*o2; i++) {
        //    for(int j=0; j<2; j++) {
        //        System.out.print(switches[i][j]+" ");
        //    }
        //    System.out.println();
        //}
        //System.out.println();
        
        //print(vec(mul(poolMat, vec(input), true, true)));
        //print(vec(mul(poolMat, vec(output), false, true)));

        //double[][] outputPrime = trans_pool(output, switches, o1, o2, i1, i2);
        //print(vec(outputPrime));
        
        //System.out.println( diff(vec(mul(poolMat, vec(output), false, true)),   vec(outputPrime))  );
        
        /////////////////////////////////////////////////////////////////////////////////////////////////////////////
        //RECURSIVE
        
        //int[] index = {0,0,0};
        //traverse(index, 0);
        
    }

    public static void traverse(int[] index, int level) {
        if(level==index.length) {
            for(int i=0; i<index.length; i++) {
                System.out.print(index[i]);
            }
            System.out.println();
        }
        
        else {
            for(index[level]=0; index[level]<10; index[level]++) {
                traverse(index, level+1);
            }
        }
    }
    
    public static void print(int[] index) {
        for(int i=0; i<index.length; i++) {
            System.out.print(index[i]);
        }
        System.out.println();
    }
    
    public static double[][] trans_pool(double[][] output, int[][] switches, int o1, int o2, int i1, int i2) {
        double[][] outputPrime = new double[i1][i2];
        for(int i=0; i<o1; i++) {
            for(int j=0; j<o2; j++) {
                int x = switches[i*o2+j][0];
                int y = switches[i*o2+j][1];
                outputPrime[x][y] = output[i][j];
            }
        }
        return outputPrime;
    }
    
    public static int[][] poolSwitch(double[][] input, int k1, int k2, int p1, int p2, int s1, int s2) {
        int i1 = input.length;
        int i2 = input[0].length;
        
        int o1 = (i1+2*p1-k1)/s1 + 1;
        int o2 = (i2+2*p2-k2)/s2 + 1;
        
        int[][] switches = new int[o1*o2][2];
        for(int i=0; i<o1; i++) {
            for(int j=0; j<o2; j++) {
                //pooling
                double max = Double.NEGATIVE_INFINITY;
                int maxI1 = 0;
                int maxI2 = 0;
                
                int x = i*o2+j;
                for(int k=0; k<k1; k++) {
                    for(int l=0; l<k2; l++) {
                        //i*s1 gives distance travelled after i strides
                        //k gives displacement due to kernel
                        //-p1 gives displacement due to padding
                        if( (i*s1+k-p1)>=0 && (i*s1+k-p1)<i1 && (j*s2+l-p2)>=0 && (j*s2+l-p2)<i2 ) {
                             if(max<input[i*s1+k-p1][j*s2+l-p2]) {
                                 maxI1 = (i*s1+k-p1);
                                 maxI2 = (j*s2+l-p2);
                                 max = input[i*s1+k-p1][j*s2+l-p2];
                             }
                        }
                    }
                }
                switches[x][0] = maxI1;
                switches[x][1] = maxI2;
            }
        }
        return switches;
    }
    
    public static double[][] poolMat(double[][] input, int k1, int k2, int p1, int p2, int s1, int s2) {
        int i1 = input.length;
        int i2 = input[0].length;
        
        int o1 = (i1+2*p1-k1)/s1 + 1;
        int o2 = (i2+2*p2-k2)/s2 + 1;
        
        double[][] output = new double[o1*o2][i1*i2];
        for(int i=0; i<o1; i++) {
            for(int j=0; j<o2; j++) {
                //pooling
                double max = Double.NEGATIVE_INFINITY;
                int maxX = 0;
                int maxY = 0;
                
                int x = i*o2+j;
                for(int k=0; k<k1; k++) {
                    for(int l=0; l<k2; l++) {
                        //i*s1 gives distance travelled after i strides
                        //k gives displacement due to kernel
                        //-p1 gives displacement due to padding
                        if( (i*s1+k-p1)>=0 && (i*s1+k-p1)<i1 && (j*s2+l-p2)>=0 && (j*s2+l-p2)<i2 ) {
                             if(max<input[i*s1+k-p1][j*s2+l-p2]) {
                                 maxX = i*o2 + j;
                                 maxY = (i*s1+k-p1)*i2 + (j*s2+l-p2);
                                 max = input[i*s1+k-p1][j*s2+l-p2];
                             }
                        }
                    }
                }
                output[maxX][maxY] = 1;
            }
        }
        return output;
    }
    
    public static double[][] pool(double[][] input, int k1, int k2, int p1, int p2, int s1, int s2) {
        int i1 = input.length;
        int i2 = input[0].length;
        
        int o1 = (i1+2*p1-k1)/s1 + 1;
        int o2 = (i2+2*p2-k2)/s2 + 1;
        
        double[][] output = new double[o1][o2];
        for(int i=0; i<o1; i++) {
            for(int j=0; j<o2; j++) {
                //pooling
                double max = Double.NEGATIVE_INFINITY;
                for(int k=0; k<k1; k++) {
                    for(int l=0; l<k2; l++) {
                        //i*s1 gives distance travelled after i strides
                        //k gives displacement due to kernel
                        //-p1 gives displacement due to padding
                        if( (i*s1+k-p1)>=0 && (i*s1+k-p1)<i1 && (j*s2+l-p2)>=0 && (j*s2+l-p2)<i2 ) {
                             if(max<input[i*s1+k-p1][j*s2+l-p2]) {
                                 max = input[i*s1+k-p1][j*s2+l-p2];
                             }
                        }
                    }
                }
                output[i][j] = max;
            }
        }
        return output;
    }
    
    public static double diff(double[][] a, double[][] b) {
        double diff = 0;
        for(int i=0; i<a.length; i++) {
            diff+=(a[i][0]-b[i][0])*(a[i][0]-b[i][0]);
        }
        System.out.println(a.length-b.length);
        return diff/a.length;
    }
    
    public static double[][] tl_pad(double[][] a, int a1, int a2) {
        int i1 = a.length+a1;
        int i2 = a[0].length+a2;
        
        double[][] tl = new double[i1][i2];
        for(int i=a1; i<i1; i++) {
            for(int j=a2; j<i2; j++) {
                tl[i][j] = a[i-a1][j-a2];
            }
        }
        return tl;
    }
    
    public static int tl(int i, int a) {
        return i-a;
    }
    
    public static double[][] dil(double[][] a, int s1, int s2) {
        int i1 = a.length;
        int i2 = a[0].length;
        
        int x = (i1-1)*(s1-1)+i1;
        int y = (i2-1)*(s2-1)+i2;
        double[][] dil = new double[x][y];
        for(int i=0; i<x; i+=s1) {
            for(int j=0; j<y; j+=s2) {
                dil[i][j] = a[i/s1][j/s2];
            }
        }
        return dil;
    }
    
    public static int dil(int i, int s) {
        return i/s;
    }
    
    public static double[][] pad(double[][] input, int p1, int p2) {        
        int i1 = input.length+2*p1;
        int i2 = input[0].length+2*p2;
        
        double[][] padded_input = new double[i1][i2];
        for(int i=p1; i<i1-p1; i++) {
            for(int j=p2; j<i2-p2; j++) { 
                padded_input[i][j] = input[i-p1][j-p2];
            }
        }
        return padded_input;
    }
    
    public static int pad(int i, int p) {
        return i-p;
    }
    
    public static double[][] rot(double[][] a) {
        int x = a.length;
        int y = a[0].length;
        double[][] rot = new double[x][y];
        for(int i=0; i<x; i++) {
            for(int j=0; j<y; j++) {
                rot[i][j] = a[x-i-1][y-j-1];
            }
        }
        return rot;
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
                for(int k=0; k<k1; k++) {
                    for(int l=0; l<k2; l++) {
                        //i*s gives distance travelled after i strides
                        //k gives displacement due to kernel
                        //-p1 gives displacement due to padding
                        
                        int x = pad(i*s+k,p1);
                        int y = pad(j*s+l,p2);
                        if( x>=0 && x<i1 && y>=0 && y<i2 ) {
                            if(x>=a1 && y>=a2) {
                                if( (rot(tl(x,a1),i1-a1)%s1==0) && rot(tl(y,a2),i2-a2)%s2==0     ) {
                                    output[rot(i,o1)][rot(j,o2)] += kernel[k][l]*input[dil(rot(tl(x,a1),i1-a1),s1)][dil(rot(tl(y,a2),i2-a2),s2)];
                                }
                            }
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
                for(int k=0; k<k1; k++) {
                    for(int l=0; l<k2; l++) {
                        //i*s1 gives distance travelled after i strides
                        //k gives displacement due to kernel
                        //-p1 gives displacement due to padding
                        
                        int x = pad(i*s1+k,p1);
                        int y = pad(j*s2+l,p2);
                        if( x>=0 && x<i1 && y>=0 && y<i2 ) {
                            output[i][j] += kernel[k][l]*input[x][y];
                        }
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
               
        int o1 = (i1+2*p1-k1)/(s1) + 1;
        int o2 = (i2+2*p2-k2)/(s2) + 1;
        
        double[][] convMat = new double[o1*o2][i1*i2];
        for(int i=0; i<o1; i++) {
            for(int j=0; j<o2; j++) {
                //convolution 
                int x = i*o2 + j;
                for(int k=0; k<k1; k++) {
                    for(int l=0; l<k2; l++) {
                        //i*s1 gives distance travelled after i strides
                        //k gives displacement due to kernel
                        //-p1 gives displacement due to padding
                        if( (i*s1+k-p1)>=0 && (i*s1+k-p1)<i1 && (j*s2+l-p2)>=0 && (j*s2+l-p2)<i2 ) {
                            int y = (i*s1+k-p1)*i2 + (j*s2+l-p2);
                            convMat[x][y] = kernel[k][l];
                        }
                    }
                }
            }
        }
        return convMat;
    }
    
    public static double[][] matConv(double[][] convMat, int i1, int i2, int k1, int k2, int p1, int p2, int s1, int s2) {               
        int o1 = (i1+2*p1-k1)/(s1) + 1;
        int o2 = (i2+2*p2-k2)/(s2) + 1;
        
        double[][] matConv = new double[k1][k2];
        for(int i=0; i<o1; i++) {
            for(int j=0; j<o2; j++) {
                //convolution 
                int x = i*o2 + j;
                for(int k=0; k<k1; k++) {
                    for(int l=0; l<k2; l++) {
                        //i*s1 gives distance travelled after i strides
                        //k gives displacement due to kernel
                        //-p1 gives displacement due to padding
                        if( (i*s1+k-p1)>=0 && (i*s1+k-p1)<i1 && (j*s2+l-p2)>=0 && (j*s2+l-p2)<i2 ) {
                            int y = (i*s1+k-p1)*i2 + (j*s2+l-p2);
                            matConv[k][l] = convMat[x][y]; //in reality you would add values corresponding to gradients
                        }
                    }
                }
            }
        }
        return matConv;
    }
    
    //combine with matConv to calculate indexing using dot product
    public static double[][] mul(double[][] a, double[][] b, boolean left, boolean right) {
        int m1=0,n1=0,m2=0,n2=0;
        
        if(left && right) {
            m1 = a.length;
            n1 = a[0].length;
            m2 = b.length;
            n2 = b[0].length;
        }
        
        if(!left && right) {
            m1 = a[0].length;
            n1 = a.length;
            m2 = b.length;
            n2 = b[0].length;
        }
        
        if(left && !right) {
            m1 = a.length;
            n1 = a[0].length;
            m2 = b[0].length;
            n2 = b.length;
        }
        
        if(!left && !right) {
            m1 = a[0].length;
            n1 = a.length;
            m2 = b[0].length;
            n2 = b.length;
        }

        assert(n1 == m2): "wrong matrix dimension";
        double[][] c = new double[m1][n2];
        for(int i=0; i<m1; i++) {
            for(int j=0; j<n2; j++) {
                for(int k=0; k<n1; k++) {
                    if(left && right) {
                        c[i][j] += a[i][k]*b[k][j];
                    }
                    
                    if(!left && right) {
                        c[i][j] += a[k][i]*b[k][j];
                    }
                    
                    if(left && !right) {
                        c[i][j] += a[i][k]*b[j][k];
                    }
                    
                    if(!left && !right) {
                        c[i][j] += a[k][i]*b[j][k];
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
    
    public static void print(double[][] a) {        
        for(int i=0; i<a.length; i++) {
            for(int j=0; j<a[i].length; j++) {
                System.out.printf("%.4f ",a[i][j]);
            }
            System.out.println();
        }
        System.out.println();
    }
}
