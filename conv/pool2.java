import java.lang.Math;
public class pool2
{
    public static void main(String[] args) {        
        /////////////////////////////////////////////////////////////////////////////////////////////////////////////
        //POOLING
        
        int i1=6, i2=6;
        int k1=2, k2=2;
        int p1=1, p2=1;
        int s1=1, s2=1;

        int[][][] switches = switches(i1,i2,k1,k2,p1,p2,s1,s2);
        
        double[][] input = random(i1,i2);
        print(input);
 
        double[][] output = pool(input,switches,k1,k2,p1,p2,s1,s2);
        print(vec(output));
        
        double[][] poolMat = poolMat(input,k1,k2,p1,p2,s1,s2);
		print(vec(mul(poolMat, vec(input), true, true)));

		System.out.println( diff(vec(mul(poolMat, vec(input), true, true)),   vec(output))  );
        
		print(vec(mul(poolMat, vec(output), false, true)));
		
        double[][] outputPrime = trans_pool(output, switches, i1, i2);
        print(vec(outputPrime));
        
        System.out.println( diff(vec(mul(poolMat, vec(output), false, true)),   vec(outputPrime))  );
    }
    
    //CHANGED = TO +=, NOW WORKS!!!!!!!!
    public static double[][] trans_pool(double[][] output, int[][][] switches, int i1, int i2) {
        int o1 = output.length;
		int o2 = output[0].length;

		double[][] outputPrime = new double[i1][i2];
        for(int i=0; i<o1; i++) {
            for(int j=0; j<o2; j++) {
                int x = switches[i][j][0];
                int y = switches[i][j][1];
                outputPrime[x][y] += output[i][j];
            }
        }
        return outputPrime;
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
				int maxInputIndex = 0;
                int maxOutputIndex = 0;
                double max = Double.NEGATIVE_INFINITY;
                for(int k=Math.max(0,-pad(i*s1,p1)); k<Math.min(k1,i1-pad(i*s1,p1)); k++) {
                    for(int l=Math.max(0,-pad(j*s2,p2)); l<Math.min(k2,i2-pad(j*s2,p2)); l++) {
                        //i*s1 gives distance travelled after i strides
                        //k gives displacement due to kernel
                        //-p1 gives displacement due to padding
						int x = pad(i*s1+k,p1);
                        int y = pad(j*s2+l,p2);

						//x*i2+y gives the index of the input
						//i*o2+j gives the index of the output
                        if(max<input[x][y]) {
                        	maxOutputIndex = i*o2+j;
                        	maxInputIndex = x*i2+y;
                        	max = input[x][y];
                        }
                    }
                }
                output[maxOutputIndex][maxInputIndex] = 1;
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

	public static int[][][] switches(int i1, int i2, int k1, int k2, int p1, int p2, int s1, int s2) {
		int o1 = (i1+2*p1-k1)/s1 + 1;
        int o2 = (i2+2*p2-k2)/s2 + 1;
        int[][][] switches = new int[o1][o2][2];
		return switches;
	}
    
    public static int pad(int i, int p) {
        return i-p;
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