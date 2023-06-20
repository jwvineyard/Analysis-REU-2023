This file describes the plots in this folder, as well as describing how they are generated.

Generation algorithm: To generate these plots, we must find the polynomial that maximizes the value K(n,F), where n is the degree of the polynomial and F is the mesh being tested. To do this, we randomly generate a large number of polynomials (the number is determined on a case-by-case basis), and for each randomly generated polynomial, we test its value at the mesh points to determine the sampling norm, and at 1000 unformly chosen points on the unit circle to determine the sup norm (this number can be higher when dealing with larger degrees of polynomial). Then, we choose the polynomial with the larget ratio between the sup norm and sampling norm.

To generate a degree n polynomial, we choose n+1 points uniformly at random on the unit disk, and set the polynomial's coefficients equal to those values. Note that that does not imply that the polynomial will have a sup norm less than or equal to 1.

Some examples of generated polynomials are as follows:

![image](https://github.com/jwvineyard/Analysis-REU-2023/assets/72844296/e22e7c12-f5e6-493a-a2dd-4d94a13b0844)
This is a plot of a degree 3 polynomial with the sampling mesh being the roots of unity, with N = 5 (N = n+2). You can see that the polynomial is symmetric around the peak, and that the values of the polynomial at the sampling points is the same for many of them.

![image](https://github.com/jwvineyard/Analysis-REU-2023/assets/72844296/844f97eb-b7f4-4133-b0a5-7f85b8745743)
This is the same as in the previous plot, but with N = 2n = 6. There is incredibly high symmetry in this plot.

![image](https://github.com/jwvineyard/Analysis-REU-2023/assets/72844296/8ee201de-eb2e-4a76-b758-35f31610b369)
![image](https://github.com/jwvineyard/Analysis-REU-2023/assets/72844296/2155633e-9102-48d8-821d-1d6d25a5c32e)
These two images show that the periodicity seen in the degree 3 polynomial with the uniform mesh at N = 2n seems to appear whenever N = kn, where k is a positive integer.
