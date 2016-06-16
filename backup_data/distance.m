function dist = gDist(x1, x2)
%RBFKERNEL returns a radial basis function kernel between x1 and x2
% Ensure that x1 and x2 are column vectors
x1 = x1(:); x2 = x2(:);

%   sim = gaussianKernel(x1, x2) returns a gaussian kernel between x1 and x2
%   and returns the value in sim
xny         =   x1-x2;
dist    =   xny'*xny;

end