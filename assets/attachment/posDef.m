
deg = pi/180;

for th = 0*deg:10*deg:90*deg,

Mt = [cos(th) -sin(th);
    sin(th) cos(th)];

%M = [1.5 -0.05; 0.5 1.2]; % rotating
%M = [-0.5 0.05; 0.04 1.2];
%M = [0.8 0.05; 0.04 1.2];
M = [0.8 0.05; 0.04 1.2];

M = inv(Mt)*M*Mt;

A = TransformMatrix(M);
[V,D] = A.eig;

A.dispTransformMatrix;
A.dispEigenvectors;
A.dispEigenvalues;

xrange = [-5:1:5];
yrange = [-5:1:5];

isShowUntransformedVectors = 0;
isShowTransformedVecotrs = 0;
isShowDifferenceVectors = 1;
isShowEigenvectors = 1;
isShowQuadraticValues = 0;

A.showField(xrange,yrange,isShowUntransformedVectors,...
    isShowTransformedVecotrs,isShowDifferenceVectors,...
    isShowEigenvectors,isShowQuadraticValues);
axis(8*[-1 1 -1 1]);

A.showEigenvalues;
axis(2*[-1 1 -1 1]);

pause(1);

end