% generate 1000 randomly distributed data points with x and y two features 
x=randn(1000,1);
y=10*x + 10.*randn(1000,1)-5;
A = horzcat(x,y);
subplot(1,2,1),scatter(A(:,1),A(:,2));

% calculate the first principle component and do the projection
B = cov(A);
ev = eig(B); %eigen values
[V,D] = eig(B); % V -> eigen vector
[~,evs] = sort(abs(ev),'descend'); %sort the eigenvalues 
eigv_l = V(:,evs); % get the largest matrix of eigenvectors corresponding to eigenvalues
eigv_one = eigv_l(:,1:1); % pick the 1 largest eigen vector
mean_A = mean(A);
centered_A = A - mean_A;
first_PCA = centered_A*eigv_one; % get the first principal component

% reconstruct the original two variables from this one principal component 
VT = transpose(eigv_one);
A_reconstructed = first_PCA*VT;
subplot(1,2,1),scatter(A_reconstructed(:,1),A_reconstructed(:,2));