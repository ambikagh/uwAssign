% load the provided dataset
load('USPS.mat')

%calculate the eigen values and eigen vectors
covA = cov(A); % covA -> covariance matrix of A
eig_values = eig(covA); % find the eigen values for covA matrix
[eig_vector, D] = eig(covA); % find the eigen vector
[~,evs] = sort(abs(eig_values),'descend'); % ensure highest to lowest values of eigen values
eigv_largest = eig_vector(:,evs);

% Principal Component Analysis to the data using 
% 10 prinicpal components
eigv_10 = eigv_largest(:,1:10); % X(:,1:k) k largest values from eigen vector
pca_10 = A*eigv_10;
% 50 principal components
eigv_50 = eigv_largest(:,1:50);
pca_50 = A*eigv_50;
% 100 principal components
eigv_100 = eigv_largest(:,1:100);
pca_100 = A*eigv_100;
% 200 principal components
eigv_200 = eigv_largest(:,1:200);
pca_200 = A*eigv_200;

% original image
A1 = reshape(A(1,:),16,16);
%imshow(A1);
A2 = reshape(A(2,:),16,16);
%imshow(A2);

% Reconstruct the images 
A10_re = pca_10*transpose(eigv_10);
A10_1 = reshape(A10_re(1,:),16,16);
%imshow(A10_1);
A10_2 = reshape(A10_re(2,:),16,16);
%imshow(A10_2);
A50_re = pca_50*transpose(eigv_50);
A50_1 = reshape(A50_re(1,:),16,16);
%imshow(A50_1);
A50_2 = reshape(A50_re(2,:),16,16);
%imshow(A50_2);
A100_re = pca_100*transpose(eigv_100);
A100_1 = reshape(A100_re(1,:),16,16);
%imshow(A100_1);
A100_2 = reshape(A100_re(2,:),16,16);
%imshow(A100_2);
A200_re = pca_200*transpose(eigv_200);
A200_1 = reshape(A200_re(1,:),16,16);
imshow(A200_1);
A200_2 = reshape(A200_re(2,:),16,16);
%imshow(A200_2);

