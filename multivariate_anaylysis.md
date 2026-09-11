About Eigenvectors,Eigenvalues,SVD and PSD.

Take 3x3 matrix as example below

`eigenvector & eigenvalues`
Linearly independent <=> 3 unique eigenvectors(doesn't mean 3 eigenvalues!! e.g. Identity matrix)

(A-$\lambda$ I)v = 0 , where v is eigenvector
One quickest and common way to find the eigenvalues is to find by det(A-$\lambda$ I) = 0, and then it's easy to find the corresponding eigenvector.

`SVD`
A = U $\Sigma$ V.T
V is the right singular matrix of A, it's an orthogonal matrix, which formed by A's right singular vectors.
*Interesting* -> Right singular vectors/values of A <=> Eigenvectors/values of (A.T)A
And the singular values are always positive, which because (A.T)A are PSD(wait it should be Non-negative SD)
-->Verify: (1)it's symmetric (2) (z.T)(A.T)(A)z = (Az).T(Az) = the dot product of vector Az = |Az|^2 >= 0


`PSD` 
Two requirements
(1) Symmetric
(2) (z.T)Az >= 0 for any vector z (it should be > 0 for PSD) 
Or (2) Can also be eigenvalues all >0

Characteristic --> Eigenvalues are all postive(>=0)

`determinant`
Det(AB) = det(A)det(B), that's why if a matrix is not linearly dependent, det(A) = det(U)det(big sigma)det(V),
and det(big sigma) = 0, since there's no full eigenvalues.

Also, det(A) = lambda1* lambda2* lambda3* ...* lambdaN, if A have full rank?



Question:
Why Cov(CX) = C * Cov(X) * C'