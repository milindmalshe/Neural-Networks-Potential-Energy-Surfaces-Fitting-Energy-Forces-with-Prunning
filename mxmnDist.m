function mx_mndist = mxmnDist(P)

%THIS FILE IS TO COMPUTE THE MAXIMUM OF MINIMUM DISTANCE AMONG THE DATA

[R,Q] = size(P);

%Compute distance
D = dist(P',P);

%Set the diag elements to be a high value
D = D + diag(1e6*ones(1,Q));

%Find max of min distance
mx_mndist = max(min(D));