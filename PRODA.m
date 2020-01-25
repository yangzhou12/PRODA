function [ model ] = PRODA( TX, gndTX, varargin )
% PRODA: Probabilistic Rank-One Discriminant Analysis
%
% %[Syntax]%: 
%   model = PRODA( TX, gndTX )
%   model = PRODA( ___, Name, Value )
%
% %[Inputs]%:
%   TX:            the dc x dr x numSpl training set of input matrices 
%   gndTX:         the numSpl x 1 class label set for the training data
%     
% %[Name-Value Pairs]
%   'Py':          the number of collective features
%                  0.5*dc*dr  (Default)
%
%   'Pz':          the number of individual features
%                  0.5*dc*dr  (Default)
%
%   'regParam':    the regularization parameter \gamma
%                  1e2  (Default)
%
%   'maxIter':    the maximum number of iterations
%                  500  (Default)
%
%   'tol':         the tolerance of the relative change of log-likelihood
%                  1e-4 (Default)
%
%   'spcInit':     the initilized model parameters
%                  []   (Default)
%
% %[Outputs]%:
%   model.Cy = Cy;          the collective column factor matrix
%   model.Cz = Cz;          the individual column factor matrix
%   model.Ry = Ry;          the collective row factor matrix
%   model.Rz = Rz;          the individual row factor matrix 
%   model.sigma = sigma;    the noise variance
%   model.TXmean = TXmean;  the mean of the training matrices TX 
%   model.liklhd = liklhd;  the log-likelihood at each iteration
%                        
% %[Toolbox needed]%:
%   This function needs the tensor toolbox v2.6 available at
%   http://www.sandia.gov/~tgkolda/TensorToolbox/
%                       
% %[Reference]%:            
%   Y. Zhou, Y.M. Cheung.
%   Probabilistic Rank-One Discriminant Analysis via Collective 
%   and Individual Variation Modeling.
%   IEEE Transactions on Cybernetics,
%   DOI:10.1109/TCYB.2018.2870440, 2018.
%                             
% %[Author Notes]%   
%   Author:        Yang ZHOU
%   Email :        yangzhou@comp.hkbu.edu.hk
%   Affiliation:   Department of Computer Science
%                  Hong Kong Baptist University
%   Release date:  Oct. 05, 2018
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% 


if nargin < 2, error('Not enough input arguments.'); end
[dc, dr, numSpl] = size(TX);

ip = inputParser;
ip.addParameter('Py', round(0.5*dc*dr), @isscalar);
ip.addParameter('Pz', round(0.5*dc*dr), @isscalar);
ip.addParameter('regParam', var(TX(:)), @isscalar);
ip.addParameter('maxIter', 500, @isscalar);
ip.addParameter('tol', 1e-4, @isscalar);
ip.addParameter('spcInit', [], @isstruct);
ip.parse(varargin{:});

Py = ip.Results.Py; 
Pz = ip.Results.Pz;

gamma   = ip.Results.regParam;
maxIter = ip.Results.maxIter; 
epsCvg  = ip.Results.tol; 
spcInit = ip.Results.spcInit;

%   Data centering
TXmean = mean(TX,3); % The mean
TX = bsxfun(@minus,TX,TXmean); % The centered data
X_vec = reshape(TX, dc*dr, numSpl); % Vectorized data

%	Indexing data by labels
classLabel = unique(gndTX);
nClass = length(classLabel); % Number of classes
ClsIdxs=cell(nClass,1);Nc=zeros(nClass,1);
sumX = zeros(dc*dr,nClass);
for c = 1:nClass
    ClsIdxs{c} = find(gndTX==classLabel(c));
    Nc(c) = length(ClsIdxs{c}); % Number of samples in each class
    sumX(:,c) = sum(X_vec(:,ClsIdxs{c}),2); % The class mean
end
covXX = X_vec*X_vec';

%   Initialization
if isempty(spcInit) 
    % Random initialization
    Cz = rand(dc, Pz); Rz = rand(dr, Pz); 
    Cy = rand(dc, Py); Ry = rand(dr, Py); 
    Cz = normc(Cz); Rz = normc(Rz); Cy = normc(Cy); Ry = normc(Ry);
    sigma = sum(sum(X_vec.^2))/(numSpl*dc*dr);
else
    % Specify the initialized model parameters
    Cy = spcInit.Cy; Ry = spcInit.Ry;
    Cz = spcInit.Cz; Rz = spcInit.Rz; 
    sigma = sum(sum(X_vec.^2))/(numSpl*dc*dr);
end

liklhd = zeros(maxIter,1);
Wz = khatrirao(Rz,Cz);
Wy = khatrirao(Ry,Cy);
R = [Ry,Rz]; 
    
%	Column-wise unfolding of TX
ufXc = reshape(permute(TX,[2 3 1]),[numSpl*dr,dc]);
%	Row-wise unfolding of TX
ufXr = reshape(permute(TX,[1 3 2]),[numSpl*dc,dr]);

%   PRODA iterations
for iI = 1 : maxIter    
%   Calc expectations      
    Sigma_z = eye(Pz) / (Wz'*Wz + sigma*eye(Pz)); 
    Psi = eye(dc*dr) - Wz*Sigma_z*Wz'; 
    WPsiWy = Wy'*Psi*Wy;
    
    expY = zeros(Py,nClass);
    expZ = zeros(Pz,numSpl);
    expF = zeros(Py+Pz,numSpl);
    expYY = zeros(Py);
    
    Nc_old = 0;
    for c = 1:nClass
        cIdx = ClsIdxs{c};
        if Nc(c) ~= Nc_old
           Sigma_y = eye(Py) / (Nc(c)*WPsiWy + sigma*eye(Py));
           Nc_old = Nc(c);
        end                
        expY(:,c) = Sigma_y*(Wy'*(Psi*sumX(:,c)));          
        expZ(:,cIdx) = Sigma_z*(Wz'*(X_vec(:,cIdx) - repmat(Wy*expY(:,c),[1 Nc(c)])));
        expF(:,cIdx) = [repmat(expY(:,c), 1, Nc(c));expZ(:,cIdx)];
        expYY = expYY + Nc(c)*Sigma_y;
    end   
    
    expYY = sigma*expYY + expY*diag(Nc)*expY';
    expYX = expY*sumX';
    expZY = Sigma_z*Wz'*(expYX'-Wy*expYY);
    expZZ = numSpl*sigma*Sigma_z + Sigma_z*Wz'*(covXX + Wy*expYY*Wy' - expYX'*Wy' - Wy*expYX)*Wz*Sigma_z;
    
    expFF = [expYY,expZY';expZY,expZZ];
    expFF = expFF + gamma*eye(Py+Pz); % Regularized E[FF]

%   Update C
    XRF = khatrirao(expF',R);
    XRF = ufXc'*XRF;
    FRRF = expFF.*(R'*R);
    C = XRF/FRRF;
    Cy = C(:,1:Py); Cz = C(:,Py+1:end); 
    
%   Update R
    tXCF = khatrirao(expF',C);
    tXCF = ufXr'*tXCF;    
    FCCF = expFF.*(C'*C);
    R = tXCF/FCCF; 
    Ry = R(:,1:Py); Rz = R(:,Py+1:end); 
   
%   Update sigma
    Wy = khatrirao(Ry,Cy);
    Wz = khatrirao(Rz,Cz);    

    sigma = sum(sum(X_vec.^2)) - sum(sum(expF.*([Wy, Wz]'*X_vec)));
    sigma = sigma/(numSpl*dc*dr); 

%   Calc likelihood
    G = Wy*Wy' + Wz*Wz' + sigma*eye(dc*dr); % Total covariance matrix
%   Calc log determinant
    det = logdet(G);
    nloglk = - 0.5*(numSpl*dc*dr*log(2*pi) + numSpl*det + sum(sum(X_vec'/G.*X_vec',2))); 

%	Check convergence:
    liklhd(iI) = nloglk;
    if iI > 1
        threshold = abs(liklhd(iI) - liklhd(iI-1)) / abs(liklhd(iI));
        if threshold < epsCvg 
            disp('Log Likelihood Converge.'); 
            liklhd = liklhd(1:iI);
            break; 
        end
        fprintf('Iteration %u, Likelihood = %f, Threshold = %f.\n', iI, liklhd(iI), threshold);
    else 
        fprintf('Iteration %u, Likelihood = %f.\n', iI, liklhd(1));
    end
end
model.Cy = Cy; model.Ry = Ry; 
model.Cz = Cz; model.Rz = Rz; 
model.sigma = sigma;
model.TXmean = TXmean; 
model.liklhd = liklhd;
end

