function [ newfea ] = projPRODA( TX, model )
% PRODA projection with the trained model
%
% %[Syntax]%: 
%    newfea = projPRODA( TX, model )
%
% %[Inputs]%:
%    TX:            the dc x dr x numSpl input matrix set
%    model:         the trained PRODA model
%
% %[Outputs]%:
%    newfea:        the projected Py-dimensional features
%
% %[Toolbox needed]%:
%   This function needs the tensor toolbox v2.6 available at
%   http://www.sandia.gov/~tgkolda/TensorToolbox/

    Cy = model.Cy; Ry = model.Ry; 
    Cz = model.Cz; Rz = model.Rz; 
    Wy = khatrirao(Ry,Cy); 
    Wz = khatrirao(Rz,Cz); 
    sigma = model.sigma;

    TXmean = model.TXmean;    
    P = size(Wy,2); [dc, dr, numSpl] = size(TX);
    
    TX = bsxfun(@minus,TX, TXmean); % Centering
    X_vec = reshape(TX, dc*dr, numSpl); % Vectorization    
    
    if isempty(Wz) % When Pz = 0
        Sigma = sigma*eye(P) + Wy'*Wy ;
        newfea = Sigma\Wy'*X_vec;
    else
        Sigma = eye(P) + Wy'/(Wz*Wz'+sigma*eye(dc*dr))*Wy;
        newfea = Sigma\Wy'/(Wz*Wz'+sigma*eye(dc*dr))*X_vec;
    end 
end

