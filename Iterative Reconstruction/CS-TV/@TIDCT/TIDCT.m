function res = TIWDCT(blkSize, ovlp)
% res = TIWDCT(blkSize, ovlp)
% 
% Implements a segmented DCT operator
%
% (c) Michael Lustig 2007

res.adjoint = 0;
res.blkSize = blkSize;
res.ovlp = ovlp;
res = class(res,'TIDCT');
