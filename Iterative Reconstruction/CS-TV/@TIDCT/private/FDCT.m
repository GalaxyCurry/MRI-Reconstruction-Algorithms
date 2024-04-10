function res = FDCT(x,blkSize,ovlp)

% res = FDCT(x,blkSize,ovlp)
%
%   local dct implementation blkSize x blkSize blocks
%
% (c) Michael Lustig 2007

N = floor((blkSize)/ovlp);
res = zeros(size(x,1),size(x,2),N,N);

for n=0:N-1
	for m=0:N-1
		res(:,:,n+1,m+1) = blkproc(circshift(x,[n,m]),[blkSize,blkSize],@dct2);
	end
end

res = res/N;
	
