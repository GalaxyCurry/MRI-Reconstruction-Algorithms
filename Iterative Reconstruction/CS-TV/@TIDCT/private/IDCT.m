function res = IDCT(x,blkSize,ovlp)

% res = IDCT(x,blkSize,ovlp)
%
%   local idct implementation
%
% (c) Michael Lustig 2007

N = floor((blkSize)/ovlp);
res = zeros(size(x,1),size(x,2),N,N);

for n=0:N-1
	for m=0:N-1
		res(:,:,n+1,m+1) = circshift(blkproc(x(:,:,n+1,m+1),[blkSize,blkSize],@idct2),[-n,-m]);
	end
end

res = sum(sum(res,3),4)/N;

