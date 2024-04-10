% 随机欠采样3DFT采集
% The data was undersampled in the acquisition by a factor of 2
% the zero filling with density compensation, not surprizingly, result in a good MIP, but has
% significantly more "noise" the slices than the CS recon.
%零填充加密度补偿的方法,虽然能产生较好的最大强度投影(MIP)图像,但与压缩感知(CS)重建相比,其结果在切片层面上会有明显更多的"噪声"


load calf_data_cs.mat


% take ifft in the fully sampled dimension
data = fftshift(ifft(fftshift(data,1),[],1),1);
data = permute(data,[2,3,1]);

im_zfwdc = zeros(size(data));
for n=1:size(data,3)
	im_zfwdc(:,:,n) = ifft2c(data(:,:,n)./pdf);
end

% scale data such that the maximum image pixel in zf-w/dc is around 1
% this way, we can use similar lambda for different problems
data = data/max(abs(im_zfwdc(:)));
im_zfwdc = im_zfwdc/max(abs(im_zfwdc(:)));


%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% L1 Recon Parameters 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%

N = [size(data,1),size(data,2)]; 		% image Size
DN = N;		 	% data Size
TVWeight = [0.01] ; 	% Weight for TV penalty - only TV is on, but I encourage you to try wavelets as well.
xfmWeight = [0.0];	% Weight for Transform L1 penalty
Itnlim = 45;		% Number of iterations
OuterIt = length(TVWeight);

%generate transform operator

%XFM = Wavelet('Daubechies',4,4);	% Wavelet
%XFM = TIDCT(8,4);			% DCT
XFM = 1;				% Identity transform 	

% initialize Parameters for reconstruction
phmask = zpad(hamming(6)*hamming(6)',N(1),N(2)); %mask to grab center frequency
phmask = phmask/max(phmask(:));			 %for low-order phase estimation and correction
res = zeros(N);
RES = zeros(size(data));

param = init;
param.XFM = XFM;
param.TV = TVOP;
param.Itnlim = Itnlim;

tic
for slice = 1:size(data,3)

	param.data = data(:,:,slice);
	ph = exp(1i*angle((ifft2c(data(:,:,slice).*phmask)))); % estimate phase for phase correction
	param.FT = p2DFT(mask, DN, ph, 2); 

	for n=1:OuterIt
		param.TVWeight =TVWeight(n);     % TV penalty 
		param.xfmWeight = xfmWeight(n);  % L1 wavelet penalty
		res = fnlCg(res, param);
		figure(100), imshow(cat(2,abs(im_zfwdc(:,:,slice)),abs(XFM'*res)),[]), drawnow;
	end
	RES(:,:,slice) = XFM'*res;
	if mod(slice,10)==0
		figure(101), imshow(cat(2,max(abs(permute(im_zfwdc,[3,1,2])),[],3),max(abs(permute(RES,[3,1,2])),[],3)),[]), drawnow;
	end
end

toc

disp('Done!, now computing MIPs');

MIP_dc = zeros(size(data,3),N(1),36);
MIP_cs = zeros(size(data,3),N(1),36);

for n=1:36
	MIP_dc(:,:,n) = max(abs(permute(imrotate(im_zfwdc,10*(n-1),'bilinear','crop'),[3,1,2])),[],3);
	MIP_cs(:,:,n) = max(abs(permute(imrotate(RES,10*(n-1),'bilinear','crop'),[3,1,2])),[],3);
	figure(101), imshow(cat(2,MIP_dc(:,:,n),MIP_cs(:,:,n)),[],'InitialMagnification',150), title('zf-w/dc                           CS'), drawnow;
end



function res = zpad(x,sx,sy)
%  Zero pads a 2D matrix around its center.

[mx,my] = size(x);
res = zeros(sx,sy);
    
idxx = sx/2+1-mx/2 : sx/2+mx/2;
idxy = sy/2+1-my/2 : sy/2+my/2;

res(idxx,idxy) = x;
end



function res = init()
% param = init()
% function returns a structure with the entries that are needed for the reconstruction.

res.FT = []; % The measurement operator (undersmapled Fourier for example)
res.XFM = []; % Sparse transform operator
res.TV = []; 	% the Total variation operator
res.data = []; % measurements to reconstruct from

res.TVWeight = 0.01;	% TV penalty
res.xfmWeight = 0.01;   % transform l1 penalty

res.Itnlim = 20;	% default number of iterations
res.gradToll = 1e-30;	% step size tollerance stopping criterea (not used)

res.l1Smooth = 1e-15;	% smoothing parameter of L1 norm
res.pNorm = 1;  % type of norm to use (i.e. L1 L2 etc)

% line search parameters
res.lineSearchItnlim = 150;
res.lineSearchAlpha = 0.01;
res.lineSearchBeta = 0.6;
res.lineSearchT0 = 1 ; % step size to start with
end



function x = fnlCg(x0,params)
%-----------------------------------------------------------------------
%
% res = fnlCg(x0,params)
%
% implementation of a L1 penalized non linear conjugate gradient reconstruction
%
% The function solves the following problem:
%
% given k-space measurments y, and a fourier operator F the function 
% finds the image x that minimizes:
%
% Phi(x) = ||F* W' *x - y||^2 + lambda1*|x|_1 + lambda2*TV(W'*x) 
%
%
% the optimization method used is non linear conjugate gradient with fast&cheap backtracking
% line-search.结合非线性共轭梯度法和快速回溯线搜索的优化方法
%-------------------------------------------------------------------------
x = x0;%变换域数据


% line search parameters
maxlsiter = params.lineSearchItnlim ;
gradToll = params.gradToll ;
alpha = params.lineSearchAlpha;   
beta = params.lineSearchBeta;
t0 = params.lineSearchT0;
k = 0;
t = 1;

% copmute g0  = grad(Phi(x))

g0 = wGradient(x,params);

dx = -g0;


% iterations
while(1)

% backtracking line-search

	% pre-calculate values, such that it would be cheap to compute the objective
	% many times for efficient line-search
    % 预先计算中间值的方法,在优化算法中很常见,尤其是在需要反复计算目标函数和梯度的情况下,比如线搜索、牛顿法等。
    % 通过这种方式,可以显著降低计算开销,提高优化算法的收敛速度
	[FTXFMtx, FTXFMtdx, DXFMtx, DXFMtdx] = preobjective(x, dx, params);
	f0 = objective(FTXFMtx, FTXFMtdx, DXFMtx, DXFMtdx,x,dx, 0, params);
	t = t0;
        [f1, ERRobj, RMSerr]  =  objective(FTXFMtx, FTXFMtdx, DXFMtx, DXFMtdx,x,dx, t, params);
	
	lsiter = 0;

	while (f1 > f0 - alpha*t*abs(g0(:)'*dx(:)))^2 && (lsiter<maxlsiter)
		lsiter = lsiter + 1;
		t = t * beta;
		[f1, ERRobj, RMSerr]  =  objective(FTXFMtx, FTXFMtdx, DXFMtx, DXFMtdx,x,dx, t, params);
	end

	if lsiter == maxlsiter
		disp('Reached max line search,.... not so good... might have a bug in operators. exiting... ');
		return;
	end

	% control the number of line searches by adapting the initial step search
	if lsiter > 2
		t0 = t0 * beta;
	end 
	
	if lsiter<1
		t0 = t0 / beta;
	end

	x = (x + t*dx);

	%--------- uncomment for debug purposes ------------------------	
	disp(sprintf('%d  , obj: %f, RMS: %f, L-S: %d', k,f1,RMSerr,lsiter));

	%---------------------------------------------------------------
	
    %conjugate gradient calculation
    
	g1 = wGradient(x,params);
	bk = g1(:)'*g1(:)/(g0(:)'*g0(:)+eps);
	g0 = g1;
	dx =  - g1 + bk* dx;
	k = k + 1;
	
	%TODO: need to "think" of a "better" stopping criteria ;-)
	if (k > params.Itnlim) || (norm(dx(:)) < gradToll) 
		break;
	end

end


return;
end


function [FTXFMtx, FTXFMtdx, DXFMtx, DXFMtdx] = preobjective(x, dx, params)

% precalculates transforms to make line search cheap

FTXFMtx = params.FT*(params.XFM'*x);%k空间数据
FTXFMtdx = params.FT*(params.XFM'*dx);%k空间数据

if params.TVWeight
    DXFMtx = params.TV*(params.XFM'*x);
    DXFMtdx = params.TV*(params.XFM'*dx);
else
    DXFMtx = 0;
    DXFMtdx = 0;
end
end





function [res, obj, RMS] = objective(FTXFMtx, FTXFMtdx, DXFMtx, DXFMtdx, x,dx,t, params)
%calculated the objective function

p = params.pNorm;

obj = FTXFMtx + t*FTXFMtdx - params.data;
obj = obj(:)'*obj(:);

if params.TVWeight
    w = DXFMtx(:) + t*DXFMtdx(:);
    TV = (w.*conj(w)+params.l1Smooth).^(p/2); 
else
    TV = 0;
end

if params.xfmWeight
   w = x(:) + t*dx(:); 
   XFM = (w.*conj(w)+params.l1Smooth).^(p/2);
else
    XFM=0;
end



TV = sum(TV.*params.TVWeight(:));
XFM = sum(XFM.*params.xfmWeight(:));
RMS = sqrt(obj/sum(abs(params.data(:))>0));

res = obj + (TV) + (XFM) ;
end

function grad = wGradient(x,params)

gradXFM = 0;
gradTV = 0;

gradObj = gOBJ(x,params);
if params.xfmWeight
gradXFM = gXFM(x,params);
end
if params.TVWeight
gradTV = gTV(x,params);
end

grad = (gradObj +  params.xfmWeight.*gradXFM + params.TVWeight.*gradTV);
end



function gradObj = gOBJ(x,params)
% computes the gradient of the data consistency

	gradObj = params.XFM*(params.FT'*(params.FT*(params.XFM'*x) - params.data));

gradObj = 2*gradObj ;
end

function grad = gXFM(x,params)
% compute gradient of the L1 transform operator

p = params.pNorm;

grad = p*x.*(x.*conj(x)+params.l1Smooth).^(p/2-1);
end


function grad = gTV(x,params)
% compute gradient of TV operator

p = params.pNorm;

Dx = params.TV*(params.XFM'*x);

G = p*Dx.*(Dx.*conj(Dx) + params.l1Smooth).^(p/2-1);
grad = params.XFM*(params.TV'*G);
end

