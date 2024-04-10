% this is a script to demonstrate the original experiment by Candes, Romberg and Tao
% 复现 Candes、Romberg 和 Tao 最初提出的压缩感知理论在实际图像重建中的应用,展示在处理欠采样数据方面的优势。


%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% L1 Recon Parameters 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%

N = [256,256]; 		% image Size
DN = [256,256]; 	% data Size
pctg = [0.33];  	% undersampling factor
P = 5;			% Variable density polymonial degree
TVWeight = 0.01; 	% Weight for TV penalty
xfmWeight = 0.00;	% Weight for Transform L1 penalty
Itnlim = 8;		% Number of iterations


% generate variable density random sampling
pdf = genPDF(DN,P,pctg , 2 ,0.1);	% generates the sampling PDF
k = genSampling(pdf,10,60);		% generates a sampling pattern %% mask

%generate image
im = (phantom(N(1)))  + randn(N)*0.01 + 1i*randn(N)*0.01;% 图像域数据

%generate Fourier sampling operator
FT = p2DFT(k, N, 1, 2);
data = FT*im; % k空间频域数据

%generate transform operator

%XFM = Wavelet('Daubechies',6,4);	% Wavelet
%XFM = TIDCT(8,4);			% DCT
XFM = 1;				% Identity transform 	

% initialize Parameters for reconstruction
param = init;
param.FT = FT;
param.XFM = XFM;
param.TV = TVOP;
param.data = data;
param.TVWeight =TVWeight;     % TV penalty 
param.xfmWeight = xfmWeight;  % L1 wavelet penalty
param.Itnlim = Itnlim;

im_dc = FT'*(data./pdf);	% init with zf-w/dc (zero-fill with density compensation)
figure(100), imshow(abs(im_dc),[]);drawnow;

res = XFM*im_dc;

% do iterations
tic
for n=1:8
	res = fnlCg(res,param);
	im_res = XFM'*res;
	figure(100), imshow(abs(im_res),[]), drawnow
end
toc


% create a low-res mask
mask_lr = genLRSampling_pctg(DN,pctg,1);
im_lr = ifft2c(zpad(fft2c(im).*mask_lr,N(1),N(2)));

im_full = ifft2c(zpad(fft2c(im),N(1),N(2)));
figure, imshow(abs(cat(2,im_full,im_lr,im_dc,im_res)),[]);
title('original             low-res              zf-w/dc              TV');

figure, plot(1:N(1), abs(im_full(end/2,:)),1:N(1), abs(im_lr(end/2,:)), 1:N(2), abs(im_dc(end/2,:)), 1:N(2), abs(im_res(end/2,:)),'LineWidth',2);
legend('original', 'LR', 'zf-w/dc', 'TV');




function mask = genLRSampling_pctg(imSize,pctg,distType)
% 百分比值代表了圆形遮罩所覆盖的像素数占整个图像像素总数的比例

sx = imSize(1);
sy = imSize(2);


	
[x,y] = meshgrid(linspace(-1,1,sy),linspace(-1,1,sx));
switch distType
	case 1
		r = max(abs(x),abs(y));
	otherwise
		r = sqrt(x.^2+y.^2);
		r = r/max(abs(r(:)));			
end


[nothing, circOrder] = sort(r(:));

mask = zeros(imSize);
mask(circOrder(1:floor(pctg*sx*sy))) = 1;

end




function [pdf,val] = genPDF(imSize,p,pctg,distType,radius)
%	多项式分布的采样密度
%	Input:
%		imSize - size of matrix or vector
%		p - power of polynomial
%		pctg - partial sampling factor e.g. 0.5 for half
%		distType - 1 or 2 for L1 or L2 distance measure
%		radius - radius of fully sampled center
%	Output:
%		pdf - the pdf
%		val - min sampling density

minval=0;
maxval=1;
val = 0.5;

if length(imSize)==1
	imSize = [imSize,1];
end

sx = imSize(1);
sy = imSize(2);
PCTG = floor(pctg*sx*sy);


[x,y] = meshgrid(linspace(-1,1,sy),linspace(-1,1,sx));
switch distType
	case 1
		r = max(abs(x),abs(y));
	otherwise
		r = sqrt(x.^2+y.^2);
		r = r/max(abs(r(:)));			
end

idx = find(r<radius);

pdf = (1-r).^p; pdf(idx) = 1;
if floor(sum(pdf(:))) > PCTG
	error('infeasible without undersampling dc, increase p');
end

% begin bisection
while(1)
	val = minval/2 + maxval/2;
	pdf = (1-r).^p + val; pdf(find(pdf>1)) = 1; pdf(idx)=1;
	N = floor(sum(pdf(:)));
	if N > PCTG % infeasible
		maxval=val;
	end
	if N < PCTG % feasible, but not optimal
		minval=val;
	end
	if N==PCTG % optimal
		break;
	end
end

end



function [minIntrVec,stat,actpctg] = genSampling(pdf,iter,tol)


%   使用蒙特卡洛方法来生成一个采样模式,目标是使得峰值干扰最小。 其中,"sum(pdf) ± tol"指的是目标采样数量。
%	pdf - probability density function to choose samples from
%	iter - number of tries
%	tol  - the deviation from the desired number of samples in samples
% returns:
%	mask - sampling pattern
%	stat - vector of min interferences measured each try
%	actpctg    - actual undersampling factor


pdf(find(pdf>1)) = 1;
K = sum(pdf(:));

minIntr = 1e99;
minIntrVec = zeros(size(pdf));

for n=1:iter
	tmp = zeros(size(pdf));
	while abs(sum(tmp(:)) - K) > tol
		tmp = rand(size(pdf))<pdf;
	end
	
	TMP = ifft2(tmp./pdf);
	if max(abs(TMP(2:end))) < minIntr
		minIntr = max(abs(TMP(2:end)));
		minIntrVec = tmp;
	end
	stat(n) = max(abs(TMP(2:end)));
end

actpctg = sum(minIntrVec(:))/prod(size(minIntrVec));
end



function res = zpad(x,sx,sy)
%  Zero pads a 2D matrix around its center.

[mx,my] = size(x);
res = zeros(sx,sy);
    
idxx = sx/2+1-mx/2 : sx/2+mx/2;
idxy = sy/2+1-my/2 : sy/2+my/2;

res(idxx,idxy) = x;
end


function res = ifft2c(x)
%将输入的二维数据进行正交归一化、中心化处理,然后再执行二维逆快速傅里叶变换,得到最终的空间域重建结果

res = sqrt(length(x(:)))*ifftshift(ifft2(fftshift(x)));
end

function res = fft2c(x)

res = 1/sqrt(length(x(:)))*fftshift(fft2(ifftshift(x)));
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