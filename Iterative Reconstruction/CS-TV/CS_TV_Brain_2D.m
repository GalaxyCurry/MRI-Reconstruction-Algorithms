% This script demonstare a CS reconstruction from Randomly undersmapled phase encodes of 2D FSE of a brain image.
% FSE-脑部图像的随机欠采样相位编码中进行压缩感知(CS)重建

load brain512

%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% L1 Recon Parameters 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%

N = size(data); 	% image Size
DN = size(data); 	% data Size
TVWeight = 0.002; 	% Weight for TV penalty
xfmWeight = 0.005;	% Weight for Transform L1 penalty
Itnlim = 8;		% Number of iterations


%generate Fourier sampling operator
FT = p2DFT(mask, N, 1, 2);

% scale data
im_dc = FT'*(data.*mask./pdf);
data = data/max(abs(im_dc(:)));%k空间数据
im_dc = im_dc/max(abs(im_dc(:)));%图像域数据

%generate transform operator
XFM = Wavelet('Daubechies',4,4);	% Wavelet

% initialize Parameters for reconstruction
param = init;
param.FT = FT;
param.XFM = XFM;
param.TV = TVOP;
param.data = data;
param.TVWeight =TVWeight;     % TV penalty 
param.xfmWeight = xfmWeight;  % L1 wavelet penalty
param.Itnlim = Itnlim;

figure(100), imshow(abs(im_dc),[]);
colormap('gray');
colorbar();
drawnow;

res = XFM*im_dc;%变换域数据

% do iterations
tic
for n=1:5
	res = fnlCg(res,param);%变换域数据
	im_res = XFM'*res;%图像域数据
	figure(100), imshow(abs(im_res),[]), drawnow
end
toc


figure, imshow(abs(cat(2,im_dc,im_res)),[]);
figure, imshow(abs(cat(2,im_dc(155:304,110:209), im_res(155:304,110:209))),[0,1],'InitialMagnification',200);
title(' zf-w/dc              l_1 Wavelet');




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
