% this is a script to demonstrate a TV recon from undersampled variable density spirals
% 从欠采样的变密度螺旋采样数据进行TV重建

%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% L1 Recon Parameters 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%

N = [160,160]; 		% image Size
TVWeight = 0.01; 	% Weight for TV penalty
xfmWeight = 0.00;	% Weight for Transform L1 penalty
Itnlim = 25;		% Number of iterations

% generate a variable density spiral with gaussian density.
SIGMA = 3;
FOV = gausswin(40,SIGMA);
FOV = FOV(end/2+1:end);
FOV = FOV*10+7;
RADIUS = linspace(0,1,20);
RESOLUTION = 1; % in mm, such that image is 256x256
NITLV = 16; % number of spiral interleaves
Gmax = 4 ; % maximum gradient in [G/CM];
Smax = 15; %Maximum slew-rate;
T = 4e-3; % time sampling (in mS);


[k] = vdSpiralDesign(NITLV, RESOLUTION,FOV,RADIUS,Gmax,Smax,T,'cubic');
k = k(2:end).'*exp(2*pi*1i*[1:NITLV]/NITLV);
k = k(:)/max(abs(k(:)))/2; % scale to range [-0.5,0.5]
w = voronoidens(k); % calculate voronoi density compensation function
w = w/max(w(:));

% generate circular mask (spirals hav a circular FOV support
[xx,yy] = meshgrid(linspace(-1,1,N(1)));
ph = double(sqrt(xx.^2 + yy.^2)<1);

%generate image
im = (phantom(N(1)))  + randn(N)*0.01 + 1i*randn(N)*0.01;

%generate Fourier sampling operator
FT = NUFFT(k,1, ph, 0,N, 2);

% scale w
tmp=zeros(N);
tmp(end/2+1,end/2+1)=1; 
tmp=FT'*(w.*(FT*tmp)); 
w = w/max(abs(tmp(:)));
data = FT*im;
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

im_dc = FT'*(data.*w);	% init with zf-w/dc (zero-fill with density compensation)
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


im_full = ifft2c(zpad(fft2c(im),N(1),N(2)));
figure, imshow(abs(cat(2,im_full,im_dc,im_res)),[0,1]);
title('original                      zf-w/dc              TV');

figure, plot(1:N(1), abs(im_full(end/2,:)), 1:N(2), abs(im_dc(end/2,:)), 1:N(2), abs(im_res(end/2,:)),'LineWidth',2);
legend('original', 'zf-w/dc', 'TV');






function [k,g,s,time] = vdSpiralDesign(Nitlv, res, fov,radius , Gmax, Smax, T, interpType)
% This function designs a variable density spiral
%	Input:
%		Nitlv	-	Number of interleves
%		res	-	resolution (in mm)
%		fov	- 	vector of fov (in cm)
%		radius	-	vector of radius corresponding to the fov
%		Gmax	-	max gradient (default 4 G/CM)
%		Smax	-	max slew (default 15)
%		T	-	sampling rate (in ms)
%		interpType- 	type of interpolation used to interpolate the fov
%				accept: linear, cubic, spline
%	Output:
%		k	-	the k-space trajectory
%		g	-	the gradient waveform
%		s	-	the slew rate
%		time	-	total time



kmax = 5/res;

if length(radius)<2
	error(' radius must be at least length=2');
end

dr = 1/1500/max(fov/Nitlv);
r = 0:dr:kmax;   kmax = max(r);

fov =  interp1(radius*kmax,fov,r,interpType);
dtheta = 2*pi*dr.*fov/Nitlv;
theta = cumsum(dtheta);

C = r.*exp(1i*theta);
[C,time,g,s,k] = minTimeGradient(C, [], [], Gmax, Smax,T,0);


k = C;
end



function area = voronoidens(k)
% input:  k = kx + i ky is the  k-space trajectory
% output: area of cells for each point 
%           (if point doesn't have neighbors the area is NaN)

r = max(abs(k(:)));
k = [k(:); r*1.005*exp(1i*2*pi*[1:256]'/256)];
kx = real(k);
ky = imag(k);

[row,column] = size(kx);

% uncomment these to plot voronoi diagram
% [vx, vy] = voronoi(kx,ky);
%figure, plot(kx,ky,'r.',vx,vy,'b-'); axis equal

kxy = [kx(:),ky(:)];
% returns vertices and cells of voronoi diagram
[V,C] = voronoin(kxy); 
area = [];
for j = 1:length(kxy)
 if ~isempty(C{j})
  x = V(C{j},1); y = V(C{j},2); lxy = length(x);
  A = abs(sum( 0.5*(x([2:lxy 1]) - x(:)).*(y([2:lxy 1]) + y(:))));
 else 
	 A = inf;
 end
  area = [area A];
end

area = area(1:end-256);
area = area(:)/sum(area(:));
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