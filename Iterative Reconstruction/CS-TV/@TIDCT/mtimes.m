function res = mtimes(a,b)


if isa(a,'TIDCT') == 0
    error('In  A.*B only A can be Wavelet operator');
end


if a.adjoint
	res = IDCT(b,a.blkSize,a.ovlp);
else
	res = FDCT(b,a.blkSize,a.ovlp);
end


