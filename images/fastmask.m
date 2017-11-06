function maskI = fastmask(I, thr)

if nargin<2
    thr = 10;
end

if size(I,3)==3
    I = double(rgb2gray(uint8(I)));
else
    I = double(I);
end
if I(1)<128  % a coarse white/black background color inference
    m = 255*double(I<=thr);
else
    m = 255*double(I>=255-thr);
end

% TODO: more binary map processing
maskI = m;