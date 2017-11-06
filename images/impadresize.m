function Iout = impadresize(Iin,sizei)
% IMPADRESIZE resizes the image without changing the aspect ratio, then
%     pads the image to the desired size
if nargin<2
    sizei = [192,256];
end

[h,w,c]=size(Iin);
bgc = double(Iin(1));  % a coarse background color estimate
Iout = bgc*ones(sizei(1), sizei(2), c);

[fac,ii] = min([sizei(1)*0.9/h, sizei(2)*0.9/w]);
Irsz = imresize(Iin, round(fac*[h,w]));
if ii==1  % thin image
    idx1 = round(sizei(2)/2-fac*w/2);
    idy1 = round(0.05*sizei(1));
else  % wide image
    idx1 = round(0.05*sizei(2));
    idy1 = round(sizei(1)/2-fac*h/2);
end
Iout(idy1:idy1+round(fac*h)-1,idx1:idx1+round(fac*w)-1,:) = Irsz;