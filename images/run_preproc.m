%allimstr = {'001_1', '001_2', '004_1', '004_2', '004_3', '004_4'};
%allimstr = {'100_1', '100_2'};
allimstr = {'116_1','116_2','116_3','116_4'};

for j=1:length(allimstr)

imstr = allimstr{j};

if exist([imstr, '.png'], 'file')
    Ic=imread([imstr, '.png']);
elseif exist([imstr, '.jpg'], 'file')
    Ic=imread([imstr, '.jpg']);
else exist([imstr, '.jpeg'], 'file')
    Ic=imread([imstr, '.jpeg']);
end

% image scaling and shifting
i1 = impadresize(Ic);
imwrite(uint8(i1),[imstr, '_192x256.png']);
i2 = impadresize(Ic,[128,128]);
imwrite(uint8(i2),[imstr, '_128x128.png']);
i3 = impadresize(Ic,[127,127]);
imwrite(uint8(i3),[imstr, '_127x127.png']);

% fast mask construction
m1 = fastmask(i1);
imwrite(m1,[imstr, '_mask.png']);

end