function ptsgif(ptsfile,nviews,isobj)

pts=dlmread(ptsfile,' ');
if nargin<3 || isempty(isobj)
    isobj=0;
end
if nargin<2 || isempty(nviews)
    nviews=50;
end

%%
if isobj
    ptsxyz=pts;
else
    ptsxyz=[pts(:,2),-pts(:,1),pts(:,3)];
end

theta=linspace(0,2*pi,nviews+1);

h=figure;
filename=[ptsfile,'.gif'];
for i=1:nviews
    ang=theta(i);
    rmat=[cos(ang), 0, -sin(ang); 0, 1, 0; sin(ang), 0, cos(ang)];
    ptsrot=ptsxyz*rmat;
    plot(ptsrot(:,1),ptsrot(:,2),'.','markersize',15);
    axis([-2.5,2.5,-1.5,2.5]);
    set(gca,'visible','off');
    drawnow
    
    frame = getframe(h);
    im = frame2im(frame); 
    [imind,cm] = rgb2ind(im,256);
    if i == 1 
      imwrite(imind,cm,filename,'gif','Loopcount',inf); 
    else
      imwrite(imind,cm,filename,'gif','DelayTime',0.1,'WriteMode','append'); 
    end
end