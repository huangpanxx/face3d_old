function showface(path)
figure(1);
dep = -40;
if ~isdir(path)
    path = strcat([path,'/../']);
end
disp('loading...');

map = load(strcat([path,'/MAP']));
a = load(strcat([path,'/A']));
B = load(strcat([path,'/B']));
BOUND = load(strcat([path,'/BOUND']));
F = load(strcat([path,'/F']));
img = imread(strcat([path,'/I.bmp']));
s = size(img);
face_size = s(1);

disp('solving...');
i = a(:,1);
j = a(:,2);
s = a(:,3);
A = sparse(i,j,s);
x = B/A;
r = floor(map(:,2)/face_size);
c = mod(map(:,2),face_size);
Z = ones(face_size,face_size)*NaN;
sz = size(A);

x(BOUND+1) = NaN;
x(BOUND) = NaN;

s = 0;
cnt = 0;
for i = 1:sz(1)
    v = x(i);
    if ~isnan(v) && v > dep
        Z(r(i)+1,face_size+2-c(i)) = v;
        s = s + v;
        cnt = cnt + 1;
    end
end

avg = s/cnt;
Z = Z - avg;

subplot(2,2,1);
surf(Z);
colormap(pink);
shading interp;
light('position',[0,0,1500]);
material dull;
w = face_size+100;
axis([100-w/2,100+w/2,100-w/2,100+w/2,-w/2,w/2]);

subplot(2,2,2);
Z2 = ones(face_size,face_size)*NaN;
sz = size(Z);
for i = 1:sz(1)
    for j = 1:sz(2)
        Z2(i,j) = Z(i,sz(2)+1-j);
    end
end
surf(Z2);
warp(Z2,img);
shading interp;
axis([100-w/2,100+w/2,100-w/2,100+w/2,-w/2,w/2]);


subplot(2,2,3);
x = F(:,1);
y = F(:,2);
z = F(:,3);
sl = z > dep;
x =  face_size+1-x(sl);
y = y(sl);
z = z(sl);
[xq,yq] = meshgrid(1:face_size,1:face_size);
zq = griddata(x,y,z,xq,yq,'cubic');%拟合
surf(zq);%表面
colormap(pink);
shading interp;
light('position',[0,0,1500]);
material dull;
axis([100-w/2,100+w/2,100-w/2,100+w/2,-w/2,w/2]);


subplot(2,2,4);
[xq,yq] = meshgrid(1:face_size,1:face_size);
zq = griddata(face_size+1-x,y,z,xq,yq,'cubic');%拟合
surf(zq);
warp(zq,img);
shading interp;
axis([100-w/2,100+w/2,100-w/2,100+w/2,-w/2,w/2]);
end