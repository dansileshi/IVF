close all
clear
clc

fmt='%s ';  % match vName, string, two numerics
fid=fopen('Book1.csv','r');
data = textscan(fid,fmt,'delimiter',',');

cc = 0;
for ii =1:2:10
    Pre_ = data{1,1};
     cc=ii-1;
    sens_L = str2num(char(Pre_{(cc*72)+5:(cc*72)+5+67}));
    sens_R = str2num(char(Pre_{(cc*72)+77:(cc*72)+77+67}));
    file_name = (char(Pre_{(ii-1)*72+1}));
    
    age = str2num(char(Pre_{((ii-1)*72)+3}));
    MDL = str2num(char(Pre_{((ii-1)*72)+4}));
    MDR = str2num(char(Pre_{((ii-1)*72)+4+72}));
    
    plot_30d2_IVF(sens_L,sens_R ,MDL ,MDR ,age ,file_name);
   
    close all
end



%%


function plot_30d2_IVF(data_L,data_R,MDL,MDR,age,filename)
% prepate index mat
Indx_Mat = Create_idxMat();
% index rows
size(data_L)
Interpol_Mat1 = Prepare_mat(Indx_Mat,data_L);
Interpol_Mat2 = Prepare_mat(Indx_Mat,data_R);
IVF = max(Interpol_Mat1,Interpol_Mat2);
Ivf_ = (IVF(:));
Ivf_ = IVF(~isnan(Ivf_));

genrate_plot(Interpol_Mat1,'left');
genrate_plot(Interpol_Mat2,'right');
genrate_plot(IVF,'ivf');

set(gcf, 'Position', [1, 1, 1900, 1000]);
%% savinf and putting MD values

% compute PSD and MD
mdl = computeSummaryStatistics(data_L, age);
mdr = computeSummaryStatistics(data_R, age);

md_ivf = computeSummaryStatistics(Ivf_, age);

mdl = MDL;
mdr = MDR;
figure(1)
subplot(1,3,1)
text(250,920,['MD: ',num2str(mdl)],'FontSize',18)

subplot(1,3,3)
text(250,920,['MD: ',num2str(mdr)],'FontSize',18)

subplot(1,3,2)
text(250,920,['MD: ',num2str(md_ivf)],'FontSize',18)


% save the fig
%print((filename),'-dpdf','-r300');
%orient(figure(1),'landscape')
print(fullfile(filename),'-djpeg','-r100');


end

function genrate_plot(Interpol_Mat,eye)

pos  =  0;
switch eye
    case 'left'
        pos = 1;
    case 'right'
        pos = 3;
    case 'ivf'
        pos = 2;
    otherwise
        disp('Which eye?')
end

load('Tiles.mat')

figure(1)

subplot(1,3,pos); hold on;


for ii=1:10
    Interpol_Mat(ii,:) = fillmissing(Interpol_Mat(ii,:),'nearest');
end
Interpol_Mat
%Interpol_Mat = interp2(Interpol_Mat,'linear')


% interpolate for every quadrant to crate the 28 X 28 matrix
V = Interpol_Mat;
V1 = V(1:5,1:5);
V2 = V(1:5,6:end);
V3 = V(6:end,1:5);
V4 = V(6:end,6:end);

Big_mat = zeros(28,28);

len = 14;
len1 = 12;

inter1 = 'linear';
st = (len-14)/2+1;
%Q1
[X,Y]  = meshgrid(1:5,1:5);
[Xq,Yq] = meshgrid(linspace(1,5,len),linspace(1,5,len));

V1_ = interp2(X,Y,V1,Xq,Yq,inter1);
Big_mat(1:14,1:14) = V1_(st:st+13,st:st+13) ;

%Q2
V2_ = interp2(X,Y,V2,Xq,Yq,inter1);
Big_mat(1:14,15:end) = V2_(st:st+13,st:st+13);

%Q3
V3_ = interp2(X,Y,V3,Xq,Yq,inter1);
Big_mat(15:end,1:14) = V3_(st:st+13,st:st+13);

%Q4
V4_ = interp2(X,Y,V4,Xq,Yq,inter1);
Big_mat(15:end,15:end) = V4_(st:st+13,st:st+13);


%% interpolate the misssing values
inte2 = 'linear';

Big_mat1 = Big_mat;
Big_mat2 = Big_mat;
for ii = 1:28
    Big_mat1(:,ii) = fillmissing(Big_mat(:,ii),inte2);
end

for ii = 1:28
    Big_mat2(ii,:) = fillmissing(Big_mat(ii,:),inte2);
end

Big_mat = (.5*Big_mat1 + .5*Big_mat2);
%Big_mat= Big_mat1;
%Big_mat = Big_mat2;
Maks_ = Genrate_mask();
Res_MAT = Maks_ .* Big_mat;

Plot_MAT = ones(33*28,31*24,3)+255;
mar = 30;
Res_MAT = Res_MAT+3;
Res_MAT(Res_MAT<0) = 0;
Res_MAT(Res_MAT>35) = 35;

for ii = 1:28
    for jj = 1:28
        if ~isnan(Res_MAT(ii,jj))
            ii_ = ii; jj_ = jj;
            if ii>14
                ii_ = ii_ +1;
            end
            
            if jj_>14
                jj_= jj_+1;
            end
            
            if( Res_MAT(ii,jj)<=0)
                tile = squeeze(Tiles(1,:,:,:));
            else
                tile = squeeze(Tiles((int8(Res_MAT(ii,jj)/5))+1,:,:,:));
            end
            
            %tile = squeeze(Tiles((int8(Res_MAT(ii,jj)/5))+1,:,:,:));
            Plot_MAT(ii_*28:ii_*28+27,jj_*24:jj_*24+23,:) = tile;
            
        end
    end
end
imshow(mat2gray(Plot_MAT),[]);

% draw markers points
%Vertical line
x = [27*14 27*14];
y = [10 31*28];
line(x,y,'Color','k','LineWidth',2)

x = [27*14-7 27*14+7];
yy = [15 30.5*28,25*28,20.5*28,11*28,5*28,];
for ii =1:length(yy)
    y =[yy(ii) yy(ii)];
    line(x,y,'Color','k','LineWidth',2)
    
end

% Hor line
x = [31*14 31*14];
y = [10 31*24];
line(y,x,'Color','k','LineWidth',2)

y = [31*14-7 31*14+7];
xx = [15,5*24,10.*24,20*24,25*24,30.5*24];

for ii =1:length(xx)
    x =[xx(ii) xx(ii)];
    line(x,y,'Color','k','LineWidth',2)
    
end



end
%%

function Interpol_Mat2 = Prepare_mat(Indx_Mat,sens)
Interpol_Mat2 = Indx_Mat;

idx  =0;
for ii =1:10
    for kk =1:10
        if ~isnan(Indx_Mat(ii,kk))
            idx = idx+1;
            Interpol_Mat2(ii,kk ) = sens(idx);
            
        end
    end
    
end
end


function gen_mas = Genrate_mask()
%% Generate mask
rows = [1:9];
cols = [9,7,5,4,3,2,2,1,1];

Mask =  ones(28,28);

for ii =1:length(rows)
    Mask(ii,1:cols(ii))= nan;
    Mask(ii,29-cols(ii):end)= nan;
    
    Mask(29-ii,1:cols(ii))= nan;
    Mask(29-ii,29-cols(ii):end)= nan;
end
gen_mas =Mask;
end


%%
%
function ret_mat =  Create_idxMat()
%% create 26 X 28 matrix to store the interpolated senstivity data

Mat = nan(26,28);
% col index
cols = nan(10,10);

idx = [nan,nan,nan,nan,0];
cols(1,:) = [idx, fliplr(28-idx)];

idx = [nan,nan,0,0,0];
cols(2,:) = [idx, fliplr(28-idx)];

idx =  [nan,0, 0,0,0];
cols(3,:) = [idx, fliplr(28-idx)];
cols(4,:) = cols(3,:);

idx = [0,0,0,0,0];
cols(5,:) = [idx, fliplr(28-idx)];

%lower half of the matrix by fliping the upeer half upside down
lower_half = flipud(cols);

cols(6:end,:)= lower_half(6:end,:);
ret_mat = cols;
end


function [MD]  = computeSummaryStatistics(DLS, age)
% Compute summary statistics
%



weight=[31.1, 30.91, 31.88, 31.82,31.76,31.67,31.54,31.33,32.24,32.48,32.65,...
    32.72,32.7,32.56,32.25,31.74,32.73,33.16,33.46,33.63,33.64,33.47,33.1,...
    32.47,32.45,33.14,33.66,34.02,34.22,34.24,34.06,33.65,32.96,31.93,...
    32.62,33.3,33.81,34.16,34.36,34.38,34.2,33.8,33.11,32.1,33.17,...
    33.59,33.88,34.03,34.05,33.9,33.55,32.96,32.82,33.07,33.24,33.34,...
    33.36,33.27,33.04,32.62,32.45,32.47,32.51,32.54,32.55,32.51,31.85,31.93];

slope =[-0.059 -0.061 -0.056 -0.055 -0.055 -0.056 -0.059 -0.067 -0.053...
    -0.051 -0.054 -0.056 -0.057 -0.056 -0.057 -0.067 -0.048 -0.05 -0.055 ...
    -0.058 -0.058 -0.055 -0.053 -0.056 -0.056 -0.047 -0.048 -0.054 -0.058 ...
    -0.058 -0.054 -0.05 -0.05 -0.063 -0.06 -0.047 -0.047 -0.051 -0.055 -0.055...
    -0.052 -0.047 -0.047 -0.059 -0.051 -0.045 -0.047 -0.05 -0.05 -0.048 -0.045...
    -0.047 -0.059 -0.046 -0.043 -0.043 -0.043 -0.043 -0.044 -0.051 -0.051...
    -0.041 -0.038 -0.038 -0.041 -0.047 -0.039 -0.039];

sens = slope * age + weight;
td = DLS' - sens;
pd = td - quantile(td,0.85);
md = mean(td);

psd = std(pd);

MD =md;


end



