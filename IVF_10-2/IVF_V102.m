close all 
clear
clc

fmt='%s ';  % match vName, string, two numerics
fid=fopen('Book1.csv','r');
data = textscan(fid,fmt,'delimiter',',');

age = [69,0,73,50,60,73,49,69,64,34,63,79,49,72,82,62,74,51,75,75,66,70,60,39,50];

for ii =1:2:2
    Pre_ = data{1,1};
    
    sens_L = str2num(char(Pre_{5:5+67}));  sens_R = str2num(char(Pre_{77:77+67}));
    file_name = (char(Pre_{1}));
    
    age = str2num(char(Pre_{3}));
    MDL = str2num(char(Pre_{4}));
    MDR = str2num(char(Pre_{4+72}));
    
    plot_30d2_IVF(sens_L,sens_R,MDL,MDR,age,file_name);    
   
    
    close all
    
end



%%


function plot_30d2_IVF(data_L,data_R,MDL,MDR,age,filename)
% prepate index mat
Indx_Mat = Create_idxMat();
% index rows
size(data_L)
Interpol_Mat1 = Prepare_mat(Indx_Mat,data_L);
Interpol_Mat2 = Prepare_mat(Indx_Mat,data_R)
IVF = max(Interpol_Mat1,Interpol_Mat2);

genrate_plot(Interpol_Mat1,'left');
genrate_plot(Interpol_Mat2,'right');
genrate_plot(IVF,'ivf');

set(gcf, 'Position', [1, 1, 1900, 1000]);
%% savinf and putting MD values

% compute PSD and MD
mdl = computeSummaryStatistics(Interpol_Mat1, age, 'left');
mdr = computeSummaryStatistics(Interpol_Mat2, age, 'right');

md_ivf = computeSummaryStatistics(IVF, age, 'left');

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
print(fullfile(filename),'-djpeg','-r300');


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
                tile = squeeze(Tiles((floor(Res_MAT(ii,jj)/5))+1,:,:,:));
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


function [MD]  = computeSummaryStatistics(DLS, age, eye)
% Compute summary statistics
%
% @param    DLS    XXXX
% @param    age  	XXXX
% @param    eye     XXXX
%
% @date     03/04/19
% @author   PRJ
%

%@todo this code is all unchecked

% convert 24-2 to 30-2
if all(size(DLS) == [8 10])
    DLS = [nan(1,10); DLS; nan(1,10)];
end

% get normative values
tt = 'normative';
switch lower(tt)
    case lower('normative')
        DLS_norm_mu_rightEye = [
            NaN  NaN  NaN  26.8 26.3 25.5 25.1 NaN  NaN  NaN    % +27
            NaN  NaN  29.0 29.2 28.3 27.6 27.8 28.0 NaN  NaN    % +21
            NaN  28.9 29.1 29.8 30.2 29.8 30.9 30.4 28.8 NaN    % +15
            28.3 29.6 31.2 32.6 32.3 31.7 31.2 30.6 30.4 30.3   % +9
            28.9 30.4 32.1 32.8 33.7 33.5 32.2 NaN  31.3 31.9   % +3
            29.4 31.0 32.6 33.3 33.8 33.8 32.7 NaN  31.5 31.3   % -3
            29.6 30.4 31.5 33.4 32.9 32.8 32.8 31.1 31.6 31.1   % -9
            NaN  30.5 30.6 31.2 32.1 31.3 30.9 32.4 31.2 NaN    % -15
            NaN  NaN  30.1 29.8 30.9 31.4 31.8 31.2 NaN  NaN    % -21
            NaN  NaN  NaN  29.0 30.1 30.8 30.7 NaN  NaN  NaN    % -27
            ]; %-27  -21  -15   -9   -3   +3   +9  +15  +21  +27
        
        DLS_norm_mu_LEFTEye = [ NaN  NaN  NaN 	22.72	23.04	24.06	23.91	NaN  NaN  NaN
            NaN  NaN  25.64	25.92	26.27	26.74	26.62	25.92 NaN  NaN
            NaN  27.02	27.65	28.15	28.42	28.91	28.67	28.06	26.48	NaN
            27.62	28.38	29	29.68	30.33	30.83	30.8	29.56	27.92	25.72
            28.88	29.52	NaN	30.79	31.92	32.18	31.69	30.66	28.51	26.64
            29.33	29.89	NaN	31.35	32.36	32.42	32.08	31.07	29.18	26.94
            28.92	29.61	30.13	31.08	31.43	31.61	31.65	30.32	28.67	26.41
            NaN 29.19	30.01	30.44	30.38	30.38	30.01	29.06	27.61	NaN
            NaN  NaN 28.95	29.47	29.43	28.91	28.35	27.52   NaN  NaN
            NaN  NaN  NaN  27.78	27.89	27.17	26.47   NaN  NaN  NaN ];
        
        
        DLS_norm_mu_RIGHTEye = [ NaN  NaN  NaN 23.91	24.06	23.04	22.72 NaN  NaN  NaN
            NaN  NaN   25.92	26.62	26.74	26.27	25.92	25.64	NaN  NaN
            NaN  26.48	28.06	28.67	28.91	28.42	28.15	27.65	27.02	NaN
            25.72	27.92	29.56	30.8	30.83	30.33	29.68	29	28.38	27.62
            26.64	28.51	30.66	31.69	32.18	31.92	30.79	NaN	29.52	28.88
            26.94	29.18	31.07	32.08	32.42	32.36	31.35	NaN	29.89	29.33
            26.41	28.67	30.32	31.65	31.61	31.43	31.08	30.13	29.61	28.92
            NaN  27.61	29.06	30.01	30.38	30.38	30.44	30.01	29.19 NaN
            NaN  NaN  27.52	28.35	28.91	29.43	29.47	28.95  NaN  NaN
            NaN  NaN  NaN 26.47	27.17	27.89	27.78 NaN  NaN  NaN];
        
        
        DLS_norm_sd_rightEye = [
            NaN NaN NaN 5.8 3.5 4.7 3.9 NaN NaN NaN
            NaN NaN 3.6 2.6 4.1 2.6 2.6 2.0 NaN NaN
            NaN 2.9 3.2 2.7 2.2 2.4 2.2 2.3 2.7 NaN
            2.7 2.4 1.6 1.7 1.6 2.1 2.1 2.7 2.8 3.2
            2.5 2.2 1.7 1.5 2.0 1.7 1.9 NaN 2.9 2.4
            2.3 1.7 1.4 2.2 1.4 1.4 2.1 NaN 2.5 2.0
            2.2 2.1 2.5 1.7 1.9 1.8 2.2 3.2 2.0 2.6
            NaN 1.8 1.8 2.8 2.1 2.0 2.3 1.9 2.4 NaN
            NaN NaN 2.5 2.4 2.1 1.9 2.1 2.9 NaN NaN
            NaN NaN NaN 3.0 2.2 2.6 2.4 NaN NaN NaN
            ];
        
        age_effect_perDecade_rightEye = [
            NaN  NaN  NaN   -0.8 -0.6 -0.6 -0.6 NaN  NaN NaN
            NaN  NaN  -0.7  -0.9 -0.7 -0.5 -0.6 -0.8 NaN NaN
            NaN  -0.6 -0.6 -0.6 -0.6 -0.8 -0.9 -1.0 -0.7 NaN
            -0.7 -0.6 -0.6 -0.8 -0.6 -0.7 -0.8 -0.7 -0.9 -0.9
            -0.7 -0.6 -0.7 -0.7 -0.7 -0.8 -0.7 NaN  -0.9 -0.9
            -0.9 -0.7 -0.6 -0.6 -0.5 -0.7 -0.5 NaN  -0.8 -0.7
            -0.8 -0.5 -0.6 -0.7 -0.5 -0.6 -0.7 -0.7 -0.7 -0.9
            NaN  -0.9 -0.7 -0.5 -0.7 -0.6 -0.5 -0.8 -0.8 NaN
            NaN  NaN  -0.9 -0.7 -0.7 -0.7 -0.7 -0.6 NaN  NaN
            NaN  NaN  NaN  -0.6 -0.8 -0.6 -0.6 NaN  NaN  NaN
            ];
        
        switch lower(eye)
            case 'left'
                DLS_norm_mu       	=  DLS_norm_mu_rightEye;%DLS_norm_mu_LEFTEye
                DLS_norm_var     	= fliplr(DLS_norm_sd_rightEye).^2;
                age_effect_perYear  = fliplr(age_effect_perDecade_rightEye)/10;
            case 'right'
                DLS_norm_mu       	= fliplr(DLS_norm_mu_rightEye); %  DLS_norm_mu_RIGHTEye
                DLS_norm_var      	= DLS_norm_sd_rightEye.^2;
                age_effect_perYear	= age_effect_perDecade_rightEye/10;
            case 'ivf'
                DLS_norm_mu      	= max(DLS_norm_mu_RIGHTEye, fliplr(DLS_norm_mu_rightEye));
                DLS_norm_var     	= min(DLS_norm_sd_rightEye, fliplr(DLS_norm_sd_rightEye)).^2;
                age_effect_perYear	= min(age_effect_perDecade_rightEye, fliplr(age_effect_perDecade_rightEye))/10;
            otherwise
                error('error not recognised');
        end
        
end

% get expected values
normalDLS = (age_effect_perYear * (age-40)) + DLS_norm_mu;
% compute deviation from normative values
td = DLS - normalDLS;
% compute MD
idx = ~isnan(td);
aa = sum(td(idx) .* (1./DLS_norm_var(idx)));
bb = sum((1./DLS_norm_var(idx)));
MD = aa/bb;

% Compute pattern standard deviation
%psd = sqrt(cov(pd));
PSD = NaN; % not yet implemented
end % all done!


