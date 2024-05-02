%load the images
Yhim =  double(readgeoraster('path...\PRISMA_HS')); %PRISMA HS image
Ymim = double(readgeoraster('path...\Seninel-2')); %Sentinel-2 (or PAN, for pansharpening) image


downsamp_factor = 3; %or 6; ratio between the GSD of coarser and finest resolution images


%bands corrispondences between PRISMA and Sentinel-2 images. Es.: the first Sentinel band spectrally corresponds from 9th to 16th PRISMA bands, and so on
intersection{1} = [9:16]';
intersection{2} = [20:22]';
intersection{3} = [32:34]';
intersection{4} = [37:38]';
intersection{5} = [40:41]';
intersection{6} = [44:45]';
intersection{7} = [45:48]'; 
intersection{8} = [52:53]';
intersection{9} = [118:129]'; 
intersection{10} = [175:204]';
%
%
%intersection{1} = [6:36]'; %in case of pansharpening application
%
%
contiguous = intersection; 


%set model parameters
shift = 0; % default
blur_center = 0; % default
basis_type={'VCA'}

%set for parameters tuning loop
lambda_R=[80 10 2];
lambda_B=[80 10 2];
p=[80 50 20];
lambda_phi=[0.1 0.2 0.5];
lambda_m=[1 20 80];
hsize_h = [5 10 15];
hsize_w = [5 10 15];

for w=1:length(basis_type)
    for i=1:length(lambda_R)
        for j=1:length(lambda_B)
            for k=1:length(p)
                for q=1:length(lambda_phi)
                    for z=1:length(lambda_m)
                        for g=1:length(hsize_h)
                            disp(['lambda_R = ',num2str(lambda_R(i)),', lambda_B = ',num2str(lambda_B(j)),' p = ',num2str(p(k)),' lambda_phi = ',num2str(lambda_phi(q)),' lambda_m = ',num2str(lambda_m(z))]);
                            [V, R_est, B_est] = sen_resp_est(Yhim, Ymim, downsamp_factor, intersection, contiguous, p(k), lambda_R(i), lambda_B(j), hsize_h(g), hsize_w(g), shift, blur_center);
                            iters = 200;
                            mu = 0.05;
                            Zimhat = data_fusion(Yhim, Ymim, downsamp_factor, R_est, B_est, p(k), basis_type{w}, lambda_phi(q), lambda_m(z), iters, mu);
                            Zhat = im2mat(Zimhat);
                            Zhat_denoised = (V*V')*Zhat;
                            [nl, nc, ~] = size(Ymim);
                            Zimhat_denoised = mat2im(Zhat_denoised, nl);
                            save(['Result_',basis_type{w},'_R_',num2str(lambda_R(i)),'_B_',num2str(lambda_B(j)),'_p_',num2str(p(k)),'_phi_',num2str(lambda_phi(q)),'_m_',num2str(lambda_m(z)),'_wind_',num2str(hsize_h(g)),'.mat'],'Zimhat_denoised','-v7.3');
                            clearvars  Zimhat Zhat Zhat_denoised Zimhat_denoised
                        end
                    end
                end
            end
        end
    end
end