clear all; clc;
addpath('E:\0UNSW_PhD\Code\particle filter')
addpath("..\Multi_rater_SMC\gpml-matlab-master")
addpath("..\Multi_rater_SMC\my_funtion")
startup
%%
load('RECOLAR_all.mat')
%%
global win overlap pcadim Ns
win = 25;
overlap = 12;
pcadim = 40;
Ns = 100; % number of particles
delay = 4/0.04; % 4 second for arousal and 2 second for valence
window_level = 1;
multi_rater = 1;
%% Load labels for GP
for u = 1:9
        [ave_train{u,1}] = my_window_rating(arousal_train{u}(:,2:7),win,overlap);
        [ave_test{u,1}] = my_window_rating(arousal_dev{u}(:,2:7),win,overlap);
end
%% Train Gaussian Process Parameters
time = [1:length(ave_train{1})]';
for u = 1:9
    mean_func{u} = mean(ave_train{u}');
end
meanfunc = [];                    % empty: don't use a mean function
covfunc = @covSEiso;              % Squared Exponental covariance function
likfunc = @likGauss;              % Gaussian likelihood
% Finally we initialize the hyperparameter struct
hyp = struct('mean', [], 'cov', [0 0], 'lik', -1);
%hyp.mean = mean_func;
if multi_rater == 1
    inf_fun = @my_infGaussLik_9utt_6rater;
else
    inf_fun = @my_infGaussLik_9utt;
end
hyp2 = minimize(hyp, @gp, -100, @my_infGaussLik_9utt_6rater, meanfunc, covfunc, likfunc, time, ave_train); % Training Phase with 9 training utterance cacatanent
%% Transition Model equation y[k] = my_sys(k, y[1:k-1]);
my_sys = @(hyp2, infGaussLik, meanfunc, covfunc, likfunc, past_time, past_states, target_time)my_gp_test_phase(hyp2, infGaussLik, meanfunc, covfunc, likfunc, past_time, past_states, target_time);
GP_cc = my_GP_validation(my_sys,hyp2,@infGaussLik,meanfunc,covfunc,likfunc, ave_test{1}(1:300,1)); % Validate GP transition Model
%% Process labels and Features for GMM1 
load( ['boaw_100_ar_mfcc39.mat']);
[train_feature,test_feature] = pca_delta_features(data_train,data_dev,pcadim);
[ave_train,ave_test,ave_train_feature,ave_test_feature] = delay_comp_window_segment(arousal_train,arousal_dev,...,
    train_feature,test_feature,delay,win,overlap);
%% Train GMM1: features to 6 raters
total_train_feature_label = [cell2mat(ave_train), cell2mat(ave_train_feature)];
total_test_feature_label = [cell2mat(ave_test), cell2mat(ave_test_feature)];
%-------------------- Train GMM1 Parameters
gt_2d = total_train_feature_label;
gt_2d(isnan(gt_2d))=0;
gt_2d(isinf(gt_2d))=0;
n_gaussian = 8;
gmfit_train = fitgmdist(gt_2d,n_gaussian,'CovarianceType','full','RegularizationValue',0.0008,'Options',statset('MaxIter',1800));
%% Validate GMM1 using training features
for u = 1:9
data_dev = ave_train_feature{u};
noisy_train{u,1} = my_posterior_y_given_x6(n_gaussian, data_dev, gmfit_train.ComponentProportion,gmfit_train.mu, gmfit_train.Sigma);
noisy_train{u} = movmean(noisy_train{u},5);
end
%------------------------ Train evaluation
for u = 1:9
figure
subplot(2,1,1)
plot(noisy_train{u})
title("GMM Predict arousal based on 6 raters Utt "+u)
subplot(2,1,2)
plot(ave_train{u})
title("Training labels 6 raters")
end
corr(mean(cell2mat(ave_train)')',mean(cell2mat(noisy_train)')')
%% Validate GMM1 using test features
close all
for u = 1:9
data_dev = ave_test_feature{u};
noisy_test{u,1} = my_posterior_y_given_x6(n_gaussian, data_dev, gmfit_train.ComponentProportion,gmfit_train.mu, gmfit_train.Sigma);
noisy_test{u} = movmean(noisy_test{u},5);
end
for u = 1:9
figure
plot(mean(noisy_test{u}'))
title("GMM Predict arousal based on 6 raters Utt "+u)
hold on
plot(mean(ave_test{u}'))
legend("GMM","Ground Testing labels 6 raters")
end
corr(mean(cell2mat(ave_test)')',mean(cell2mat(noisy_test)')')
%% Train GMM2: noisy 6 to ground truth 6
[gmm2_final_train_data,gmm2_final_test_data] = gmm2_training_data(noisy_train,noisy_test,ave_train,ave_test);
n_gaussian = 4;
gmfit_obs = fitgmdist(gmm2_final_train_data,n_gaussian,'CovarianceType','full','RegularizationValue',0.001,'Options',statset('MaxIter',1800));
%% Validate GMM prediction with mean labels
for u = 1:9
data_test = (mean(noisy_test{u}')')*3;
p{u} = my_posterior_y_given_x(n_gaussian, data_test, gmfit_obs.ComponentProportion, gmfit_obs.mu, gmfit_obs.Sigma);
end
for u = 1:9
figure
plot(p{u})
hold on
plot(mean(ave_test{u}'))
end
bb = cell2mat(p);
bb(isnan(bb)) = 0;
bb2 = mean(cell2mat(ave_test)');
corr(bb2',bb')
%% Compute posterior covariance for observation probability
mu = gmfit_obs.mu;
sigma = gmfit_obs.Sigma;
mixweights = gmfit_obs.ComponentProportion;
for i = 1: n_gaussian
    sigi=squeeze(sigma(:,:,i));
    sigxx = sigi(2:end,2:end);
    sigxy = sigi(2:end,1);
    sigyy = sigi(1,1);
    sigyx = sigi(1,2:end);
    posterior_cov_x{i,1} = sigxx - sigxy*inv(sigyy)*sigyx; %xx - xy*inv(yy)*yx
end
%% Initial Pdf 
p_y0 = @(y) normpdf(y, 0,sqrt(1));             % initial pdf
%gen_y0 = @(y) normrnd(0, 1);     % sample from p_x0 (returns column vector)
xim = -0.01;
xmax = 0.01;
gen_y0 = @(Ns) -1 + 2*sum(rand(Ns,6),2)/6;
%% Observation Model
p_xi_given_yi = @(gmfit,yi,xi,gt_train,posterior_cov_x) my_observation_model(gmfit,yi,xi,gt_train,posterior_cov_x);
%% Separate memory
clear pf
T = size(ave_test{1},1);
nstate = 1;  % dimension of states, arousal label only, dim = 1
obs_dim = size(ave_test_feature{1},2); 
y = zeros(nstate, T); 
yh = zeros(nstate, T); 
pf.k               = 1;                   % initial iteration number
pf.Ns              = Ns;                 % number of particles
pf.w               = zeros(pf.Ns, T);     % weights
pf.particles       = zeros(nstate, pf.Ns, T); % particles
pf.gen_x0          = gen_y0;              % function for sampling from initial pdf p_x0
pf.p_yk_given_xk   = p_xi_given_yi;       % function of the observation likelihood PDF p(y[k] | x[k])
pf.p_x0 = p_y0;                          % initial prior PDF p(x[0])
%hyp_save = hyp2;
%% Start running SMC
intertval = linspace(-1,1,200);
T = length(ave_test_feature{1});
train_label = 0;
for u = 1:1
    x = noisy_test{u}*3;
    pred_state{u,1} = zeros(nstate, T); 
    for k = 1:T
       fprintf('Iteration = %d/%d\n',k,T);
       pf.k = k; 
        past_states = pred_state{u,1}(1:k-1);
       [pred_state{u,1}(k), pf] = run_6rater_SMC(my_sys, x(k,:), pf,..., 
       'systematic_resampling',hyp2,@infGaussLik,meanfunc,covfunc,likfunc,past_states,gmfit_obs,train_label,posterior_cov_x);   
    end
    pred_pf{u,1} = pf;
end
%% Predicted Distribution Evaluation & Analysis
for u = 1:9
    pf = pred_pf{u};
    for k = 1:T
         particles = pf.old_particle(1,:,k);
         weight = pf.old_w(:,k);
         MEAN{u,1}(k,1) = mean(particles*weight);
         MEAN2{u,1}(k,1) = mean(pf.particles(1,:,k)*pf.w(:,k));
         STD_pf{u,1}(k,1) = std(particles,weight);    
    end
end
%%
clear STD_ground
clear MSE_ground
for u = 1:9
    for i = 1:length(ave_test{1})
        STD_ground{u,1}(i,1) = std(ave_test{u}(i,:));
        MSE_ground{u}(i,1) = sum((ave_test{u}(i,:)-mean(ave_test{u}(i,:))).^2)/6;
    end
end
%% Plots of the Predicted Mean and Uncertainty Vs ground truth labels 
for u = 1:9
    MEAN_smooth{u,1} = movmean(MEAN{u},5)*0.45;
    STD_smooth{u} = movmean(STD_pf{u},5);
    MEAN_ground{u} = mean(ave_test{u}')';
    CC_mean_SMC(u) = corr(MEAN_ground{u},MEAN{u});
    CCC_mean_SMC(u) = ccc_calculation(MEAN_ground{u},MEAN{u});
    y =MEAN_smooth{u}';
    x = 1:numel(y);
    std_dev = STD_smooth{u}';
    curve1 = y + 1.5*std_dev;
    curve2 = y - 1.5*std_dev;
    x2 = [x, fliplr(x)];
    inBetween = [curve1, fliplr(curve2)];
    figure
    subplot(2,1,1)
    fill(x2, inBetween, [7 7 7]/8,'LineStyle','none');
    hold on;
    plot(x, y, 'r', 'LineWidth', 1);
    hold on
    plot(ave_test{u})
    if PF == 1
    title("Particle Filter Utt = " + u + " CC = " + cc_mean(u) + "CCC = " + CCC(u))
    else
        title("GMM Utt = " + u + " CC = " + cc_mean(u))
    end
    xlabel("Window index")
    ylabel("Arousal")
    subplot(2,1,2)
    plot(std_dev)
    hold on
    plot(STD_ground{u})
    title("Standard deviation")
    legend("Particle STD","Ground truth STD")
end
cc_mean_total = corr(cell2mat(MEAN_ground'),cell2mat(MEAN_smooth))
ccc_mean_total = ccc_calculation(cell2mat(MEAN_ground'),cell2mat(MEAN_smooth))
%% Scatter Plot of Ground truth SD Vs predicted SD
close all
for u = 1:9
STD_smooth{u} = movmean(STD_pf{u},5.5);
cc_std_SMC(u) = corr(STD_ground{u},STD_smooth{u});
ccc_std_SMC(u) = ccc_calculation(STD_ground{u},STD_smooth{u});
figure
subplot(2,1,1)
scatter(STD_pf{u},STD_ground{u})
xlim([0,0.6])
ylim([0,0.6])
xlabel("Particle STD")
ylabel("Ground STD")
title("Particle Filter CC of STD = "+ cc_pf(u)+" Utt = "+ u)

subplot(2,1,2)
plot(STD_smooth{u})
hold on
plot(STD_ground{u})
legend("Particle STD","Ground truth STD")
end
mean(cc_pf)
ccc_total_pf = ccc_calculation(cell2mat(STD_ground),cell2mat(STD_smooth'))
%% Save all the prediction datas 
% Further Evaluations people refers to generate_plot folder
save("predicted_particles.mat")