%% This file generate the Bar&Line plot of MSE at different deciles
load("predicted_particles.mat"); % Load the prediction results obtained from main_function_new.m
%%
clear CORR CCCORR MMSE save_inter_i
total_STD_ground = cell2mat(STD_ground);
total_STD_pf = cell2mat(STD_smooth');
k = 100;
k = [10:10:100]
for i = 1: length(k)
th(i) = prctile(total_STD_ground,k(i));
save_inter_i{i} = find(total_STD_ground <=th(i));
MMSE(i) = sum((gt(save_inter_i{i})-pred(save_inter_i{i})).^2)/length(save_inter_i{i})
end
figure
bar(k,MMSE,0.6)
ylim([0.07, 0.16])
xlabel("Ground truth SD")
ylabel("MSE")
%%
k = 10*[1:10]
for i = 1:length(k)
th(i) = prctile(total_STD_ground,k(i));
end
for i = 1:length(k)
    if i == 1
    index = find(total_STD_ground <=th(i));
    else
        index = find(total_STD_ground <=th(i) & total_STD_ground >th(i-1));
    end
    save_inter_i{i} = index;
MMSE(i) = sum((gt(save_inter_i{i})-pred(save_inter_i{i})).^2)/length(save_inter_i{i})
end
hold on
plot(k,MMSE,'-x','Color','k','LineWidth',2,'MarkerSize',10,'MarkerEdgeColor','r','MarkerFaceColor',[1 .6 .6])
xlabel("Ground truth SD")
ylabel("MSE")