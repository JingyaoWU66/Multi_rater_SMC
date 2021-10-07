%% This file generate the plot of predicted uncertainty
load("predicted_particles.mat"); % Load the prediction results obtained from main_function_new.m
%%
PF = 1;
T = [260:500];
T = [170:360];
%T = [1:615];
for u = 4:4
    MM = MEAN_smooth;
    y =MM{u}(T)'; %pred_state{u}; % your mean vector;
    x = T;
    std_dev = STD{u}(T)';
    curve1 = y + 1.5*std_dev;
    curve2 = y - 1.5*std_dev;
    x2 = [x, fliplr(x)];
    inBetween = [curve1, fliplr(curve2)];
    figure
    fill(x2, inBetween, [7 7 7]/8,'LineStyle','none');
    hold on;
    plot(T,ave_test{u}(T,:),'LineWidth',1)
    xlabel('Second','FontSize', 11)
    ylabel('Arousal','FontSize', 11)
    %legend('Uncertainty','Rater 1','Rater 2','Rater 3','Rater 4','Rater 5','Rater 6','FontSize', 8)
    xlim([T(1),T(end)])
    legend('Predicted Uncertainty')
    %xt = get(gca, 'XTick');                                 % 'XTick' Values
    %set(gca, 'XTick', xt, 'XTickLabel', xt/2)
end
%%
T = [160:360];
L = [1:length(T)];
figure
plot(L/2,ave_test{2}(T,:),'--','LineWidth',1)
hold on
plot(L/2,mean(ave_test{2}(T,:)'),'r','LineWidth',2)
xlim([L(1),L(end)/2])
ylim([-1,0.65])
xlabel('Second')
ylabel('Arousal')