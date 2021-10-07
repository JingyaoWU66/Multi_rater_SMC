%% This file generate the Satter plot of predicted standard deviation (SD) Vs 6 rater SD
load("predicted_particles.mat"); % Load the prediction results obtained from main_function_new.m
%%
x = [0:0.01:1]
y = x;
figure
plot(x,y,'--','LineWidth',2)
hold on
scatter(STD_pf{2},STD_ground{2},10,'filled','MarkerFaceColor',[0 .7 .7])
xlim([0,0.55])
ylim([0,0.55])
xlabel("Particle SD")
ylabel("Multi-rater SD")