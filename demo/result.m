clear;
close all;
clc;
%% ---- illusion ---- %%
load('video filtering/illusion/illusion_data.mat')
figure(1);
hold on;
plot(img_err_df(30:end),'b');
plot(img_err_pf(30:end),'k');
plot(img_err_direct(30:end),'r');
xlabel('k');
ylabel('error');
xlim([1,130]);

%% ---- inverse_chekkers ---- %%
load('video filtering/inverse_chekkers/inverse_chekkers_data.mat')
figure(2);
hold on;
plot(img_err_df(30:end),'b');
plot(img_err_pf(30:end),'k');
plot(img_err_direct(30:end),'r');
xlabel('k');
ylabel('error');
xlim([1,130]);

%% ---- partially_observed_chekkers ---- %%
load('video filtering/partially_observed_chekkers/partially_observed_chekkers_data.mat')
figure(3);
hold on;
plot(img_err_df(20:end),'b');
plot(img_err_pf(20:end),'k');
plot(img_err_direct(20:end),'r');
xlabel('k');
ylabel('error');
xlim([1,80]);


%% ---- partially_observed_tree ---- %%
load('video filtering/partially_observed_tree/partial_observation_tree_data.mat')
figure(4);
hold on;
plot(img_err_df(30:end),'b');
plot(img_err_pf(30:end),'k');
plot(img_err_direct(30:end),'r');
xlabel('k');
ylabel('error');
xlim([1,130]);