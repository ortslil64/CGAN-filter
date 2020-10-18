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
xlim([1,100]);

n_mc = 90;
mc_df = zeros(n_mc,100);
mc_pf = zeros(n_mc,100);
mc_direct = zeros(n_mc,100);
for ii = 1:n_mc
    mc_df(ii,:) = img_err_df(ii:ii+99).^2;
    mc_pf(ii,:) = img_err_pf(ii:ii+99).^2;
    mc_direct(ii,:) = img_err_direct(ii:ii+99).^2;
end
av_df = mean(mc_df, 1);
av_pf = mean(mc_pf, 1);
av_direct = mean(mc_direct, 1);

figure(2);
hold on;
plot(av_df,'b');
plot(av_pf,'k');
plot(av_direct,'r');
xlabel('k');
ylabel('mse');

%% ---- inverse_chekkers ---- %%
load('video filtering/inverse_chekkers/inverse_chekkers_data.mat')
figure(3);
hold on;
plot(img_err_df(30:end),'b');
plot(img_err_pf(30:end),'k');
plot(img_err_direct(30:end),'r');
xlabel('k');
ylabel('error');
xlim([1,100]);

n_mc = 180;
mc_df = zeros(n_mc,100);
mc_pf = zeros(n_mc,100);
mc_direct = zeros(n_mc,100);
for ii = 1:n_mc
    mc_df(ii,:) = img_err_df(ii:ii+99).^2;
    mc_pf(ii,:) = img_err_pf(ii:ii+99).^2;
    mc_direct(ii,:) = img_err_direct(ii:ii+99).^2;
end
av_df = mean(mc_df, 1);
av_pf = mean(mc_pf, 1);
av_direct = mean(mc_direct, 1);

figure(4);
hold on;
plot(av_df,'b');
plot(av_pf,'k');
plot(av_direct,'r');
xlabel('k');
ylabel('mse');

%% ---- partially_observed_chekkers ---- %%
load('video filtering/partially_observed_chekkers/partially_observed_chekkers_data2.mat')
figure(5);
hold on;
plot(img_err_df(1:end),'b');
plot(img_err_pf(1:end),'k');
plot(img_err_direct(1:end),'r');
xlabel('k');
ylabel('error');
xlim([1,100]);
function createfigure(YMatrix1)
n_mc = 180;
mc_df = zeros(n_mc,100);
mc_pf = zeros(n_mc,100);
mc_direct = zeros(n_mc,100);
for ii = 1:n_mc
    mc_df(ii,:) = img_err_df(ii:ii+99).^2;
    mc_pf(ii,:) = img_err_pf(ii:ii+99).^2;
    mc_direct(ii,:) = img_err_direct(ii:ii+99).^2;
end
av_df = mean(mc_df, 1);
av_pf = mean(mc_pf, 1);
av_direct = mean(mc_direct, 1);

figure(6);
hold on;
plot(av_df,'b');
plot(av_pf,'k');
plot(av_direct,'r');
xlabel('k');
ylabel('mse');

%% ---- partially_observed_tree ---- %%
load('video filtering/partially_observed_tree/partial_observation_tree_data4.mat')
figure(7);
hold on;
plot(img_err_df(10:140),'b');
plot(img_err_pf(10:140),'k');
plot(img_err_direct(10:140),'r');
xlabel('k');
ylabel('error');
xlim([1,100]);

n_mc = 40;
mc_df = zeros(n_mc,100);
mc_pf = zeros(n_mc,100);
mc_direct = zeros(n_mc,100);
for ii = 1:n_mc
    mc_df(ii,:) = img_err_df(ii:ii+99).^2;
    mc_pf(ii,:) = img_err_pf(ii:ii+99).^2;
    mc_direct(ii,:) = img_err_direct(ii:ii+99).^2;
end
av_df = mean(mc_df, 1);
av_pf = mean(mc_pf, 1);
av_direct = mean(mc_direct, 1);

figure(8);
hold on;
plot(av_df,'b');
plot(av_pf,'k');
plot(av_direct,'r');
xlabel('k');
ylabel('mse');