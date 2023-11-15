clear
basepath = '/mnt/probox/buzsakilab.nyumc.org/datasets/GirardeauG/Rat11/Rat11-20150330/';
cd(basepath);

%getting ripple times
temp_rip = readtable('Rat11-20150330.rip.evt','FileType','text');
%getting airpuff times
temp_airpuff = readtable('Rat11-20150330.puf.evt','FileType','text');
%getting times of task states
load('Rat11-20150330.task.states.mat')
%loading spikes
load('Rat11-20150330.spikes.cellinfo.mat')
%position
load('Rat11-20150330.pos.mat')

%onyl hpc cells
% spks_times = spikes.times(strcmp(spikes.region,'bla') | strcmp(spikes.region,'bmp'));
spks_times = spikes.times(strcmp(spikes.region,'hpc'));
%% preparing variables
start_idx = cellfun(@(x) strcmp(x,'start'),temp_rip.Var3);
stop_idx = cellfun(@(x) strcmp(x,'stop'),temp_rip.Var3);

start_t = temp_rip.Var1(start_idx)/1000; %its in ms
stop_t  = temp_rip.Var1(stop_idx)/1000;
rip_interv = [start_t stop_t];

%get pre sleep periods
pre_sleep_int = task.states.ints([2,3],:);
post_sleep_int = task.states.ints(5,:);
pre_run  = task.states.ints(1,:);
task_int = task.states.ints(4,:);
post_run = task.states.ints(6,:);

pre_sleep_rip_idx = ismember(start_t,Restrict(start_t,pre_sleep_int));
post_sleep_rip_idx = ismember(start_t,Restrict(start_t,post_sleep_int));

pre_rip = rip_interv(pre_sleep_rip_idx,:);
post_rip = rip_interv(post_sleep_rip_idx,:);

%% spikes during ripple

spks_rip_pre = cellfun(@(x) Restrict(x,pre_rip),spks_times,'UniformOutput',false);
spks_rip_post = cellfun(@(x) Restrict(x,post_rip),spks_times,'UniformOutput',false);

bin_size = 0.010; %10 ms to start
bins = linspace(pre_rip(1),pre_rip(end),(pre_rip(end)-pre_rip(1)) / bin_size);
binned_pre = cell2mat(cellfun(@(x) histcounts(x,bins),spks_rip_pre,'UniformOutput',false)')';

zsc_pre = zscore(binned_pre(sum(binned_pre,2)>0,:));

bins = linspace(post_rip(1),post_rip(end),(post_rip(end)-post_rip(1)) / bin_size);
binned_post = cell2mat(cellfun(@(x) histcounts(x,bins),spks_rip_post,'UniformOutput',false)')';

zsc_post = zscore(binned_post(sum(binned_post,2)>0,:));

%% gcPCA
[B, S, X] = gcPCA(zsc_post,zsc_pre, 4.1);

%% does component encode puff location?
time_airpuff = temp_airpuff.Var1/1000;
bin_size = 0.1;
bins = linspace(task_int(1),task_int(2),diff(task_int)/bin_size);
spk_count =  cell2mat(cellfun(@(x) histcounts(x,bins),spks_times,'UniformOutput',false)')';
zsc_count = zscore(spk_count);
projector = (X(:,1)*X(:,1)');
projector = projector - diag(diag(projector));

time_projection = [];
for ntime=1:size(zsc_count,1)
    
    time_projection(ntime)=(zsc_count(ntime,:)*projector*zsc_count(ntime,:)');
    
end



figure;
h1=subplot(2,1,1);plot(pos.X.t,pos.X.data)
hold on
plot(time_airpuff,400,'^r')
xlim(task_int)
grid on

h2=subplot(2,1,2);
plot(bins(1:end-1),zsc_count*X(:,1))
% plot(bins(1:end-1),time_projection)
xlim(task_int)
grid on
linkaxes([h1,h2],'x')

% %% should I don run vs pre instead?
% %
% %
% 
% %doing only during running?
% pre_idx = ismember(pos.linSpd.t,Restrict(pos.linSpd.t,pre_run));
% linspd = pos.linSpd.data(pre_idx);
% linspd_t = pos.linSpd.t(pre_idx);
% pre_run_t = linspd_t(linspd>2)';
% 
% task_idx = ismember(pos.linSpd.t,Restrict(pos.linSpd.t,task_int));
% linspd = pos.linSpd.data(task_idx);
% linspd_t = pos.linSpd.t(task_idx);
% task_t = linspd_t(linspd>2)';
% 
% 
% bin_size = 0.1;
% bins = linspace(pre_run(1),pre_run(2),diff(pre_run)/bin_size);
% pre_spk_count =  cell2mat(cellfun(@(x) histcounts(x,bins),spks_times,'UniformOutput',false)')';
% 
% %getting only the bins where running speed is above 2
% temp = interp1(bins(1:end-1),1:length(bins(1:end-1)),pre_run_t);
% pre_run_idx = unique(round(temp));
% pre_run_idx = pre_run_idx(~isnan(pre_run_idx));
% 
% 
% bins = linspace(task_int(1),task_int(2),diff(task_int)/bin_size);
% task_spk_count =  cell2mat(cellfun(@(x) histcounts(x,bins),spks_times,'UniformOutput',false)')';
% 
% %getting only the bins where running speed is above 2
% temp = interp1(bins(1:end-1),1:length(bins(1:end-1)),task_t);
% task_idx = unique(round(temp));
% task_idx = task_idx(~isnan(task_idx));
% 
% cells2keep = sum(pre_spk_count(pre_run_idx,:),1)>0 & sum(task_spk_count(task_idx,:),1)>0;
% 
% zsc_pre = zscore(pre_spk_count(pre_run_idx,cells2keep));
% zsc_task = zscore(task_spk_count(task_idx,cells2keep));
% zsc_task2 = zscore(task_spk_count(:,cells2keep));
% 
% [B, S, X] = gcPCA(zsc_task,zsc_pre, 4.1);

%% testing run vs pre
%

bin_size = 0.1;
bins_pre = linspace(pre_run(1),pre_run(2),diff(pre_run)/bin_size);
pre_spk_count =  cell2mat(cellfun(@(x) histcounts(x,bins_pre),spks_times,'UniformOutput',false)')';

bins_task = linspace(task_int(1),task_int(2),diff(task_int)/bin_size);
task_spk_count =  cell2mat(cellfun(@(x) histcounts(x,bins_task),spks_times,'UniformOutput',false)')';

cells2keep = sum(pre_spk_count,1)>0 & sum(task_spk_count,1)>0;

zsc_pre = zscore(pre_spk_count(:,cells2keep));
zsc_task = zscore(task_spk_count(:,cells2keep));

bins = linspace(post_run(1),post_run(2),diff(post_run)/bin_size);
post_spk_count =  cell2mat(cellfun(@(x) histcounts(x,bins),spks_times,'UniformOutput',false)')';
zsc_post = zscore(post_spk_count(:,cells2keep));

[B, S, X] = gcPCA(zsc_task,zsc_pre, 4.1);
%% plotting results

% projector = (X(:,1)*X(:,1)');
% projector = projector - diag(diag(projector));

% time_projection = [];
% for ntime=1:size(zsc_task,1)
%     
%     time_projection(ntime)=(zsc_task(ntime,:)*projector*zsc_task(ntime,:)');
%     
% end

figure;
h1=subplot(2,1,1);plot(pos.X.t,pos.X.data)
hold on
plot(time_airpuff,400,'^r')
xlim(task_int)
grid on

h2=subplot(2,1,2);
plot(bins_task(1:end-1),zsc_task*X(:,1))
% plot(bins(1:end-1),time_projection)
xlim(task_int)
grid on
linkaxes([h1,h2],'x')


%% do a 2d histogram of vector activity
% [U,S,V] = svd(zsc_task,'econ');
proj_task = zsc_task*X(:,1);
task_idx = ismember(pos.linSpd.t,Restrict(pos.linSpd.t,task_int));
x = pos.X.data(task_idx);
y = pos.Y.data(task_idx);
linspd_t = pos.linSpd.t(task_idx);

interp_task = interp1(bins_task(1:end-1),proj_task,linspd_t);
interp_task(isnan(interp_task))=0;

r_c = normalize(interp_task,'range');
b_c = 1-normalize(interp_task,'range');
figure;
% plot(x,y,'k')
% hold on

% F = scatteredInterpolant(x,y,smooth(interp_task,25));
% xp = linspace(min(x),max(x),2000);
% yp = linspace(min(y),max(y),2000);
% [XP,YP]=meshgrid(xp,yp);

% VIP = F(XP,YP);
% interp_2d = interp2(x,y,interp_task,,);
% for a =1:2000%length(interp_task)
%     
%     scatter(x,y,abs(round(interp_task(a)*20))+1,[r_c(a) 0 b_c(a)],'filled',...
%         'markeredgecolor','none','markerfacealpha',0.2)
% end



plot3(x,y,smooth(interp_task,30))

%% plotting just trajectories from one side to the other (danger/safe)
side1_temp = diff(smooth(x,100))>0.3;
side2_temp = diff(smooth(x,100))<-0.3;

%identifying proper running over noise
temp_st = find(diff(side1_temp)==1);
temp_sp = find(diff([side1_temp; 0])==-1);

temp = find([temp_sp - temp_st]>80);
index = 1:length(x);
side1_int = [index(temp_st(temp)+1); index(temp_sp(temp)+1)];

%identifying proper running over noise
temp_st = find(diff(side2_temp)==1);
temp_sp = find(diff([side2_temp; 0])==-1);

temp = find([temp_sp - temp_st]>80);
index = 1:length(x);
side2_int = [index(temp_st(temp)+1); index(temp_sp(temp)+1)];
%% TO PLOT
[U,S,V] = svd(zsc_task,'econ');
proj_task = zsc_task*X(:,1:2);
task_idx = ismember(pos.linSpd.t,Restrict(pos.linSpd.t,task_int));

%getting airpuf
task_t = pos.linSpd.t(task_idx);
air_puff_idx = round(interp1(task_t,1:length(task_t),time_airpuff));

interp_task = interp1(bins_task(1:end-1),proj_task,linspd_t);
interp_task(isnan(interp_task))=0;

gcpc1 = smooth(interp_task(:,1),30);
gcpc2 = smooth(interp_task(:,2),30);
% gcpc1 = interp_task(:,1);
% gcpc2 = interp_task(:,2);
darkblue = [0 0 0.6];
darkred  = [0.6 0 0];
figure;


for a = 2:21
   
   plot(gcpc1(side1_int(1,a):side1_int(2,a)),gcpc2(side1_int(1,a):side1_int(2,a)),'r','linewidth',0.5) 
   hold on
   plot(gcpc1(side1_int(1,a)),gcpc2(side1_int(1,a)),'o','markerfacecolor',darkred,'markeredgecolor',darkred) 
   plot(gcpc1(side1_int(2,a)),gcpc2(side1_int(2,a)),'^','markerfacecolor',darkred,'markeredgecolor',darkred) 
%    auxplot = air_puff_idx(ismember(air_puff_idx,Restrict(air_puff_idx,side1_int(:,a)')));
%    plot(gcpc1(auxplot),gcpc2(auxplot),'s','markerfacecolor','k','markeredgecolor','k') 
   
end

figure;
for a = 2:21
   plot(gcpc1(side2_int(1,a):side2_int(2,a)),gcpc2(side2_int(1,a):side2_int(2,a)),'b','linewidth',0.5) 
   hold on
   plot(gcpc1(side2_int(1,a)),gcpc2(side2_int(1,a)),'o','markerfacecolor',darkblue,'markeredgecolor',darkblue) 
   plot(gcpc1(side2_int(2,a)),gcpc2(side2_int(2,a)),'^','markerfacecolor',darkblue,'markeredgecolor',darkblue) 
   
   %finding the location of the threat
   [~,I] = min(abs(258-x(side2_int(1,a):side2_int(2,a))));
   plot(gcpc1(side2_int(1,a)+I),gcpc2(side2_int(1,a)+I),'s','markerfacecolor','g','markeredgecolor','g') 
end
%it seems to me gcPCA finds a vector that disentangle safe vs dangerous
%zones

%plot with color for dangerous trajectories vs safe trajectories.
%plot also the location of air puff

%do these analysis for many sessions to see if it holds

%how many gcPCA to use? we haven't defined a null yet

%more variability in the safe spot than in the other one? I've seeside2_temp = diff(smooth(x,60))<-0.11;n that in
%multiple


