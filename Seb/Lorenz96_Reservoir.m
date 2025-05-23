%% Parameters

% Model 
N_system = 8;

F_training = 3.50;
F_pert = 2.50;
%F_pert = 3.12;
%F_pert = 4.00; % This one is a little complicated, seemingly due to multistability?

% Time vector
T0 = 5000;
T = 100;
dt = 0.01;                  
t = 0 : dt : T;
nt = numel(t);

tau = 1;
alpha = 10;

% training forcing functions
A = 1.0;
force_training = A*cos((0.1+0.1*rand(N_system,1))*t);

%% Transient

% Initial condition
x0 = rand(N_system,1) - 0.5;

nt0 = T0/dt;

for i = 1 : nt0
    x0p1 = circshift(x0,-1);
    x0m1 = circshift(x0,1);
    x0m2 = circshift(x0,2);

    f = (x0p1 - x0m2).*x0m1 - x0 + F_training;

    xp = x0 + dt*f;

    xpp1 = circshift(xp,-1);
    xpm1 = circshift(xp,1);
    xpm2 = circshift(xp,2);

    g = (xpp1 - xpm2).*xpm1 - xp + F_training;

    x0 = x0 + (dt/2)*(f + g);
end

%% Initialize solution vector

x_target = zeros(N_system, nt);

x_training = zeros(N_system, nt);

x_unknown = zeros(N_system, nt);

x_control = zeros(N_system, nt);

%% Target Dynamics

x_target(:,1) = x0;

for i = 1:nt-1
    xp1 = circshift(x_target(:,i),-1);
    xm1 = circshift(x_target(:,i),1);
    xm2 = circshift(x_target(:,i),2);

    f = (xp1 - xm2).*xm1 - x_target(:,i) + F_training;

    xp = x_target(:,i) + dt*f;

    xpp1 = circshift(xp,-1);
    xpm1 = circshift(xp,1);
    xpm2 = circshift(xp,2);

    g = (xpp1 - xpm2).*xpm1 - xp + F_training;

    x_target(:,i+1) = x_target(:,i) + (dt/2)*(f + g);
end

%% Reservoir Computing Parameters

% Size of the Reservoir
N = 200;

% Dimension of input
D = N_system;

% Mean degree of the reservoir
km = 6;

% Spectral radius of the reservoir
a_param = 0.95;

% Input parameter
b = 0.01;

% Ridge Regression Parameter
lambda = 1e-8;

% Generate random matrices Adjacency matrix for Reservoir
A = zeros(N, N);
M = N * km;
for i = 1:M
    row = randi(N);
    col = randi(N);
    while A(row, col) == 1 || row == col
        row = randi(N);
        col = randi(N);
    end
    A(row,col) = 2*rand() - 1;
end
A = A/abs(eigs(A,1));

% Input matrix
B = 2 * rand(N,D) - 1;

%% Reservoir states

r_training = zeros(N,nt);
r_training(:,1) = 2.0*rand(N,1)-1;
r_unknown = zeros(N,nt);
r_control = zeros(N,nt);

%% Train Reservoir

x_training(:,1) = x_target(:,end);

for i = 1:nt-1
    xp1 = circshift(x_training(:,i),-1);
    xm1 = circshift(x_training(:,i),1);
    xm2 = circshift(x_training(:,i),2);

    f = (xp1 - xm2).*xm1 - x_training(:,i) + F_training + force_training(:,i);

    xp = x_training(:,i) + dt*f;

    xpp1 = circshift(xp,-1);
    xpm1 = circshift(xp,1);
    xpm2 = circshift(xp,2);

    g = (xpp1 - xpm2).*xpm1 - xp + F_training + force_training(:,i+1);

    x_training(:,i+1) = x_training(:,i) + (dt/2)*(f + g);

    % Step reservoir
    r_training(:,i+1) = tanh(a_param*A*r_training(:,i) + b*B*x_training(:,i) + 1);
end

%% Find output with redge regression

Wout = force_training*(r_training')*inv(r_training*(r_training') + lambda*eye(N,N));
Forcing_training_recovered = Wout*r_training;

%% Another Transient

x0 = x_training(:,end);

rp = r_training(:,end);

nt0 = T0/dt;

for i = 1 : nt0
    x0p1 = circshift(x0,-1);
    x0m1 = circshift(x0,1);
    x0m2 = circshift(x0,2);

    f = (x0p1 - x0m2).*x0m1 - x0 + F_pert;

    xp = x0 + dt*f;

    xpp1 = circshift(xp,-1);
    xpm1 = circshift(xp,1);
    xpm2 = circshift(xp,2);

    g = (xpp1 - xpm2).*xpm1 - xp + F_pert;

    x0 = x0 + (dt/2)*(f + g);
end

%% Recovery Phase

r_unknown(:,1) = rp;
x_unknown(:,1) = x0;

up = Wout*rp;
vp = up;

for i = 1:nt-1
    xp1 = circshift(x_unknown(:,i),-1);
    xm1 = circshift(x_unknown(:,i),1);
    xm2 = circshift(x_unknown(:,i),2);

    f = (xp1 - xm2).*xm1 - x_unknown(:,i) + F_pert;

    xp = x_unknown(:,i) + dt*f;

    xpp1 = circshift(xp,-1);
    xpm1 = circshift(xp,1);
    xpm2 = circshift(xp,2);

    g = (xpp1 - xpm2).*xpm1 - xp + F_pert;

    x_unknown(:,i+1) = x_unknown(:,i) + (dt/2)*(f + g);

    up = Wout*r_unknown(:,i);
    vp = vp + (dt/tau)*(up - vp);

    % Step reservoir
    r_unknown(:,i+1) = tanh(a_param*A*r_unknown(:,i) + b*B*x_unknown(:,i) + 1);
end

%% Find output with redge regression

Forcing_unknown_recovered = Wout*r_unknown;

%% Control Phase

r_control(:,1) = r_unknown(:,end);
x_control(:,1) = x_unknown(:,end);

u = zeros(D,nt);
v = zeros(D,nt);

% u(:,1) = Wout*r_unknown(:,end);
% v(:,1) = u(:,1);
u(:,1) = up;
v(:,1) = vp;

for i = 1:nt-1
    u(:,i+1) = Wout*r_control(:,i);
    v(:,i+1) = v(:,i) + (dt/tau)*(u(:,i) - v(:,i));

    xp1 = circshift(x_control(:,i),-1);
    xm1 = circshift(x_control(:,i),1);
    xm2 = circshift(x_control(:,i),2);

    f = (xp1 - xm2).*xm1 - x_control(:,i) + F_pert - alpha*v(:,i);

    xp = x_control(:,i) + dt*f;

    xpp1 = circshift(xp,-1);
    xpm1 = circshift(xp,1);
    xpm2 = circshift(xp,2);

    g = (xpp1 - xpm2).*xpm1 - xp + F_pert - alpha*v(:,i);

    x_control(:,i+1) = x_control(:,i) + (dt/2)*(f + g);

    % Step reservoir
    r_control(:,i+1) = tanh(a_param*A*r_control(:,i) + b*B*x_control(:,i) + 1);
end

%% Plotting

% figure; hold on;
% plot(t,f_x_training, 'b-', 'LineWidth', 1.5);
% plot(t,f_y_training, 'r-', 'LineWidth', 1.5);
% plot(t,f_z_training, 'g-', 'LineWidth', 1.5,'Color',[0 0.5 0]);
% plot(t,f_x_training_recovered, 'k--', 'LineWidth', 1.5);
% plot(t,f_y_training_recovered, 'k--', 'LineWidth', 1.5);
% plot(t,f_z_training_recovered, 'k--', 'LineWidth', 1.5);
% hold off;
% grid on;
% box on;
% %axis square;
% %xlim([0 T]);
% %ylim([0 4.5]);
% %legend(['\beta = ',num2str(beta(j)),', ER'],'FontSize',14,'FontName','Helvetica','Location','NorthWest');
% xlabel('time, t','FontSize',16,'FontName','Helvetica');
% ylabel('forcing','FontSize',16,'FontName','Helvetica');
% set(gca,'FontSize',16,'FontName','Helvetica');
% %set(gca,'XTick',[0 0.2 0.4 0.6 0.8 1]);
% %set(gca,'YTick',[]);
% %set(gca,'XTickLabel',{'';'';''});
% %set(gca,'YTickLabel',{'-\pi';'';'0';'';'\pi';'3\pi/2';'2\pi'});
% %set(gca,'TickLength',[0.025 0.025]);
% 
% figure; hold on;
% plot(t,f_x_unknown_recovered, 'b-', 'LineWidth', 1.5);
% plot(t,f_y_unknown_recovered, 'r-', 'LineWidth', 1.5);
% plot(t,f_z_unknown_recovered, 'g-', 'LineWidth', 1.5,'Color',[0 0.5 0]);
% hold off;
% grid on;
% box on;
% %axis square;
% %xlim([0 T]);
% %ylim([0 4.5]);
% %legend(['\beta = ',num2str(beta(j)),', ER'],'FontSize',14,'FontName','Helvetica','Location','NorthWest');
% xlabel('time, t','FontSize',16,'FontName','Helvetica');
% ylabel('forcing','FontSize',16,'FontName','Helvetica');
% set(gca,'FontSize',16,'FontName','Helvetica');
% %set(gca,'XTick',[0 0.2 0.4 0.6 0.8 1]);
% %set(gca,'YTick',[]);
% %set(gca,'XTickLabel',{'';'';''});
% %set(gca,'YTickLabel',{'-\pi';'';'0';'';'\pi';'3\pi/2';'2\pi'});
% %set(gca,'TickLength',[0.025 0.025]);

% figure; hold on;
% plot3(x_unknown(ceil(nt/2):end),y_unknown(ceil(nt/2):end),z_unknown(ceil(nt/2):end), 'r-', 'LineWidth', 1.5);
% plot3(x_target(ceil(nt/2):end),y_target(ceil(nt/2):end),z_target(ceil(nt/2):end), 'k-', 'LineWidth', 1.5);
% plot3(x_control(ceil(nt/2):end),y_control(ceil(nt/2):end),z_control(ceil(nt/2):end), 'b-', 'LineWidth', 1.5);
% hold off;
% grid on;
% box on;
% %axis square;
% %xlim([0 T]);
% %ylim([0 4.5]);
% %legend(['\beta = ',num2str(beta(j)),', ER'],'FontSize',14,'FontName','Helvetica','Location','NorthWest');
% xlabel('x','FontSize',16,'FontName','Helvetica');
% ylabel('y','FontSize',16,'FontName','Helvetica');
% zlabel('z','FontSize',16,'FontName','Helvetica');
% set(gca,'FontSize',16,'FontName','Helvetica');
% %set(gca,'XTick',[0 0.2 0.4 0.6 0.8 1]);
% %set(gca,'YTick',[]);
% %set(gca,'XTickLabel',{'';'';''});
% %set(gca,'YTickLabel',{'-\pi';'';'0';'';'\pi';'3\pi/2';'2\pi'});
% %set(gca,'TickLength',[0.025 0.025]);

figure; hold on;
plot(x_unknown(1,ceil(nt/2):end),x_unknown(2,ceil(nt/2):end), 'r-', 'LineWidth', 1.5);
plot(x_target(1,ceil(nt/2):end),x_target(2,ceil(nt/2):end), 'k-', 'LineWidth', 1.5);
plot(x_control(1,ceil(nt/2):end),x_control(2,ceil(nt/2):end), 'b-', 'LineWidth', 1.5);
% plot(x_unknown,y_unknown, 'r-', 'LineWidth', 1.5);
% plot(x_target,y_target, 'k-', 'LineWidth', 1.5);
% plot(x_control,y_control, 'b-', 'LineWidth', 1.5);
hold off;
grid on;
box on;
axis equal;
xlim([-2.4 5.5]);
ylim([-2.6 6.2]);
%legend('perturbed','target','controlled','FontSize',14,'FontName','Helvetica','Location','NorthWest');
xlabel('x_{1}','FontSize',16,'FontName','Helvetica');
ylabel('x_{2}','FontSize',16,'FontName','Helvetica');
set(gca,'FontSize',16,'FontName','Helvetica');
set(gca,'XTick',[-2 0 2 4]);
set(gca,'YTick',[-2 0 2 4 6]);
%set(gca,'XTickLabel',{'';'';''});
%set(gca,'YTickLabel',{'-\pi';'';'0';'';'\pi';'3\pi/2';'2\pi'});
%set(gca,'TickLength',[0.025 0.025]);